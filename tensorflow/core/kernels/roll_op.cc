/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "roll_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/register_types_traits.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// num_eff_dims - dimensions are reduced to the number of dimensions that are
//    shifted plus the first dimension (0) regardless of whether it's shifted
//    or not. Unused dimensions are flattend to their parent/outer dimension.
//    num_eff_dims is the number of effective dimensions that were not reduced.
// eff_shift - effective shift: the amount of shift in the flattened tensor
//    for a given effective dimension.
// eff_range - the number of indices in the flattened tensor that cover the
//    given effective dimension. For example if you have a 3D tensor:
//        v tensor with shape = [2,3,4]
//    [[[x,x,x,x], [x,x,x,x], [x,x,x,x]],   [[x,x,x,x], [x,x,x,x], [x,x,x,x]]]
//        v tensor flattened
//    [  x,x,x,x,   x,x,x,x,   x,x,x,x,       x,x,x,x    x,x,x,x    x,x,x,x  ]
//      --------- < eff_range[2] == 4
//     --------------------------------  < eff_range[1] == 12
//    ------------------------------------------------------------------------
//                                ^ eff_range[0] == 24
template <typename T>
void DoRoll(OpKernelContext* context, const int64 num_elements,
            const int num_eff_dims,
            const T* input, T* output,
            const gtl::ArraySlice<int64>& eff_threshold,
            const gtl::ArraySlice<int64>& eff_range) {
  auto work = [input, output, num_eff_dims, &eff_threshold, &eff_range](
                  int64 start, int64 end) {
    // offset - the shift along the flattened tensor for current element
    int offset = 0;
    // initialize indices and offset
    for (int i = 0; i < num_eff_dims; i++) {
      if (start % eff_range[i] < eff_threshold[i]) {
        // range - threshold = shift
        offset += eff_range[i] - eff_threshold[i];
      } else {
        offset -= eff_threshold[i];
      }
    }

    for (int64 i = start; i < end; i++) {
      output[i+offset] = input[i];
      for (int j = num_eff_dims - 1; j >= 0; j--) {
        const int idx = (i + 1) % eff_range[j];
        if (idx != 0) {
          if (idx == eff_threshold[j]) {
            // we've reached the threshold
            // to wrap around our offset for this dimension needs to be
            // -threshold instead of +shift
            // range = threshold + shift
            // offset = shift + ... other offsets
            // offset - range = -threshold + ... other offsets
            // thus we undo our previous shift as well as add a new offset of
            // -threshold[j] in one operation
            offset -= eff_range[j];  // now wraps around
          }
          break;                         // idx != 0 don't need to carry
        } else {
          // idx became 0
          // undo wrap around and add back shift in one operation
          offset += eff_range[j];
        }
      }
    }
  };
  // Shard
  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  // 15 - expiramentally determined with float and bool types
  const int cost_per_element = 15 * sizeof(T);  // rough estimate
  Shard(worker_threads->num_threads, worker_threads->workers, num_elements,
        cost_per_element, std::move(work));
}

// num_eff_dims - dimensions are reduced to the number of dimensions that are
//    shifted plus the first dimension (0) regardless of whether it's shifted
//    or not. Unused dimensions are flattend to their parent/outer dimension.
//    num_eff_dims is the number of effective dimensions that were not reduced.
// eff_shift - effective shift: the amount of shift in the flattened tensor
//    for a given effective dimension.
// eff_range - the number of indices in the flattened tensor that cover the
//    given effective dimension. For example if you have a 3D tensor:
//        v tensor with shape = [2,3,4]
//    [[[x,x,x,x], [x,x,x,x], [x,x,x,x]],   [[x,x,x,x], [x,x,x,x], [x,x,x,x]]]
//        v tensor flattened
//    [  x,x,x,x,   x,x,x,x,   x,x,x,x,       x,x,x,x    x,x,x,x    x,x,x,x  ]
//      --------- < eff_range[2] == 4
//     --------------------------------  < eff_range[1] == 12
//    ------------------------------------------------------------------------
//                                ^ eff_range[0] == 24
// isd - inner shift dimension
template <typename T>
// Use memcpy to copy memory in groups when the data type supports memcpy
void DoRollWithMemcpy(OpKernelContext* context, const int64 num_elements,
                      const int num_eff_dims,
                      const T* input, T* output,
                      const gtl::ArraySlice<int64>& eff_threshold,
                      const gtl::ArraySlice<int64>& eff_range) {
  const int isd = num_eff_dims - 1;
  auto work = [input, output, num_eff_dims, &eff_threshold, &eff_range, isd](
                  int64 start, int64 end) {
    // the number of indices over in the flattened tensor you need to skip in
    // order to make it over from one side of the isd to the other
    const int64 isd_range = std::max<int>(eff_range[isd], 1);
    // the distance along the flattend tensor to the next element in the isd

    // start and end represent the i-th group currently so we will convert
    // them into numbers representing the i-th elements.
    // there are 2 groups per isd one for all elements before threshold[isd]
    // and another for all elements after threshold[isd].
    const int64 start_remainder = (start % 2) * eff_threshold[isd];
    const int64 end_remainder = (end % 2) * eff_threshold[isd];
    start = (start / 2) * isd_range + start_remainder;
    end = (end / 2) * isd_range + end_remainder;

    const T* in_ptr = &input[0];
    T* out_ptr = &output[0];
    in_ptr += start;
    out_ptr += start;

    // initialize indices
    for (int i = 0; i < num_eff_dims; i++) {
      if (start % eff_range[i] < eff_threshold[i]) {
        // range - threshold = shift
        out_ptr += eff_range[i] - eff_threshold[i];
      } else {
        out_ptr -= eff_threshold[i];
      }
    }

    // the size of the next group
    int64 group_size = 0;
    int64 i = start;
    const int64 eff_shift_isd = eff_range[isd] - eff_threshold[isd];
    while (i < end) {
      const int64 i_mod_range = i % eff_range[isd];
      if (i_mod_range < eff_threshold[isd]) {
        group_size = eff_threshold[isd];
      } else {
        group_size = eff_shift_isd;
      }
      // copy group of elements
      memcpy(out_ptr, in_ptr, group_size * sizeof(T));

      // shift i and the pointers over to the next group position
      i += group_size;
      out_ptr += group_size;
      in_ptr += group_size;

      for (int j = isd; j >= 0; j--) {
        const int idx = i % eff_range[j];
        if (idx != 0) {
          if (idx == eff_threshold[j]) {
            out_ptr -= eff_range[j];  // now wraps around
          }
          break; // idx != 0 don't need to carry
        } else {
          // idx became 0 so reverse wrap around
          out_ptr += eff_range[j];
        }
      }
    }
  };
  // Shard
  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  const int64 group_size = eff_range[isd];
  const int total_work = 2 * num_elements / std::max<int>(eff_range[isd], 1);
  // 25 - expiramentally determined with float and bool types
  const int cost_per_group = 15 * sizeof(T) * group_size;
  Shard(worker_threads->num_threads, worker_threads->workers, total_work,
        cost_per_group, std::move(work));
}

template <typename Device, typename T, typename Tshift, typename Taxis>
class RollOp : public OpKernel {
 public:
  explicit RollOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input = context->input(0);
    const Tensor& shift = context->input(1);
    const Tensor& axis = context->input(2);

    auto shift_flat = shift.flat<Tshift>();
    auto axis_flat = axis.flat<Taxis>();

    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(input.shape()),
                errors::InvalidArgument("input must be 1-D or higher"));
    OP_REQUIRES(context, shift.shape().dims() <= 1,
                errors::InvalidArgument(
                    "shift must be a scalar or a 1-D vector. Found: ",
                    shift.shape().DebugString()));
    OP_REQUIRES(context, axis.shape().dims() <= 1,
                errors::InvalidArgument(
                    "axis must be a scalar or a 1-D vector. Found: ",
                    axis.shape().DebugString()));
    OP_REQUIRES(
        context, shift.shape() == axis.shape(),
        errors::InvalidArgument("shift and axis must have the same size"));
    const int64 num_elements = input.NumElements();
    const int num_shifts = static_cast<int>(shift_flat.size());
    const int num_dims = input.dims();

    // if there are any duplicate axes, shift_mod_sum will have the
    // total modulo sum of shifts for each dimension
    gtl::InlinedVector<int, 4> shift_mod_sum(num_dims, 0);
    for (int i = 0; i < num_shifts; i++) {
      int axis = axis_flat(i);
      if (axis < 0) {
        axis += num_dims;
      }
      OP_REQUIRES(context, FastBoundsCheck(axis, num_dims),
                  errors::InvalidArgument("axis ", axis, " is out of range"));
      const int ds = std::max<int>(static_cast<int>(input.dim_size(axis)), 1);
      const int sum = shift_mod_sum[axis] + static_cast<int>(shift_flat(i));
      // modulo that works with negatives: ((x % y) + y) % y
      shift_mod_sum[axis] = (sum % ds + ds) % ds;
    }

    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    if (std::is_same<Device, GPUDevice>::value) {
      alloc_attr.set_gpu_compatible(true);
    }

    // the size of each dimension
    gtl::InlinedVector<int, 4> dim_size(num_dims);
    // threshold[i] is the index that the roll starts to wrap back to the front
    gtl::InlinedVector<int, 4> threshold(num_dims);
    // dim_range is the number of indices over in the flattened tensor
    // you need to skip in order to make it over from one side of a dimension
    // to the other. Used to make the shifts wrap around after a threshold.
    gtl::InlinedVector<int64, 4> dim_range(num_dims);
    int64 dim_size_prod = 1;  // dimension size product
    for (int i = num_dims - 1; i >= 0; i--) {
      const int ds = std::max<int32>(static_cast<int32>(input.dim_size(i)), 1);
      dim_size[i] = ds;
      threshold[i] = (ds - shift_mod_sum[i]) % ds;
      dim_size_prod *= static_cast<int64>(input.dim_size(i));
      dim_range[i] = dim_size_prod;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto input_flat = input.flat<T>().data();
    auto output_flat = output->flat<T>().data();

    // count the effective dimentions and determine if we want to run on gpu
    int num_eff_dims = 0;
    for (size_t i = 0; i < num_dims; i++) {
      if (threshold[i] != 0 || i == 0) {
        num_eff_dims++;
      }
    }

    bool can_do_gpu = num_eff_dims <= MAX_DIM_GPU && num_elements <= kint32max;
    if (std::is_same<Device, GPUDevice>::value && can_do_gpu) {
      // run on GPU

      // reduce the tensor dimensions to the dimensions that will get shifted.
      // unshifted dimensions will get flattened to an outer shifted dimension
      // or the first dimension (outer most dimension) if there are no
      // outer shifted dimensions.
      // num_eff_dims = number of efective dimensions
      // eff_dims = array of original indicies of effective dimensions
      gtl::InlinedVector<int, 4> eff_dims(num_eff_dims);
      // always keeping the fist dimension regardless of whether it's shifted
      eff_dims[0] = 0;
      for (size_t i = 1; i < num_eff_dims; i++) {
        // search after the previous effective dimension
        size_t j = eff_dims[i-1] + 1;
        // when threshold[j] == 0, there is no shift, so skip this dimension
        while (threshold[j] == 0 && j < num_dims) {
          j++;
        }
        eff_dims[i] = j;
      }
      // eff_shift = effective shift - the amount elements will shift in the
      // flattend tensor for a given effective dimension.
      Eigen::array<int64, MAX_DIM_GPU> eff_shift;
      Eigen::array<int64, MAX_DIM_GPU> eff_range;
      Eigen::array<int64, MAX_DIM_GPU> eff_size;

      size_t last_idx = (num_eff_dims-1 > 0) ? num_eff_dims-1 : 0;
      size_t last_eff_dim = eff_dims[last_idx];

      eff_range[last_idx] = dim_range[last_eff_dim];
      eff_size[last_idx] = dim_range[last_eff_dim];
      if (last_eff_dim+1 < num_dims) {
        const int64 stride = dim_range[last_eff_dim+1];
        const int64 eff_threshold = threshold[last_eff_dim] * stride;
        eff_shift[last_idx] = dim_range[last_eff_dim] - eff_threshold;
      } else {
        eff_shift[last_idx] = dim_range[last_eff_dim] - threshold[last_eff_dim];
      }

      for (int i = num_eff_dims-2; i >= 0; --i) {
        size_t eff_dim = eff_dims[i];
        eff_range[i] = dim_range[eff_dim];
        eff_size[i] = eff_range[i] / eff_range[i+1];

        const int64 eff_threshold = threshold[eff_dim] * dim_range[eff_dim+1];
        eff_shift[i] = dim_range[eff_dim] - eff_threshold;
      }

      functor::RollFunctor<Device, T> func;
      func(context->eigen_device<Device>(), num_elements, num_eff_dims,
           input_flat, output_flat, eff_shift, eff_range, eff_size);
    } else {
      // reduce the tensor dimensions to the dimensions that will get shifted.
      // unshifted dimensions will get flattened to an outer shifted dimension
      // or the first dimension (outer most dimension) if there are no
      // outer shifted dimensions.
      // num_eff_dims = number of efective dimensions
      // eff_dims = array of original indicies of effective dimensions
      // eff_shift = effective shift - the amount elements will shift in the
      // flattend tensor for a given effective dimension.
      gtl::InlinedVector<int64, 4> eff_range(num_eff_dims);
      gtl::InlinedVector<int64, 4> eff_size(num_eff_dims);
      gtl::InlinedVector<int64, 4> eff_threshold(num_eff_dims);
      for (int i = 0; i < num_eff_dims; i++) {
        eff_size[i] = 1;
        eff_range[i] = 1;
        eff_threshold[i] = 1;
      }

      int64 stride = 1;
      size_t j = num_eff_dims - 1;
      for (int i = num_dims - 1; i >= 0; i--) {
        const int size = std::max<int>(input.dim_size(i), 1);
        eff_size[j] *= size;
        if (shift_mod_sum[i] != 0 || i == 0) {
          const int threshold = ((size - shift_mod_sum[i]) % size);
          eff_threshold[j] = stride * threshold;
          eff_range[j] = size * stride;
          j--;
        }
        stride *= static_cast<int64>(input.dim_size(i));
      }

      // memcpy is faster when the contiguous block is size > 32
      // when num_elements < 64 time doesn't matter so much
      // and our test cases can test this algorithm
      bool preferMemcpy = eff_range[num_eff_dims-1] > 32 || num_elements < 64;
      if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v()) && preferMemcpy) {
        // copies memory in groups instead of element by element
        DoRollWithMemcpy<T>(context, num_elements, num_eff_dims,
                  input_flat, output_flat, eff_threshold, eff_range);
      } else {
        // in-case memcpy does not work for current data type or is slower
        DoRoll<T>(context, num_elements, num_eff_dims,
                  input_flat, output_flat, eff_threshold, eff_range);
      }
    }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<CPUDevice, type, int32, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<CPUDevice, type, int64, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<CPUDevice, type, int32, int64>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<CPUDevice, type, int64, int64>)
TF_CALL_ALL_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<GPUDevice, type, int32, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<GPUDevice, type, int64, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<GPUDevice, type, int32, int64>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<GPUDevice, type, int64, int64>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU)
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
