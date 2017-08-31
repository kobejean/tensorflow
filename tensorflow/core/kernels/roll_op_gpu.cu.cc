/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "roll_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {
// CUDA kernel.
template <typename T>
__global__ void RollCudaKernel(const int N, const int D, const int* dim_size,
                               const T* input, T* output, const int* threshold,
                               const int* dim_range) {
  const int64 start = blockIdx.x * blockDim.x + threadIdx.x;
  const int64 end = N;

  int indices[2];  // array of indices for each dimension
  int offset = 0;  // the shift along the flat tensor for current element
  // initialize indices and offset
  for (int d = 0; d < D; d++) {
    // stride is the number of indices over in the flattened tensor
    // you need to skip in order to make it over to an adjacent element
    // along a dimension.
    const int64 stride = dim_range[d] / dim_size[d];
    const int shift = dim_size[d] - threshold[d];
    const int indx = (start / stride) % dim_size[d];
    indices[d] = indx;
    // calculate dimension index after the shift
    const int shifted_indx = (indx + shift) % dim_size[d];
    offset += (shifted_indx - indx) * stride;
  }

  for (int i = start; i < end; i += blockDim.x * gridDim.x) {
    output[i + offset] = input[i];
    // create next combination of indices
    // while at it adjust offset if needed
    for (int d = D - 1; d >= 0; d--) {
      const int indx = (indices[d] + 1) % dim_size[d];
      indices[d] = indx;
      if (indx != 0) {
        if (indx == threshold[d]) {  // we've reached the threshold
          // dim_range[d] = threshold[d] + shift[d]
          // offset = shift[d] + ... other offsets
          // offset - dim_range[d] = -threshold[d] + ... other offsets
          // thus we undo our previous offset as well as add a new offset of
          // -threshold[d] in one opperation
          offset -= dim_range[d];  // now wraps around
        }
        break;                         // indx != 0 don't need to carry
      } else if (threshold[d] != 0) {  // if threshold is 0 shift is 0
        offset += dim_range[d];        // indx became 0 so reverse wrap around
      }
    }
  }
}

}  // namespace

namespace functor {
// GPU implementation that launches the CUDA kernel.
template <typename T, int Dims>
struct RollFunctor<GPUDevice, T, Dims> {
  void operator()(const GPUDevice& d, const tensorflow::Tensor input,
                  const tensorflow::Tensor shift,
                  const tensorflow::Tensor axis,
                  tensorflow::Tensor* output) {
    auto shift_flat = shift.flat<Tshift>();
    auto axis_flat = axis.flat<Taxis>();
    const int64 N = input.NumElements();
    const int M = static_cast<int>(shift_flat.size());
    const int D = static_cast<int>(input.dims());

    int shift_mod_sum[D];  // if any duplicate axes, will sum corresponding
                           // shifts
    for (int d = 0; d < D; d++) shift_mod_sum[d] = 0;  // default is 0
    for (int m = 0; m < M; m++) {
      const int a = axis_flat(m);
      OP_REQUIRES(context, a < D,
                  errors::InvalidArgument("axis ", a, " is out of range"));
      const int ds = fmax(static_cast<int>(input.dim_size(a)), 1);
      const int sum = shift_mod_sum[a] + static_cast<int>(shift_flat(m));
      // modulo that works with negatives: ((x % y) + y) % y
      shift_mod_sum[a] = (sum % ds + ds) % ds;
    }
    // the size of each dimension
    int dim_size[D];
    // threshold[d] is the index that the roll starts to wrap back to the front
    int threshold[D];
    // dim_range is the number of indices over in the flattened tensor
    // you need to skip in order to make it over from one side of a dimension
    // to the other. Used to make the shifts wrap around after a threshold.
    int64 dim_range[D];
    int64 dim_size_prod = 1;
    // inner shift dimension (inner most shifted dimension)
    int64 isd = 0;
    for (int d = D - 1; d >= 0; d--) {
      if (!isd && shift_mod_sum[d]) isd = d;
      const int ds = fmax(static_cast<int>(input.dim_size(d)), 1);
      dim_size[d] = ds;
      threshold[d] = (ds - shift_mod_sum[d]) % ds;
      dim_size_prod *= static_cast<int64>(input.dim_size(d));
      dim_range[d] = dim_size_prod;
    }
    auto input_flat = input.flat<T>().data();
    auto output_flat = output->flat<T>().data();
    CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
    RollCudaKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            N, D, dim_size, input_flat, output_flat, threshold, dim_range);
  }
};

// Definition of the GPU implementations declared in roll_op.h.
#define DEFINE_GPU_SPEC_TYPE_DIMS(T, Dims)                                 \
  template struct RollFunctor<GPUDevice, T, Dims>; \
  template struct RollFunctor<GPUDevice, T, Dims>; \
  template struct RollFunctor<GPUDevice, T, Dims>; \
  template struct RollFunctor<GPUDevice, T, Dims>;

#define DEFINE_GPU_SPECS(T)            \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 0);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 1);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 2);     \

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPEC



}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
