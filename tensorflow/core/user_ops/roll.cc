#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "roll.h"

using namespace tensorflow;

REGISTER_OP("Roll")
        .Input("input: T")
        .Input("shift: Tshift")
        .Input("axis: Taxis")
        .Output("output: T")
        .Attr("T: type")
        .Attr("Tshift: {int32,int64}")
        .Attr("Taxis: {int32,int64}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

#define EIGEN_USE_THREADS
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct RollFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int N, int D, int* dim_size, const T* input, T* output, \
                  int* shifts, int* strides) {

      for (int in_i = 0; in_i < N; in_i++) {
          int out_i = in_i;
          // loop through dimensions
          for (int d = 0; d < D; d++) {
              // find indices input/output for current dimension
              const int in_dim_i = (in_i / strides[d]) % dim_size[d];
              const int out_dim_i = (in_dim_i + shifts[d]) % dim_size[d];
              // convert back to flat index
              out_i += (out_dim_i - in_dim_i) * strides[d];
          }

          output[out_i] = input[in_i];
      }
  }
};

template <typename Device, typename T, typename Tshift, typename Taxis>
class RollOp : public OpKernel {
 public:
  explicit RollOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input = context->input(0);
    const Tensor& shift = context->input(1);
    const Tensor& axis = context->input(2);

    // auto input_flat = input.flat<T>();
    auto shift_flat = shift.flat<Tshift>();
    auto axis_flat = axis.flat<Taxis>();

    OP_REQUIRES(context, shift.shape().dims() <= 1,
                errors::InvalidArgument("Roll expects a scalar or a 1-D vector for shift."));
    OP_REQUIRES(context, axis.shape().dims() <= 1,
                errors::InvalidArgument("Roll expects a scalar or a 1-D vector for axis."));
    OP_REQUIRES(context, shift.shape() == axis.shape(),
                errors::InvalidArgument("Roll expects shift and axis to be the same size."));

    const int D = static_cast<int>(input.dims());
    const int M = static_cast<int>(shift_flat.size());
    // const int N = input_flat.size();

    int shifts[D];
    for (int i = 0; i < D; i++) {
        shifts[i] = 0;
    }
    for (int i = 0; i < M; i++) {
        const int j = axis_flat(i);
        OP_REQUIRES(context, j < D,
                    errors::InvalidArgument("Roll expects axis to be in range."));
        shifts[j] = shift_flat(i);
    }

    int strides[D];
    int last_stride = 1;
    for (int i = D-1; i >= 0; i--) {
        strides[i] = last_stride;
        last_stride *= input.dim_size(i);
    }

    int dim_size[D];
    for (int i = 0; i < D; i++) {
        dim_size[i] = static_cast<int>(input.dim_size(i));
    }

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(),
                                                     &output));

    OP_REQUIRES(context, input.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    RollFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input.NumElements()),
        static_cast<int>(input.dims()),
        dim_size,
        input.flat<T>().data(),
        output->flat<T>().data(),
        shifts,
        strides
    );

  }
};


// Register the CPU kernels.
#define REGISTER_CPU(type)                                        \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int32>("Tshift")        \
                          .TypeConstraint<int32>("Taxis"),        \
                        RollOp<CPUDevice, type, int32, int32>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int64>("Tshift")        \
                          .TypeConstraint<int32>("Taxis"),        \
                        RollOp<CPUDevice, type, int64, int32>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int32>("Tshift")        \
                          .TypeConstraint<int64>("Taxis"),        \
                        RollOp<CPUDevice, type, int32, int64>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int64>("Tshift")        \
                          .TypeConstraint<int64>("Taxis"),        \
                        RollOp<CPUDevice, type, int64, int64>)

TF_CALL_ALL_TYPES(REGISTER_CPU);
REGISTER_CPU(bfloat16);
#undef REGISTER_CPU


// // Register the GPU kernels.
// #ifdef GOOGLE_CUDA
// #define REGISTER_GPU(T)                                          \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
//       ExampleOp<GPUDevice, T>);
// REGISTER_GPU(float);
// REGISTER_GPU(int32);
// #endif  // GOOGLE_CUDA
