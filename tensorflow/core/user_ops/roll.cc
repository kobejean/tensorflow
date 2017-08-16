#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

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

class RollOp : public OpKernel {
 public:
  explicit RollOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input = context->input(0);
    const Tensor& shift = context->input(1);
    const Tensor& axis = context->input(2);

    auto input_flat = input.flat<int32>();
    auto shift_flat = shift.flat<int32>();
    auto axis_flat = axis.flat<int32>();

    OP_REQUIRES(context, shift.shape().dims() <= 1,
                errors::InvalidArgument("Roll expects a scalar or a 1-D vector for shift."));
    OP_REQUIRES(context, axis.shape().dims() <= 1,
                errors::InvalidArgument("Roll expects a scalar or a 1-D vector for axis."));
    OP_REQUIRES(context, shift.shape() == axis.shape(),
                errors::InvalidArgument("Roll expects shift and axis to be the same size."));

    const int D = input.dims();
    const int M = shift_flat.size();
    const int N = input_flat.size();

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

    // Create an output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(),
                                                     &output));
    auto output_flat = output->flat<int32>();


    // Compute.
    for (int in_i = 0; in_i < N; in_i++) {
        int out_i = in_i;
        // loop through dimensions
        for (int d = 0; d < D; d++) {
            // find indices input/output for current dimension
            const int in_dim_i = (in_i / strides[d]) % input.dim_size(d);
            const int out_dim_i = (in_dim_i + shifts[d]) % input.dim_size(d);
            // convert back to flat index
            out_i += (out_dim_i - in_dim_i) * strides[d];
        }

        output_flat(out_i) = input_flat(in_i);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Roll").Device(DEVICE_CPU), RollOp);
