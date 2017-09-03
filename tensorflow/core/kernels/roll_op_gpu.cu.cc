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
template <typename T, int Dims>
__global__ void RollCudaKernel(const tensorflow::int64 N, const int D,
                               const Eigen::DSizes<Eigen::DenseIndex, Dims>& dim_size,
                               const T* input, T* output,
                               const Eigen::DSizes<Eigen::DenseIndex, Dims>& threshold,
                               const Eigen::DSizes<Eigen::DenseIndex, Dims>& dim_range) {
  const int64 start = blockIdx.x * blockDim.x + threadIdx.x;
  const int64 end = N;

  int indices[Dims];  // array of indices for each dimension
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
  void operator()(const GPUDevice& d, const tensorflow::int64 N, const int D,
                  const Eigen::DSizes<Eigen::DenseIndex, Dims>& dim_size,
                  typename TTypes<T, Dims>::ConstTensor input,
                  typename TTypes<T, Dims>::Tensor output,
                  const Eigen::DSizes<Eigen::DenseIndex, Dims>& threshold,
                  const Eigen::DSizes<Eigen::DenseIndex, Dims>& dim_range) {

    CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
    RollCudaKernel<T, Dims>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            N, D, dim_size, input.data(), output.data(), threshold,
            dim_range);
  }
};

// Definition of the GPU implementations declared in roll_op.h.
#define DEFINE_GPU_SPEC_TYPE_DIMS(T, Dims)                                 \
  template struct RollFunctor<GPUDevice, T, Dims>;

#define DEFINE_GPU_SPECS(T)            \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 0);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 1);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 2);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 3);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 4);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 5);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 6);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 7);     \
  DEFINE_GPU_SPEC_TYPE_DIMS(T, 8);     \

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPEC



}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
