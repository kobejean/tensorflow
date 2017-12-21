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

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

// CUDA kernel.
template <typename T>
__global__ void RollCudaKernel(const int64 N, const int32 D,
                               const T* input, T* output,
                               const int32* buf, const int32 tpb) {
  const int64 start = blockIdx.x * blockDim.x + threadIdx.x;
  const int64 end = N;

  const int32* dim_size = buf;
  const int32* threshold = buf + D;
  const int32* dim_range = buf + D * 2;

  // array of indices for each dimension for each thread using shared memory
  extern __shared__ int indices[];
  int offset = 0;  // the shift along the flat tensor for current element
  // initialize indices and offset
  for (int d = 0; d < D; d++) {
    // stride is the number of indices over in the flattened tensor
    // you need to skip in order to make it over to an adjacent element
    // along a dimension. dim_size[d] != 0 because we set it to max(dim, 1)
    const int64 stride = dim_range[d] / dim_size[d];
    const int shift = dim_size[d] - threshold[d];
    const int indx = (start / stride) % dim_size[d];
    indices[d * tpb + threadIdx.x] = indx;
    // calculate dimension index after the shift
    const int shifted_indx = (indx + shift) % dim_size[d];
    offset += (shifted_indx - indx) * stride;
  }

  for (int i = start; i < end; i += blockDim.x * gridDim.x) {
    output[i + offset] = input[i];
    // create next combination of indices
    // while at it adjust offset if needed
    for (int d = D - 1; d >= 0; d--) {
      const int indx = (indices[d * tpb + threadIdx.x] + 1) % dim_size[d];
      indices[d * tpb + threadIdx.x] = indx;
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

namespace functor {
// GPU implementation that launches the CUDA kernel.
template <typename T>
struct RollFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, const int64 N, const int D,
              const gtl::ArraySlice<int>& dim_size, const T* input, T* output,
              const gtl::ArraySlice<int>& threshold,
              const gtl::ArraySlice<int64>& dim_range) {
    gtl::InlinedVector<int32, 12> host_buf(D * 3);
    for (int i = 0; i < D; i++) {
      host_buf[i] = dim_size[i];
      host_buf[D + i] = threshold[i];
      host_buf[D * 2 + i] = static_cast<int32>(dim_range[i]);
    }

    CudaLaunchConfig config = GetCudaLaunchConfig(static_cast<int>(N), d);

    int32 buf_bytes = sizeof(int32) * host_buf.size();
    int32 shared_mem_bytes = sizeof(int32) * D * config.thread_per_block;
    auto dev_buf = d.allocate(buf_bytes);
    // NOTE: host_buf is not allocated by CudaHostAllocator, and
    // therefore we are doing a sync copy effectively.
    d.memcpyHostToDevice(dev_buf, host_buf.data(), buf_bytes);

    RollCudaKernel<T>
        <<<config.block_count, config.thread_per_block, shared_mem_bytes,
           d.stream()>>>(
            N, D, input, output, reinterpret_cast<const int32*>(dev_buf),
            config.thread_per_block);
    // Safe to deallocate immediately after the kernel launch.
    d.deallocate(dev_buf);
  }
};

// Definition of the GPU implementations declared in roll_op.h.
#define DEFINE_GPU_SPEC(T)                                                    \
  template struct RollFunctor<GPUDevice, T>;                                  \

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);

#undef DEFINE_GPU_SPEC

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
