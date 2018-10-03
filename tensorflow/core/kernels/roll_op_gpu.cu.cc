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
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {
// CUDA kernel.

template <typename T>
__global__ void RollKernelV2(CudaLaunchConfig config,
                             const int32 num_eff_dims,
                             const int work_per_thread,
                             const int num_elements,
                             const T* input, T* output,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_shift,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_range,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_size) {
  CUDA_1D_KERNEL_LOOP(virtual_thread, config.virtual_thread_count) {
    const int64 start = virtual_thread * work_per_thread;
    const int64 end = tf_min<int64>(start+work_per_thread, num_elements);
    // array of indices for each dimension
    Eigen::array<int64, MAX_DIM_GPU> indices;
    int64 offset = 0; // the shift along the flattened tensor for current element
    // initialize indices and offset
    for (int i = 0; i < num_eff_dims; i++) {
      const int64 stride = (i+1 < num_eff_dims) ? eff_range[i+1] : 1;
      const int indx = (start / stride) % eff_size[i];
      const int shift = (start + eff_shift[i]) % eff_range[i];
      indices[i] = indx;
      offset += shift - (start % eff_range[i]);
    }


    for (int64 i = start; i < end; i++) {
      output[i + offset] = input[i];
      // create next combination of indices
      // while at it adjust offset if needed
      for (int j = num_eff_dims - 1; j >= 0; j--) {
        const int indx = (indices[j] + 1) % eff_size[j];
        indices[j] = indx;
        if (indx != 0) {
          if (i % eff_range[j] == eff_range[j] - eff_shift[j]) {  // we've reached the threshold
            // dim_range[j] = threshold[j] + shift[j]
            // offset = shift[j] + ... other offsets
            // offset - dim_range[j] = -threshold[j] + ... other offsets
            // thus we undo our previous offset as well as add a new offset of
            // -threshold[j] in one operation
            offset -= eff_range[j];  // now wraps around
          }
          break;                         // indx != 0 don't need to carry
        } else if (eff_shift[j] != 0) {  // if threshold is 0 shift is 0
          offset += eff_range[j];        // indx became 0 so reverse wrap around
        }
      }
    }
  }
}

template <typename T>
__global__ void RollKernel(CudaLaunchConfig config, const int32 num_eff_dims,
                           const T* input, T* output,
                           const Eigen::array<int64, MAX_DIM_GPU> eff_shift,
                           const Eigen::array<int64, MAX_DIM_GPU> eff_range) {
  CUDA_1D_KERNEL_LOOP(i, config.virtual_thread_count) {
    int64 offset = 0; // the shift along the flattened tensor for current element
    // calculate offset
    for (int j = 0; j < num_eff_dims; j++) {
      const int64 idx = i % eff_range[j];
      const int64 shifted_idx = (idx + eff_shift[j]) % eff_range[j];
      offset += shifted_idx - idx;
    }
    output[i + offset] = input[i];
  }
}


template <typename T>
__global__ void RollKernel1D(CudaLaunchConfig config, const T* input, T* output,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_shift,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_range) {
  const int eff_threshold_x = eff_range[0] - eff_shift[0];
  CUDA_1D_KERNEL_LOOP(i, config.virtual_thread_count) {
    if (i < eff_threshold_x) {
      output[i + eff_shift[0]] = input[i];
    } else {
      output[i - eff_threshold_x] = input[i];
    }
  }
}

template <typename T>
__global__ void RollKernel2D(Cuda2DLaunchConfig config,
                             const T* input, T* output,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_shift,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_range) {
  const int eff_threshold_x = eff_range[0] - eff_shift[0];
  const int eff_threshold_y = eff_range[1] - eff_shift[1];
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    int offset = (x < eff_threshold_x) ? eff_shift[0] : -eff_threshold_x;
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      offset += (y < eff_threshold_y) ? eff_shift[1] : -eff_threshold_y;
      int i = x * eff_range[1] + y;
      output[i + offset] = input[i];
    }
  }
}

template <typename T>
__global__ void RollKernel3D(Cuda3DLaunchConfig config,
                             const T* input, T* output,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_shift,
                             const Eigen::array<int64, MAX_DIM_GPU> eff_range) {
  const int eff_threshold_x = eff_range[0] - eff_shift[0];
  const int eff_threshold_y = eff_range[1] - eff_shift[1];
  const int eff_threshold_z = eff_range[2] - eff_shift[2];
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count.x, X) {
    int offset = (x < eff_threshold_x) ? eff_shift[0] : -eff_threshold_x;
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count.y, Y) {
      offset += (y < eff_threshold_y) ? eff_shift[1] : -eff_threshold_y;
      CUDA_AXIS_KERNEL_LOOP(z, config.virtual_thread_count.z, Z) {
        offset += (z < eff_threshold_z) ? eff_shift[2] : -eff_threshold_z;
        int i = x * eff_range[1] + y * eff_range[2] + z;
        output[i + offset] = input[i];
      }
    }
  }
}

}  // namespace

namespace functor {
// GPU implementation that launches the CUDA kernel.
template <typename T>
struct RollFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, const int64 num_elements,
                  const int num_eff_dims, const T* input, T* output,
                  const Eigen::array<int64, MAX_DIM_GPU> eff_shift,
                  const Eigen::array<int64, MAX_DIM_GPU> eff_range,
                  const Eigen::array<int64, MAX_DIM_GPU> eff_size) {
    const int thread_count =
          d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor();
    // 266241260084039
    //
    const int work_per_thread = DivUp(num_elements, thread_count);
    if (false/*work_per_thread > 8*/) {
      CudaLaunchConfig config;
      const int thread_per_block = std::min(1024, d.maxCudaThreadsPerBlock());
      const int virtual_block_count = DivUp(thread_count, thread_per_block);
      const int block_count =
          std::min(virtual_block_count, d.getNumCudaMultiProcessors());
      // thread_count > num_elements because work_per_thread > 8
      config.virtual_thread_count = thread_count;
      config.thread_per_block = thread_per_block;
      config.block_count = block_count;
      RollKernelV2<T>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              config, num_eff_dims, work_per_thread, num_elements, input,
              output, eff_shift, eff_range, eff_size);

    } else {
      switch (num_eff_dims) {
        // case 1: {
        //   auto config = GetCudaLaunchConfig(num_elements, d);
        //   RollKernel1D<T>
        //       <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        //           config, input, output, eff_shift, eff_range);
        // } break;
        // case 2: {
        //   auto config = GetCuda2DLaunchConfig(eff_size[0], eff_size[1], d);
        //   RollKernel2D<T>
        //       <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        //           config, input, output, eff_shift, eff_range);
        // } break;
        // case 3: {
        //   auto config = GetCuda3DLaunchConfig(eff_size[0], eff_size[1],
        //                                       eff_size[2], d, RollKernel3D<T>, 0, 0);
        //   RollKernel3D<T>
        //       <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        //           config, input, output, eff_shift, eff_range);
        // } break;
        default: {
          auto config = GetCudaLaunchConfig(num_elements, d);
          RollKernel<T>
              <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                  config, num_eff_dims, input, output, eff_shift, eff_range);
        } break;
      }
    }
  }
};

// Definition of the GPU implementations declared in roll_op.h.
#define DEFINE_GPU_SPECS(T)                            \
  template struct RollFunctor<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPEC
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
