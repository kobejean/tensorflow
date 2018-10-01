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
    switch (num_eff_dims) {
      case 1: {
        auto config = GetCudaLaunchConfig(num_elements, d);//, RollKernel1D<T>, 0, 0);
        RollKernel1D<T>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                config, input, output, eff_shift, eff_range);
      } break;
      case 2: {
        auto config = GetCuda2DLaunchConfig(eff_size[0], eff_size[1],
                                            d);//, RollKernel2D<T>, 0, 0);
        RollKernel2D<T>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                config, input, output, eff_shift, eff_range);
      } break;
      case 3: {
        auto config = GetCuda3DLaunchConfig(eff_size[0], eff_size[1],
                                            eff_size[2], d, RollKernel3D<T>, 0, 0);
        RollKernel3D<T>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                config, input, output, eff_shift, eff_range);
      } break;
      default: {
        auto config = GetCudaLaunchConfig(num_elements, d);
        RollKernel<T>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                config, num_eff_dims, input, output, eff_shift, eff_range);
      } break;
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
