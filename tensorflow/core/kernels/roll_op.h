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

#ifndef KERNEL_ROLL_H_
#define KERNEL_ROLL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/types.h"

const int MAX_DIM_GPU = 8;

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct RollFunctor {
  void operator()(const Device& d, const int64 num_elements,
                  const int num_eff_dims, const T* input, T* output,
                  const Eigen::array<int64, MAX_DIM_GPU> eff_shift,
                  const Eigen::array<int64, MAX_DIM_GPU> eff_range,
                  const Eigen::array<int64, MAX_DIM_GPU> eff_size);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // KERNEL_ROLL_H_
