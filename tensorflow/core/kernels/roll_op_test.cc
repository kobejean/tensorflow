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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace test {
namespace graph {

class Node* Roll(Graph* g, class Node* input, class Node* shift, class Node* axis) {
  class Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Roll")
                  .Input(input)
                  .Input(shift)
                  .Input(axis)
                  .Finalize(g, &ret));
  return ret;
}

}  // namespace graph
}  // namespace test

namespace {

class RollOpTest : public OpsTestBase {
 protected:
  enum class Device { CPU, GPU };

  void MakeOp(Device device, DataType data_type, DataType index_type) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }
    TF_ASSERT_OK(NodeDefBuilder("myop", "Roll")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RollOpTest, ScalarIndices) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA
  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {2, 3, 4, 0, 1});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ScalarIndices_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(TensorShape({5}), {"a", "b", "c", "d", "e"});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({5}));
  test::FillValues<string>(&expected, {"c", "d", "e", "a", "b"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ScalarIndices_Complex) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_COMPLEX64, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_COMPLEX64, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<std::complex<float>>(
      TensorShape({5}), {std::complex<float>(0, 10), std::complex<float>(1, 11),
                         std::complex<float>(2, 12), std::complex<float>(3, 13),
                         std::complex<float>(4, 14)});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_COMPLEX64, TensorShape({5}));
  test::FillValues<std::complex<float>>(
      &expected, {std::complex<float>(2, 12), std::complex<float>(3, 13),
                  std::complex<float>(4, 14), std::complex<float>(0, 10),
                  std::complex<float>(1, 11)});
  test::ExpectTensorEqual<std::complex<float>>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD32) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({3, 5}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({2}), {2, -1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3, 5}));
  test::FillValues<float>(&expected,
                          {6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 1, 2, 3, 4, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD32_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(TensorShape({3, 5}),
                            {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                             "k", "l", "m", "n", "o"});
  AddInputFromArray<int32>(TensorShape({2}), {2, -1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({3, 5}));
  test::FillValues<string>(&expected, {"g", "h", "i", "j", "f", "l", "m", "n",
                                       "o", "k", "b", "c", "d", "e", "a"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD32) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int32>(TensorShape({3}), {1, -1, -1});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 3}));
  test::FillValues<float>(&expected, {10, 11, 9, 7, 8, 6, 4, 5, 3, 1, 2, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD32_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(
      TensorShape({2, 2, 3}),
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  AddInputFromArray<int32>(TensorShape({3}), {1, -1, -1});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({2, 2, 3}));
  test::FillValues<string>(
      &expected, {"k", "l", "j", "h", "i", "g", "e", "f", "d", "b", "c", "a"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD64) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT64);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT64);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int64>(TensorShape({2}), {-1, 4});
  AddInputFromArray<int64>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected,
                          {5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 2, 0, 1});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_TwoD64_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT64);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT64);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(TensorShape({5, 3}),
                            {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                             "k", "l", "m", "n", "o"});
  AddInputFromArray<int64>(TensorShape({2}), {-1, 4});
  AddInputFromArray<int64>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({5, 3}));
  test::FillValues<string>(&expected, {"f", "d", "e", "i", "g", "h", "l", "j",
                                       "k", "o", "m", "n", "c", "a", "b"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD64) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT64);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT64);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({4, 1, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int64>(TensorShape({3}), {4, 3, 2});
  AddInputFromArray<int64>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 1, 3}));
  test::FillValues<float>(&expected, {1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Simple_ThreeD64_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT64);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT64);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(
      TensorShape({4, 1, 3}),
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  AddInputFromArray<int64>(TensorShape({3}), {4, 3, 2});
  AddInputFromArray<int64>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({4, 1, 3}));
  test::FillValues<string>(
      &expected, {"b", "c", "a", "e", "f", "d", "h", "i", "g", "k", "l", "j"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroShift_ThreeD32) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int32>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroShift_ThreeD32_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(
      TensorShape({2, 2, 3}),
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  AddInputFromArray<int32>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({2, 2, 3}));
  test::FillValues<string>(
      &expected, {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroSize_ThreeD32) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 0, 0}), {});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 0, 0}));
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroSize_ThreeD32_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(TensorShape({5, 0, 0}), {});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({5, 0, 0}));
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, OneSize_ThreeD32) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({1, 1, 1}), {5});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1}));
  test::FillValues<float>(&expected, {5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, OneSize_ThreeD32_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(TensorShape({1, 1, 1}), {"a"});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({1, 1, 1}));
  test::FillValues<string>(&expected, {"a"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, MultiShifts_TwoD32) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({3, 5}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {-2, 2, -1, 1});
  AddInputFromArray<int32>(TensorShape({4}), {1, 0, 0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3, 5}));
  test::FillValues<float>(&expected,
                          {11, 12, 13, 14, 10, 1, 2, 3, 4, 0, 6, 7, 8, 9, 5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, MultiShifts_TwoD32_NoMemcpy) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_STRING, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_STRING, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<string>(TensorShape({3, 5}),
                            {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                             "k", "l", "m", "n", "o"});
  AddInputFromArray<int32>(TensorShape({4}), {-2, 2, -1, 1});
  AddInputFromArray<int32>(TensorShape({4}), {1, 0, 0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_STRING, TensorShape({3, 5}));
  test::FillValues<string>(&expected, {"l", "m", "n", "o", "k", "b", "c", "d",
                                       "e", "a", "g", "h", "i", "j", "f"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, Error_InputMustBeVectorOrHigher) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {7});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "input must be 1-D or higher"))
      << s;
}

TEST_F(RollOpTest, Error_AxisMustBeScalarOrVector) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({1, 2}), {0, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(str_util::StrContains(s.ToString(),
                                    "axis must be a scalar or a 1-D vector"))
      << s;
}

TEST_F(RollOpTest, Error_ShiftMustBeScalarOrVector) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({1, 2}), {0, 1});
  AddInputFromArray<int32>(TensorShape({}), {1});
  Status s = RunOpKernel();
  EXPECT_TRUE(str_util::StrContains(s.ToString(),
                                    "shift must be a scalar or a 1-D vector"))
      << s;
}

TEST_F(RollOpTest, Error_ShiftAndAxisMustBeSameSize) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({1}), {1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(str_util::StrContains(s.ToString(),
                                    "shift and axis must have the same size"))
      << s;
}

TEST_F(RollOpTest, Error_AxisOutOfRange) {
  #ifdef GOOGLE_CUDA
  MakeOp(Device::GPU, DT_FLOAT, DT_INT32);
  #else
  MakeOp(Device::CPU, DT_FLOAT, DT_INT32);
  #endif  // GOOGLE_CUDA

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {1});
  Status s = RunOpKernel();
  EXPECT_TRUE(str_util::StrContains(s.ToString(), "is out of range")) << s;
}

// isd - (inner shift dimension) The inner most dimension to be shifted.
//    All outer dimensions will also be shifted for testing.
static Graph* RollGraph(TensorShape& shape, int isd) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, shape);
  input.flat<float>().setRandom();
  int dims = input.dims();
  Tensor shift(DT_INT32, TensorShape({dims}));
  for (int i = 0; i < dims; i++) {
    // shift the inner shift dimension and all outer dimensions
    shift.flat<int32>()(i) = (i <= isd) ? input.dim_size(i) / 2 : 0;
  }
  Tensor axis(DT_INT32, TensorShape({dims}));
  for (int i = 0; i < dims; i++) {
    axis.flat<int32>()(i) = i;
  }
  test::graph::Roll(g, test::graph::Constant(g, input),
                    test::graph::Constant(g, shift),
                    test::graph::Constant(g, axis));
  return g;
}

#define BM_ROLL_VIDEO_CHANNELS(DEVICE)                                        \
  static void BM_##DEVICE##_roll_vid_channels(int iters, int frames, int n) { \
    TensorShape shape{ frames, n, n, 3 };                                     \
    const int64 num_items = static_cast<int64>(iters) * shape.num_elements(); \
    testing::ItemsProcessed(num_items);                                       \
    testing::BytesProcessed(num_items * sizeof(float));                       \
    testing::UseRealTime();                                                   \
    test::Benchmark(#DEVICE, RollGraph(shape, 3)).Run(iters);                 \
  }                                                                           \
  BENCHMARK(BM_##DEVICE##_roll_vid_channels)                                  \
      ->ArgPair(30, 64)                                                       \
      ->ArgPair(30, 128)                                                      \
      ->ArgPair(30, 256)

#define BM_ROLL_VIDEO_TIME(DEVICE)                                            \
  static void BM_##DEVICE##_roll_vid_time(int iters, int frames, int n) {     \
    TensorShape shape{ frames, n, n, 3 };                                     \
    const int64 num_items = static_cast<int64>(iters) * shape.num_elements(); \
    testing::ItemsProcessed(num_items);                                       \
    testing::BytesProcessed(num_items * sizeof(float));                       \
    testing::UseRealTime();                                                   \
    test::Benchmark(#DEVICE, RollGraph(shape, 0)).Run(iters);                 \
  }                                                                           \
  BENCHMARK(BM_##DEVICE##_roll_vid_time)                                      \
      ->ArgPair(30, 64)                                                       \
      ->ArgPair(30, 128)                                                      \
      ->ArgPair(30, 256)


#define BM_ROLL_IMAGE_CHANNELS(DEVICE)                                        \
  static void BM_##DEVICE##_roll_img_channels(int iters, int n) {             \
    TensorShape shape{ n, n, 3 };                                             \
    const int64 num_items = static_cast<int64>(iters) * shape.num_elements(); \
    testing::ItemsProcessed(num_items);                                       \
    testing::BytesProcessed(num_items * sizeof(float));                       \
    testing::UseRealTime();                                                   \
    test::Benchmark(#DEVICE, RollGraph(shape, 2)).Run(iters);                 \
  }                                                                           \
  BENCHMARK(BM_##DEVICE##_roll_img_channels)                                  \
      ->Arg(256)                                                              \
      ->Arg(512)                                                              \
      ->Arg(1024)

#define BM_ROLL_IMAGE(DEVICE)                                                 \
  static void BM_##DEVICE##_roll_img(int iters, int n) {                      \
    TensorShape shape{ n, n, 3 };                                             \
    const int64 num_items = static_cast<int64>(iters) * shape.num_elements(); \
    testing::ItemsProcessed(num_items);                                       \
    testing::BytesProcessed(num_items * sizeof(float));                       \
    testing::UseRealTime();                                                   \
    test::Benchmark(#DEVICE, RollGraph(shape, 1)).Run(iters);                 \
  }                                                                           \
  BENCHMARK(BM_##DEVICE##_roll_img)                                           \
      ->Arg(256)                                                              \
      ->Arg(512)                                                              \
      ->Arg(1024)

BM_ROLL_VIDEO_CHANNELS(cpu);
BM_ROLL_VIDEO_TIME(cpu);
BM_ROLL_IMAGE_CHANNELS(cpu);
BM_ROLL_IMAGE(cpu);

#ifdef GOOGLE_CUDA
BM_ROLL_VIDEO_CHANNELS(gpu);
BM_ROLL_VIDEO_TIME(gpu);
BM_ROLL_IMAGE_CHANNELS(gpu);
BM_ROLL_IMAGE(gpu);
#endif  // GOOGLE_CUDA
}  // namespace
}  // namespace tensorflow
