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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class RollOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "Roll")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RollOpTest, ScalarIndices) {
  MakeOp(DT_FLOAT, DT_INT32);

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

TEST_F(RollOpTest, ScalarIndices_Complex) {
  MakeOp(DT_COMPLEX64, DT_INT32);

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
  MakeOp(DT_FLOAT, DT_INT32);

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

TEST_F(RollOpTest, Simple_ThreeD32) {
  MakeOp(DT_FLOAT, DT_INT32);

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

TEST_F(RollOpTest, Simple_TwoD64) {
  MakeOp(DT_FLOAT, DT_INT64);

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

TEST_F(RollOpTest, Simple_ThreeD64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int64>(TensorShape({3}), {-1, 1, 2});
  AddInputFromArray<int64>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 3}));
  test::FillValues<float>(&expected, {10, 11, 9, 7, 8, 6, 4, 5, 3, 1, 2, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, ZeroShift_ThreeD32) {
  MakeOp(DT_FLOAT, DT_INT32);

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

TEST_F(RollOpTest, ZeroSize_ThreeD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({0, 5, 0}), {});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({0, 5, 0}));
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RollOpTest, DuplicateShifts_TwoD32) {
  MakeOp(DT_FLOAT, DT_INT32);

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

TEST_F(RollOpTest, Error_InputMustBeVectorOrHigher) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {7});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString()).contains("input must be 1-D or higher"))
      << s;
}

TEST_F(RollOpTest, Error_AxisMustBeScalarOrVector) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({1, 2}), {0, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("axis must be a scalar or a 1-D vector"))
      << s;
}

TEST_F(RollOpTest, Error_ShiftMustBeScalarOrVector) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({1, 2}), {0, 1});
  AddInputFromArray<int32>(TensorShape({}), {1});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("shift must be a scalar or a 1-D vector"))
      << s;
}

TEST_F(RollOpTest, Error_ShiftAndAxisMustBeSameSize) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 2}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({1}), {1});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("shift and axis must be the same size"))
      << s;
}

TEST_F(RollOpTest, Error_AxisOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {1});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString()).contains("is out of range")) << s;
}

static Graph* RollGraph(const TensorShape& shape, int isd) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, shape);
  input.flat<float>().setRandom();
  const int D = input.dims();
  Tensor shift(DT_INT32, TensorShape({D}));
  for (int i = 0; i <= D; i++) {
    shift.flat<int32>()(i) = (i > isd) ? 0 : 2;
  }
  Tensor axis(DT_INT32, TensorShape({D}));
  for (int i = 0; i < D; i++) {
    axis.flat<int32>()(i) = i;
  }
  test::graph::Roll(g, test::graph::Constant(g, input),
                    test::graph::Constant(g, shift),
                    test::graph::Constant(g, axis));
  return g;
}

#define BM_ROLL_OUTER(DEVICE)                                                 \
  static void BM_##DEVICE##_roll_outer(int iters, int rows, int columns) {    \
    TensorShape shape{rows, columns};                                         \
    const int64 num_items = static_cast<int64>(iters) * shape.num_elements(); \
    testing::ItemsProcessed(num_items);                                       \
    testing::BytesProcessed(num_items * sizeof(float));                       \
    testing::UseRealTime();                                                   \
    test::Benchmark(#DEVICE, RollGraph(shape, 0)).Run(iters);                 \
  }                                                                           \
  BENCHMARK(BM_##DEVICE##_roll_outer)                                         \
      ->ArgPair(256, 256)                                                     \
      ->ArgPair(512, 512)                                                     \
      ->ArgPair(1024, 1024)                                                   \
      ->ArgPair(2048, 2048)

#define BM_ROLL_ALL(DEVICE)                                                   \
  static void BM_##DEVICE##_roll_all(int iters, int rows, int columns) {      \
    TensorShape shape{rows, columns};                                         \
    const int64 num_items = static_cast<int64>(iters) * shape.num_elements(); \
    testing::ItemsProcessed(num_items);                                       \
    testing::BytesProcessed(num_items * sizeof(float));                       \
    testing::UseRealTime();                                                   \
    test::Benchmark(#DEVICE, RollGraph(shape, 0)).Run(iters);                 \
  }                                                                           \
  BENCHMARK(BM_##DEVICE##_roll_all)                                           \
      ->ArgPair(256, 256)                                                     \
      ->ArgPair(512, 512)                                                     \
      ->ArgPair(1024, 1024)                                                   \
      ->ArgPair(2048, 2048)

BM_ROLL_OUTER(cpu);
BM_ROLL_ALL(cpu);
BM_ROLL_OUTER(gpu);
BM_ROLL_ALL(gpu);

}  // namespace
}  // namespace tensorflow
