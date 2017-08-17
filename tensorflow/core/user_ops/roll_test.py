import tensorflow as tf
import numpy as np
import os

from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradient_checker

# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# g++ -std=c++11 -shared roll.cc -o roll.so -fPIC -I $TF_INC -O2 -undefined dynamic_lookup
# python3 roll_test.py

roll_module = tf.load_op_library("./roll.so")

@ops.RegisterGradient("Roll")
def _RollGrad(op, grad):
    """Gradient for Roll."""
    # The gradient should be just the roll reversed
    shift = op.inputs[1]
    axis = op.inputs[2]
    roll_grad = roll_module.roll(grad, -shift, axis)
    return roll_grad, None, None


class RollTest(tf.test.TestCase):
    def _testRoll(self, np_input, shift, axis, use_gpu=False):
        expected_roll = np.roll(np_input, shift, axis)
        with self.test_session(use_gpu=use_gpu):
            roll = roll_module.roll(np_input, shift, axis)
            self.assertAllEqual(roll.eval(), expected_roll)

    def _testGradient(self, np_input, shift, axis, use_gpu=False):
        with self.test_session(use_gpu=use_gpu):
            inx = constant_op.constant(np_input.tolist())
            xs = list(np_input.shape)
            y = roll_module.roll(inx, shift, axis, name="roll")
            # Expected y's shape to be the same
            ys = xs
            jacob_t, jacob_n = gradient_checker.compute_gradient(
                inx, xs, y, ys, x_init_value=np_input)
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

    def _testAll(self, np_input, shift, axis, use_gpu=False):
        self._testRoll(np_input, shift, axis, use_gpu)
        if np_input.dtype == np.float32:
            self._testGradient(np_input, shift, axis, use_gpu)

    def testIntTypes(self):
        for t in [np.int32, np.int64]:
            self._testAll(
                np.random.randint(-100, 100, (5)).astype(t), 3, 0)
            self._testAll(
                np.random.randint(-100, 100, (4, 4, 3)).astype(t),
                [1, -2, 3], [0, 1, 2])
            self._testAll(
                np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t),
                [0, 1, -2], [1, 2, 3])

    def testFloatTypes(self):
        for t in [np.float32, np.float64]:
            self._testAll(np.random.rand(5).astype(t), 2, 0)
            self._testAll(np.random.rand(3, 4).astype(t), [1,2], [1,0])
            self._testAll(np.random.rand(1, 3, 4).astype(t), [1,0,-3], [0,1,2])

    def testComplexTypes(self):
        for t in [np.complex64, np.complex128]:
            x = np.random.rand(2, 5).astype(t)
            self._testAll(x + 1j * x, [1,2], [1,0])
            x = np.random.rand(3, 2, 1, 1).astype(t)
            self._testAll(x + 1j * x, [2,1,1,0], [0,3,1,2])


    def testRollInputMustVectorHigherRaises(self):
        tensor = 7
        shift = 1
        axis = 0
        with self.test_session():
            with self.assertRaisesRegexp(InvalidArgumentError, "input must be 1-D or higher"):
                roll_module.roll(tensor, shift, axis).eval()

    def testRollAxisMustBeScalarOrVectorRaises(self):
        tensor = [[1, 2],
                  [ 3, 4]]
        shift = 1
        axis = [[0,1]]
        with self.test_session():
            with self.assertRaisesRegexp(InvalidArgumentError, "axis must be a scalar or a 1-D vector"):
                roll_module.roll(tensor, shift, axis).eval()

    def testRollShiftMustBeScalarOrVectorRaises(self):
        tensor = [[1, 2],
                  [ 3, 4]]
        shift = [[0,1]]
        axis = 1
        with self.test_session():
            with self.assertRaisesRegexp(InvalidArgumentError, "shift must be a scalar or a 1-D vector"):
                roll_module.roll(tensor, shift, axis).eval()

    def testRollShiftAndAxisMustBeSameSizeRaises(self):
        tensor = [[1, 2],
                  [ 3, 4]]
        shift = [1]
        axis = [0,1]
        with self.test_session():
            with self.assertRaisesRegexp(InvalidArgumentError, "shift and axis must be the same size"):
                roll_module.roll(tensor, shift, axis).eval()

    def testRollAxisOutOfRangeRaises(self):
        tensor = [1, 2]
        shift = 1
        axis = 1
        with self.test_session():
            with self.assertRaisesRegexp(InvalidArgumentError, "is out of range"):
                roll_module.roll(tensor, shift, axis).eval()

    # def testIntTypesGPU(self):
    #     for t in [np.int32, np.int64]:
    #         self._testAll(
    #             np.random.randint(-100, 100, (5)).astype(t), 3, 0, use_gpu=True)
    #         self._testAll(
    #             np.random.randint(-100, 100, (4, 4, 3)).astype(t),
    #             [1, -2, 3], [0, 1, 2], use_gpu=True)
    #         self._testAll(
    #             np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t),
    #             [0, 1, -2], [1, 2, 3], use_gpu=True)

    def testFloatTypesGPU(self):
        for t in [np.float32, np.float64]:
            self._testAll(np.random.rand(5).astype(t), 2, 0, use_gpu=True)
            self._testAll(np.random.rand(3, 4).astype(t), [1,2], [1,0], use_gpu=True)
            self._testAll(np.random.rand(1, 3, 4).astype(t), [1,0,-3], [0,1,2], use_gpu=True)

    # def testComplexTypesGPU(self):
    #     for t in [np.complex64, np.complex128]:
    #         x = np.random.rand(2, 5).astype(t)
    #         self._testAll(x + 1j * x, [1,2], [1,0], use_gpu=True)
    #         x = np.random.rand(3, 2, 1, 1).astype(t)
    #         self._testAll(x + 1j * x, [2,1,1,0], [0,3,1,2], use_gpu=True)


if __name__ == "__main__":
    tf.test.main()
