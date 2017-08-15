import tensorflow as tf
import os

class RollTest(tf.test.TestCase):
    def testRoll(self):
        roll_module = tf.load_op_library("./roll.so")
        with self.test_session():
            result = roll_module.roll([5, 4, 3, 2, 1])
            self.assertAllEqual(result.eval(), [1, 5, 4, 3, 2])

if __name__ == "__main__":
    tf.test.main()
