import tensorflow as tf
import os

# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# g++ -std=c++11 -shared roll.cc -o roll.so -fPIC -I $TF_INC -O2 -undefined dynamic_lookup
# python3 roll_test.py

class RollTest(tf.test.TestCase):
    def testRoll(self):
        roll_module = tf.load_op_library("./roll.so")
        with self.test_session():
            x1 = [ 5, 4, 3, 2, 1]
            y1 = [ 1, 5, 4, 3, 2]
            result1 = roll_module.roll(x1,1,0)
            self.assertAllEqual(result1.eval(), y1)


            x2 = [[10, 9, 8, 7, 6],
                  [ 5, 4, 3, 2, 1]]
            y2 = [[ 2, 1, 5, 4, 3],
                  [ 7, 6,10, 9, 8]]
            result2 = roll_module.roll(x2,[2,1],[1,0])
            self.assertAllEqual(result2.eval(), y2)

if __name__ == "__main__":
    tf.test.main()
