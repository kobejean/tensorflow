import tensorflow as tf
from tensorflow.python.framework import ops

roll_module = tf.load_op_library("./roll.so")

@ops.RegisterGradient("Roll")
def _RollGrad(op, grad):
    """Gradient for Roll."""
    # The gradient should be just the roll reversed
    shift = op.inputs[1]
    axis = op.inputs[2]

    roll_grad = roll_module.roll(grad, -shift, axis)
    return roll_grad, None, None
