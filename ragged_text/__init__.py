import tensorflow as tf
from .ops import map_ragged_time_sequences

if tf.__version__ < '2.4.1':
    @tf.RegisterGradient("RaggedTensorFromVariant")
    def _RaggedTensorFromVariantGrad(*args):
        if len(args) == 2:
            op, grad = args
            res = [tf.raw_ops.RaggedTensorToVariant(rt_nested_splits=[], rt_dense_values=grad, batched_input=False)]
        else:
            op, *empty, grad = args
            res = [tf.raw_ops.RaggedTensorToVariant(
                rt_nested_splits=[op.outputs[0]],
                rt_dense_values=grad,
                batched_input=True
            )]
        return res
