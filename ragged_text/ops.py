import tensorflow as tf


def map_ragged_time_sequences(op, rt: tf.RaggedTensor, **kwargs) -> tf.RaggedTensor:
    """Handles op mapping of `tf.RaggedTensor` inputs.

    Parameters
    ----------
    op : Tensorflow callable
  
    rt : tf.RaggedTensor

    Returns
    -------
    tf.RaggedTensor
    """

    if len(tf.config.list_physical_devices('GPU')) > 0:
        # rt = tf.ragged.stack([op(tf.expand_dims(x, axis=0), **kwargs) for x in rt], axis=0)
        return op(rt.to_tensor(), **kwargs)

    rt = tf.map_fn(
        lambda x: tf.RaggedTensor.from_tensor(op(tf.expand_dims(x, axis=0), **kwargs), ragged_rank=1),
        elems=rt,
        fn_output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.float32)
    )

    # acts like tf.squeeze, end up with extra dim due to tf.map_fn
    return rt.merge_dims(0, 1)
