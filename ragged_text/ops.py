import tensorflow as tf


def map_ragged_time_sequences(op, rt: tf.RaggedTensor, **kwargs) -> tf.RaggedTensor:
    rt = tf.map_fn(
        lambda x: tf.RaggedTensor.from_tensor(op(tf.expand_dims(x, axis=0)), ragged_rank=1),
        elems=rt,
        fn_output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.float32)
    )

    # acts like tf.squeeze, end up with extra dim due to tf.map_fn
    return rt.merge_dims(0, 1)
