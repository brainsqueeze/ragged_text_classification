import tensorflow as tf


class WordEmbedding(tf.keras.layers.Layer):
    """Standard word embedding layer, with forward passes specific to `tf.RaggedTensor`
    inputs.

    Parameters
    ----------
    vocab : list
        Token list
    embedding_size : int
        Token embedding dimensionality
    """

    def __init__(self, vocab: list, embedding_size: int):
        super().__init__()

        with tf.name_scope("WordEmbedding"):
            self.table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=vocab,
                    values=list(range(len(vocab))),
                    value_dtype=tf.int64
                ),
                default_value=len(vocab)
            )
            self.embeddings = tf.Variable(
                tf.random.uniform([len(vocab) + 1, embedding_size], -1.0, 1.0),
                name='embeddings',
                dtype=tf.float32,
                trainable=True
            )

    def __call__(self, tokens):
        tokens = tf.ragged.map_flat_values(self.table.lookup, tokens)
        return tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, tokens)
