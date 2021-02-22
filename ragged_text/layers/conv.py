import tensorflow as tf


class ConvNgram(tf.keras.layers.Layer):
    """Conv1D + MaxPool1D layer

    Parameters
    ----------
    ngram_size : int
        Window size for the convolution filter (how many time steps to convolve over)
    embedding_size : int
        Token embedding dimensionality
    output_size : int
        Convolution output dimensionality
    pool_size : int
        Window size for the maxpool layer (how many n-grams to consider)
    """

    def __init__(self, ngram_size: int, embedding_size: int, output_size: int, pool_size: int):
        super().__init__(name=f"Conv1D-{ngram_size}")

        self.scope = f"ConvNgram-{ngram_size}-gram"
        with tf.name_scope(self.scope):
            self.ngram = tf.keras.layers.Conv1D(
                filters=output_size,
                kernel_size=ngram_size,
                padding='valid',
                activation='relu',
                input_shape=[None, embedding_size]
            )
            self.ngram.build([None, embedding_size])

            self.pool = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='valid')
            self.pool.build([None, output_size])

    def __call__(self, embedded_tokens):
        with tf.name_scope(self.scope):
            x = self.ngram(embedded_tokens)
            x = self.pool(x)
            return x
