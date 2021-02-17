import tensorflow as tf
import sys


class ConvNgram(tf.keras.layers.Layer):

    def __init__(self, ngram_size: int, embedding_size: int, output_size: int, pool_size: int):
        super().__init__()

        self.scope = f"ConvNgram-{ngram_size}-gram"
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
        tf.print(embedded_tokens, output_stream=sys.stderr)
        with tf.name_scope(self.scope):
            x = self.ngram(embedded_tokens)
            x = self.pool(x)
            return x
