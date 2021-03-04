from re import I
import tensorflow as tf
from text2vec.models import Tokenizer

from ragged_text import map_ragged_time_sequences
from .conv import ConvNgram
from .word_embed import WordEmbedding


class Inception(tf.keras.layers.Layer):
    """Feature building layer which abstracts n-gram BOW type models by using 1D convolutions of
    varying window sizes.

    Parameters
    ----------
    vocab : list
        Token list
    embedding_size : int
        Token embedding dimensionality
    conv_filter_size : int
        Output dimensionality of each of the 1D convolutional layers
    ngrams : list
        List of values of `n` in n-grams (ie. [1, 2, 3] will produce uni-grams, bi-grams and tri-grams)
    pool_size : int
        Window size (groups of n-grams) for the maxpool layer(s)
    sep : str, optional
        Token separator to split on, by default ' '
    """

    def __init__(self, vocab: list, embedding_size: int, conv_filter_size: int, ngrams: list, pool_size: int, sep=' '):
        super().__init__()

        with tf.name_scope('ConvFeatures'):
            self.tok = Tokenizer(sep)
            self.embed = WordEmbedding(vocab=vocab, embedding_size=embedding_size)
            self.ngram_layers = [
                ConvNgram(
                    ngram_size=ng,
                    output_size=conv_filter_size,
                    pool_size=pool_size,
                    embedding_size=embedding_size
                )
                for ng in ngrams
            ]

    def __call__(self, documents):
        with tf.name_scope('Inception'):
            tokens = self.embed(self.tok(documents))
            x = []
            for feature_layer in self.ngram_layers:
                x_ = map_ragged_time_sequences(feature_layer, tokens)

                # add n-grams as vectors
                x_ = tf.reduce_sum(x_, axis=1)
                x.append(x_)
            x = tf.concat(x, axis=-1)
