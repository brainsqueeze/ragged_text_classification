import tensorflow as tf

from ragged_text.layers.word_embed import WordEmbedding
from ragged_text.layers.conv import ConvNgram
from ragged_text import map_ragged_time_sequences


class ConvGramClassifier(tf.keras.Model):

    def __init__(self, vocab: list, embedding_size: int, conv_filter_size: int, ngrams: list, pool_size: int,
                 n_classes: int, multi_label=True):
        super().__init__()

        self.multi_label = multi_label
        self.n_classes = n_classes
        if n_classes <= 2:
            self.n_classes = 1
            self.activation = tf.nn.sigmoid
            self.multi_label = False
        else:
            self.activation = tf.nn.sigmoid if multi_label else tf.nn.softmax

        self.embed = WordEmbedding(vocab=vocab, embedding_size=embedding_size)
        self.ngram_layers = [
            ConvNgram(ngram_size=ng, output_size=conv_filter_size, pool_size=pool_size, embedding_size=embedding_size)
            for ng in ngrams
        ]

        with tf.name_scope("LinearSVC"):
            self.svm = tf.keras.layers.Dense(self.n_classes, name="LinearSvm")
            self.svm.build([None, conv_filter_size * len(ngrams)])

        with tf.name_scope("PlattPosterior"):
            self.platt_dense = tf.keras.layers.Dense(self.n_classes, activation=self.activation)
            self.platt_dense.build([None, self.n_classes])

    def feature_forward(self, tokens):
        tokens = self.embed(tokens)
        x = []
        for feature_layer in self.ngram_layers:
            x_ = map_ragged_time_sequences(feature_layer, tokens)

            # add n-grams as vectors
            x_ = tf.reduce_sum(x_, axis=1)
            x.append(x_)
        x = tf.concat(x, axis=-1)
        return tf.linalg.l2_normalize(x, axis=1)

    def call(self, tokens, svm_output=False):
        with tf.name_scope("ConvGramClassifier"):
            x = self.feature_forward(tokens)
            x = self.svm(x)
            if svm_output:
                return x
            return self.platt_dense(x)

    @property
    def svm_variables(self):
        return [v for v in self.trainable_variables if 'PlattPosterior' not in v.name]

    @property
    def platt_variables(self):
        return self.platt_dense.trainable_variables
