import tensorflow as tf

from ragged_text.layers.dense import LinearSvmPlatt
from ragged_text.layers.word_embed import WordEmbedding
from ragged_text.layers.conv import ConvNgram
from ragged_text import map_ragged_time_sequences


class ConvGramClassifier(tf.keras.Model):
    """Classifier model which abstracts n-gram BOW type models by using 1D convolutions of
    varying window sizes. Features are then fed through two dense output layers which are designed
    to mimic a Linear SVM model with calibrated posterior probabilities.

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
    n_classes : int
        Number of class labels
    multi_label : bool, optional
        Set to True to perform multi-label classification, will be overriden in n_classes <= 2, by default False
    """

    def __init__(self, vocab: list, embedding_size: int, conv_filter_size: int, ngrams: list, pool_size: int,
                 n_classes: int, multi_label=False):
        super().__init__()

        with tf.name_scope('ConvFeatures'):
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

        # with tf.name_scope("LinearSVC"):
        #     self.svm = tf.keras.layers.Dense(self.n_classes, name="LinearSvm")
        #     self.svm.build([None, conv_filter_size * len(ngrams)])

        # with tf.name_scope("PlattPosterior"):
        #     self.platt_dense = tf.keras.layers.Dense(self.n_classes, activation=self.activation)
        #     self.platt_dense.build([None, self.n_classes])

        with tf.name_scope('Classifier'):
            self.classifier = LinearSvmPlatt(
                n_features=conv_filter_size * len(ngrams),
                n_classes=n_classes, 
                multi_label=multi_label
            )

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
            return self.classifier(x, svm_output=svm_output)

    @property
    def multi_label(self):
        return self.classifier.multi_label

    @property
    def n_classes(self):
        return self.classifier.n_classes

    @property
    def svm_variables(self):
        return [v for v in self.trainable_variables if 'PlattPosterior' not in v.name]

    @property
    def platt_variables(self):
        return self.classifier.platt.trainable_variables
