import tensorflow as tf
from text2vec.models import Tokenizer

from ragged_text.layers.dense import LinearSvmPlatt
from ragged_text.layers.word_embed import WordEmbedding
from ragged_text.layers.conv import ConvNgram
from ragged_text.layers.sequence import Text2VecAttentionEmbed

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
        Set to True to perform multi-label classification, will be overriden if n_classes <= 2, by default False
    lite : bool, optional
        Set to True to use a point-wise dense layer for Platt probability calibrartion. This is useful when
        `n_classes` is large, by default False
    sep : str, optional
        Token separator to split on, by default ' '
    """

    def __init__(self, vocab: list, embedding_size: int, conv_filter_size: int, ngrams: list, pool_size: int,
                 n_classes: int, multi_label=False, lite=False, sep=' '):
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

        with tf.name_scope('Classifier'):
            self.classifier = LinearSvmPlatt(
                n_features=conv_filter_size * len(ngrams),
                n_classes=n_classes,
                multi_label=multi_label,
                lite=lite
            )

    def feature_forward(self, documents):
        tokens = self.embed(self.tok(documents))
        x = []
        for feature_layer in self.ngram_layers:
            x_ = map_ragged_time_sequences(feature_layer, tokens)

            # add n-grams as vectors
            x_ = tf.reduce_sum(x_, axis=1)
            x.append(x_)
        x = tf.concat(x, axis=-1)
        return tf.linalg.l2_normalize(x, axis=1)

    def call(self, documents, svm_output=False):
        with tf.name_scope("ConvGramClassifier"):
            x = self.feature_forward(documents)
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


class ContextClassifier(tf.keras.Model):
    """Classifier model which uses the text2vec (https://github.com/brainsqueeze/text2vec) encoding layers to encode a
    feature space. Features are then fed through two dense output layers which are designed to mimic a Linear SVM model
    with calibrated posterior probabilities.

    Parameters
    ----------
    vocab : list
        Token list
    embedding_size : int
        Token embedding dimensionality
    n_classes : int
        Number of class labels
    conv_filter_size : int
        Output dimensionality of each of the 1D convolutional layers
    max_sequence_len : int, optional
        Longest sequence seen at training time, by default 512
    input_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.
    hidden_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.
    multi_label : bool, optional
        Set to True to perform multi-label classification, will be overriden if n_classes <= 2, by default False
    lite : bool, optional
        Set to True to use a point-wise dense layer for Platt probability calibrartion. This is useful when
        `n_classes` is large, by default False
    sep : str, optional
        Token separator to split on, by default ' '
    """

    def __init__(self, vocab: list, embedding_size: int, n_classes: int, max_sequence_len=512,
                 input_keep_prob=1.0, hidden_keep_prob=1.0, multi_label=True, lite=False, sep=' '):
        super().__init__()

        self.embedder = Text2VecAttentionEmbed(
            sep=sep,
            tokens={tok: i for i, tok in enumerate(vocab)},
            embedding_size=embedding_size,
            max_sequence_len=max_sequence_len,
            input_keep_prob=input_keep_prob,
            hidden_keep_prob=hidden_keep_prob
        )

        with tf.name_scope('classifier'):
            self.classifier = LinearSvmPlatt(
                n_features=embedding_size,
                n_classes=n_classes,
                multi_label=multi_label,
                lite=lite
            )

    def __call__(self, documents, svm_output=False, training=False):
        X = tf.ragged.map_flat_values(self.embedder, documents, training=training)
        X = tf.reduce_sum(X, axis=1)  # add up context vectors for each document
        X = tf.linalg.l2_normalize(X, axis=1)  # get L2 normalized document vectors
        return self.classifier(X, svm_output=svm_output, training=training)

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
