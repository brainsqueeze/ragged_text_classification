import tensorflow as tf


class LinearSvmPlatt(tf.keras.layers.Layer):

    def __init__(self, n_features: int, n_classes: int, multi_label=True):
        super().__init__()

        if not all([isinstance(n_classes, int), n_classes >= 1]):
            raise TypeError("`n_classes` must be an integer greater than or equal to 1")

        self.activation = tf.nn.sigmoid
        self.multi_label = multi_label
        self.n_classes = n_classes
        if n_classes in {1, 2}:
            self.n_classes = 1
            self.multi_label = False
        else:
            if not multi_label:
                self.activation = tf.nn.softmax

        with tf.name_scope('LinearSVM'):
            self.svm = tf.keras.layers.Dense(units=self.n_classes)
            self.svm.build([None, n_features])
        with tf.name_scope('PlattPosterior'):
            self.platt = tf.keras.layers.Dense(units=self.n_classes, activation=self.activation)
            self.platt.build([self.n_classes, self.n_classes])

    def __call__(self, X, svm_output=False, training=False):
        X = self.svm(X)
        if svm_output:
            return X
        return self.platt(X)
