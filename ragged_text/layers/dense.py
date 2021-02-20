import tensorflow as tf


class PointwiseLinear(tf.keras.layers.Layer):
    """Calibrated posterior probabilities, ignoring off-diagonal mixing weights.

    Parameters
    ----------
    units : int
        Output dimensions
    activation : Tensorflow activation function
        Activation function, typically sigmoid or softmax functions
    """

    def __init__(self, units: int, activation):
        super().__init__()

        self.activation = activation
        self.kernel = tf.Variable(
            tf.random.truncated_normal((units,), mean=-0.01, stddev=0.01),
            name="kernel",
            dtype=tf.float32,
            trainable=True
        )
        self.bias = tf.Variable(tf.zeros(shape=(units,)), name="bias", dtype=tf.float32, trainable=True)

    def __call__(self, X):
        return self.activation(X * self.kernel + self.bias)


class LinearSvmPlatt(tf.keras.layers.Layer):
    """Linear SVM + calibrated posterior probabilities via Platt scaling.

    Parameters
    ----------
    n_features : int
        Input dimensionality
    n_classes : int
        Number of class labels
    multi_label : bool, optional
        Set to True to perform multi-label classification, will be overriden if n_classes <= 2, by default True
    lite : bool, optional
        Set to True to use the `PointwiseLinear` layer and ignore off-diagonal calibration weights, by default False

    Raises
    ------
    TypeError
        Raised for invalid `n_classes` values; must be an integer >= 1
    """

    def __init__(self, n_features: int, n_classes: int, multi_label=True, lite=False):
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
            params = dict(units=self.n_classes, activation=self.activation)
            if lite:
                self.platt = PointwiseLinear(**params)
            else:
                self.platt = tf.keras.layers.Dense(**params)
            self.platt.build([self.n_classes, self.n_classes])

    def __call__(self, X, svm_output=False, training=False):
        X = self.svm(X)
        if svm_output:
            return X
        return self.platt(X)
