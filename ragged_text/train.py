import tensorflow as tf


PLATT_REQUIRED_ATTRIBUTES = ['svm_variables', 'platt_variables', 'multi_label', 'n_classes']


def platt_scale(labels) -> tf.Tensor:
    """Performs Platt scaling on input labels (https://en.wikipedia.org/wiki/Platt_scaling).
    Labels are expected to be binary or one-hot-encoded.

    Parameters
    ----------
    labels : array-like, able to be converted to a `tf.Tensor` object
        [description]

    Returns
    -------
    tf.Tensor
        Shape matches the shape of the input labels.
    """

    labels = tf.cast(labels, tf.float32)
    n_plus = tf.reduce_sum(labels, axis=0)
    n_minus = tf.reduce_sum(1 - labels, axis=0)
    n_plus_rate = (n_plus + 1) / (n_plus + 2)
    n_minus_rate = 1 / (n_minus + 2)
    return n_plus_rate * labels + n_minus_rate * (1 - labels)


def svm_platt_train_step(model: tf.keras.Model, opt: tf.keras.optimizers.Optimizer, inputs, labels):
    """Main backprop step per batch for the SVM+Platt posterior calibration model.

    Parameters
    ----------
    model : tf.keras.Model
    opt : tf.keras.optimizers.Optimizer
    inputs : tf.RaggedTensor
        Input layer, ragged_rank=1
    labels : Array-like
        Targets, accepts `np.ndarray` or `tf.Tensor` objects

    Returns
    -------
    tuple
        (SVM loss, Platt parameter loss)

    Raises
    ------
    AttributeError
        Raised if `model` is missing one or more variables attributes
    """

    if not all([hasattr(model, attr) for attr in PLATT_REQUIRED_ATTRIBUTES]):
        raise AttributeError(f"{model} is missing one or more required attributes: {PLATT_REQUIRED_ATTRIBUTES}")

    # if 1-class predictions then re-shape the targets to (N, 1)
    if model.n_classes == 1:
        labels = tf.reshape(labels, shape=[-1, 1])

    # back propagation starting from SVM layer
    with tf.GradientTape() as tape:
        logits = model(inputs, svm_output=True)
        if not model.multi_label:
            svm_loss = tf.keras.losses.hinge(y_true=labels, y_pred=logits)
        else:
            svm_loss = tf.keras.losses.categorical_hinge(y_true=labels, y_pred=logits)
        svm_loss = tf.reduce_mean(loss)
    gradients = tape.gradient(svm_loss, model.svm_variables)
    opt.apply_gradients(zip(gradients, model.svm_variables))

    # back propagation from output layer to SVM layer
    with tf.GradientTape() as tape:
        y_hat = model(inputs)
        if not model.multi_label:
            platt_loss = tf.keras.losses.binary_crossentropy(y_true=platt_scale(labels), y_pred=y_hat)
        else:
            platt_loss = tf.keras.losses.categorical_crossentropy(y_true=platt_scale(labels), y_pred=y_hat)
        platt_loss = tf.reduce_mean(platt_loss)
    gradients = tape.gradient(platt_loss, model.platt_variables)
    opt.apply_gradients(zip(gradients, model.platt_variables))
    return svm_loss, platt_loss, y_hat
