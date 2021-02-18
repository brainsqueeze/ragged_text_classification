import tensorflow as tf

from ragged_text.models.conv_classifier import ConvGramClassifier


def platt_scale(labels):
    labels = tf.cast(labels, tf.float32)
    n_plus = tf.reduce_sum(labels, axis=0)
    n_minus = tf.reduce_sum(1 - labels, axis=0)
    n_plus_rate = (n_plus + 1) / (n_plus + 2)
    n_minus_rate = 1 / (n_minus + 2)
    return n_plus_rate * labels + n_minus_rate * (1 - labels)


def svm_platt_train_step(model: ConvGramClassifier, opt: tf.keras.optimizers.Optimizer, tokens, labels):
    """Main backprop step per batch for the SVM+Platt posterior calibration model.

    Parameters
    ----------
    model : ConvGramClassifier
    opt : tf.keras.optimizers.Optimizer
    tokens : tf.RaggedTensor
        Input tokens, ragged_rank=1
    labels : Array-like
        Targets, accepts `np.ndarray` or `tf.Tensor` objects

    Returns
    -------
    tuple
        (SVM loss, Platt parameter loss)
    """

    # if 1-class predictions then re-shape the targets to (N, 1)
    if model.n_classes == 1:
        labels = tf.reshape(labels, shape=[-1, 1])

    # back propagation starting from SVM layer
    with tf.GradientTape() as tape:
        logits = model(tokens, svm_output=True)
        svm_loss = tf.reduce_mean(tf.keras.losses.hinge(y_true=labels, y_pred=logits))
    gradients = tape.gradient(svm_loss, model.svm_variables)
    opt.apply_gradients(zip(gradients, model.svm_variables))

    # back propagation from output layer to SVM layer
    with tf.GradientTape() as tape:
        y_hat = model(tokens)
        if not model.multi_label:
            platt_loss = tf.keras.losses.binary_crossentropy(y_true=platt_scale(labels), y_pred=y_hat)
        else:
            platt_loss = tf.keras.losses.categorical_crossentropy(y_true=platt_scale(labels), y_pred=y_hat)
        platt_loss = tf.reduce_mean(platt_loss)
    gradients = tape.gradient(platt_loss, model.platt_variables)
    opt.apply_gradients(zip(gradients, model.platt_variables))
    return svm_loss, platt_loss


def svm_platt_multi_label_train_step(model: ConvGramClassifier, opt: tf.keras.optimizers.Optimizer, tokens, labels):
    with tf.GradientTape() as tape:
        logits = model(tokens, svm_output=True)
        svm_loss = tf.reduce_mean(tf.keras.losses.hinge(y_true=labels, y_pred=logits))
    gradients = tape.gradient(svm_loss, model.svm_variables)
    opt.apply_gradients(zip(gradients, model.svm_variables))

    with tf.GradientTape() as tape:
        y_hat = model(tokens)
        platt_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=platt_scale(labels), y_pred=y_hat))
    gradients = tape.gradient(platt_loss, model.platt_variables)
    opt.apply_gradients(zip(gradients, model.platt_variables))
    return svm_loss, platt_loss


def svm_platt_binary_train_step(model: ConvGramClassifier, opt: tf.keras.optimizers.Optimizer, tokens, labels):
    labels = tf.reshape(labels, shape=[-1, 1])
    with tf.GradientTape() as tape:
        logits = model(tokens, svm_output=True)
        svm_loss = tf.reduce_mean(tf.keras.losses.hinge(y_true=labels, y_pred=logits))
    gradients = tape.gradient(svm_loss, model.svm_variables)
    opt.apply_gradients(zip(gradients, model.svm_variables))

    with tf.GradientTape() as tape:
        y_hat = model(tokens)
        platt_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=platt_scale(labels), y_pred=y_hat))
    gradients = tape.gradient(platt_loss, model.platt_variables)
    opt.apply_gradients(zip(gradients, model.platt_variables))
    return svm_loss, platt_loss
