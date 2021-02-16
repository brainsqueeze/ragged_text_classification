import tensorflow as tf

from ragged_text.models.conv_classifier import ConvGramClassifier


def platt_scale(labels):
    labels = tf.cast(labels, tf.float32)
    n_plus = tf.reduce_sum(labels, axis=0)
    n_minus = tf.reduce_sum(1 - labels, axis=0)
    n_plus_rate = (n_plus + 1) / (n_plus + 2)
    n_minus_rate = 1 / (n_minus + 2)
    return n_plus_rate * labels + n_minus_rate * (1 - labels)


def svm_platt_multi_label_train_step(model: ConvGramClassifier, opt: tf.keras.optimizers.Optimizer, tokens, labels):
    with tf.GradientTape() as tape:
        logits = model(tokens, svm_output=True)
        svm_loss = tf.keras.losses.hinge(y_true=labels, y_pred=logits)
    gradients = tape.gradient(svm_loss, model.svm_variables)
    opt.apply_gradients(gradients, model.svm_variables)

    with tf.GradientTape() as tape:
        y_hat = model(tokens)
        platt_loss = tf.keras.losses.categorical_crossentropy(y_true=platt_scale(labels), y_pred=y_hat)
    gradients = tape.gradient(platt_loss, model.platt_variables)
    opt.apply_gradients(gradients, model.platt_variables)
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
