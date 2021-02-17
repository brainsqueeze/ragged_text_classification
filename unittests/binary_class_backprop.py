from collections import Counter
import tensorflow as tf
from nltk.corpus import stopwords
from ragged_text.models.conv_classifier import ConvGramClassifier
from ragged_text.train import svm_platt_binary_train_step

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
token_lookup = tf.keras.datasets.imdb.get_word_index()
inv_token_lookup = {v: k for k, v in token_lookup.items()}
stop_words = set(stopwords.words('english'))
batch_size = 256

# re-map this to use the static hash table
x_train = [[inv_token_lookup.get(t, '<unk>').lower() for t in doc] for doc in x_train]
x_test = [[inv_token_lookup.get(t, '<unk>').lower() for t in doc] for doc in x_test]
vocab = Counter([t for doc in x_train for t in doc if t not in stop_words])

optimizer = tf.keras.optimizers.Adam()
model = ConvGramClassifier(
    vocab=[tok for tok, _ in vocab.most_common(100000)],
    embedding_size=64,
    conv_filter_size=16,
    ngrams=[1, 2, 3],
    pool_size=3,
    n_classes=2,
    multi_label=False
)
svm_loss = tf.keras.metrics.Mean('svm-train-loss', dtype=tf.float32)
platt_loss = tf.keras.metrics.Mean('platt-train-loss', dtype=tf.float32)


@tf.function(input_signature=[
    tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.string),
    tf.TensorSpec(shape=[None], dtype=tf.float32)
])
def train_step(tokens, labels):
    svm, platt = svm_platt_binary_train_step(model, optimizer, tokens=tokens, labels=labels)
    svm_loss(svm)
    platt_loss(platt)


for epoch in range(10):
    print(f'\n------ Epoch {epoch + 1} ------\n')
    for i, b in enumerate(range(0, len(x_train) + batch_size, batch_size), start=1):
        if y_train[b: b + batch_size].shape[0] == 0:
            continue
        train_step(tokens=tf.ragged.constant(x_train[b: b + batch_size]), labels=y_train[b: b + batch_size])

        if i % 10 == 0:
            print(f"\tStep {i} --- SVM Loss: {svm_loss.result()} Platt posterior Loss: {platt_loss.result()}")
            svm_loss.reset_states()
            platt_loss.reset_states()
