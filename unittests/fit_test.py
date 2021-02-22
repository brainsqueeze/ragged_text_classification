from collections import Counter
import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords
from ragged_text.models.linear_classifiers import ConvGramClassifier
from ragged_text.train import svm_platt_train_step


class ClassifierModel(ConvGramClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self, data):
        documents, labels = data
        svm, platt, y_hat = svm_platt_train_step(self, self.optimizer, inputs=documents, labels=labels)
        self.compiled_metrics.update_state(labels, y_hat)
        return {'svm_loss': svm, 'platt_loss': platt, **{m.name: m.result() for m in self.metrics}}


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
token_lookup = tf.keras.datasets.imdb.get_word_index()
inv_token_lookup = {v: k for k, v in token_lookup.items()}
stop_words = set(stopwords.words('english'))
batch_size = 256

# re-map this to use the static hash table
x_train = [[inv_token_lookup.get(t, '<unk>').lower() for t in doc] for doc in x_train]
x_test = [[inv_token_lookup.get(t, '<unk>').lower() for t in doc] for doc in x_test]
vocab = Counter([t for doc in x_train for t in doc if t not in stop_words])

model = ClassifierModel(
    vocab=[tok for tok, _ in vocab.most_common(100000)],
    embedding_size=64,
    conv_filter_size=16,
    ngrams=[1, 2, 3],
    pool_size=3,
    n_classes=2,
    multi_label=False,
    lite=False
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)
model.fit(
    x=np.array([' '.join(d) for d in x_train]),
    y=y_train,
    batch_size=batch_size,
    epochs=5
)
