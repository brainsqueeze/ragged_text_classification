# ragged_text_classification

This project is intended to provide a mechanism to leverage `tf.RaggedTensor` objects **fully**
to do basic text classification tasks, while avoiding any conversions of batches to a dense, padded 
`tf.Tensor` where possible. By maintaining a ragged structure throughout the forward pass we can hopefully maintain a lower
memory profile, reduce the number of weights in a given model and allow for any sequence length,
without bound.

## Caveats

Parallelizing the `Conv1D` operations on each sequence in a ragged tensor is performant on CPU
bound models, however underperforms on a GPU. For this reason convolutions are vectorized when running
on a GPU by calling `rt.to_tensor()`. This may cause numerical inconsistencies between GPU/CPU bound models since the GPU model will be convolving over some 0-pad values.


## Getting started

To install this library run

```bash
pip install git+https://github.com/brainsqueeze/ragged_text_classification.git
```

### ConvNet classifier

The convnet-based classifier is an abstraction of n-gram generation from bag-of-word models. This
uses a standard word embedding and then performs 1D convolutions down the time steps, with a window
size determined by the user. The window size can be thought of as the value `n` in n-grams.

The core model here can be extended to use the `fit` method for Keras models.
```python
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
```

Suppose one would like to use this model on the IMDB binary classification task, one would begin by
loading the data and building the vocabulary.
```python
from collections import Counter
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
token_lookup = tf.keras.datasets.imdb.get_word_index()
inv_token_lookup = {v: k for k, v in token_lookup.items()}

# IMDB data comes as the token ID, we will convert it back to original token string
x_train = [[inv_token_lookup.get(tok, '<unk>').lower() for tok in doc] for doc in x_train]
x_test = [[inv_token_lookup.get(tok, '<unk>').lower() for tok in doc] for doc in x_test]
vocab = Counter([tok for doc in x_train for tok in doc])
```

Once the data is pre-processed the model can be initialized with the top 100k vocab terms
```python
# definte the model
model = ClassifierModel(
    vocab=[tok for tok, _ in vocab.most_common(100000)],
    embedding_size=64,
    conv_filter_size=16,
    ngrams=[1, 2, 3],
    pool_size=3,
    n_classes=2
)

# compile the model with training metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=[
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)
```

then fit the model to the data in batches of 256 examples, over 5 epochs
```python
import numpy as np

model.fit(
    x=np.array([' '.join(d) for d in x_train]),
    y=y_train,
    batch_size=256,
    epochs=5
)
```

### Stopword improvements

To improve the model one can remove stop words from the vocabulary prior to building the model
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
vocab = Counter([tok for doc in x_train for tok in doc if tok not in stop_words])
```

Using the same compile/fit parameters as before, one should get a training set F1 score of 0.997.