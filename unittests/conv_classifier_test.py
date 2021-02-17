from collections import Counter
import tensorflow as tf
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import WordPunctTokenizer

from ragged_text.models.conv_classifier import ConvGramClassifier
from . import TEXT

word_tokenize = WordPunctTokenizer().tokenize
sent_tokenize = PunktSentenceTokenizer().tokenize
tokens = [word_tokenize(sent.lower()) for sent in sent_tokenize(TEXT)]
vocab = Counter([tok for record in tokens for tok in record])

model = ConvGramClassifier(
    vocab=list(vocab.keys()),
    embedding_size=32,
    conv_filter_size=16,
    ngrams=[1, 2],
    pool_size=3,
    n_classes=3,
    multi_label=True
)
model(tf.ragged.constant(tokens))
print(model)
