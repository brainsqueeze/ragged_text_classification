from collections import Counter
import tensorflow as tf
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import WordPunctTokenizer

from ragged_text.layers.word_embed import WordEmbedding
from ragged_text.layers.conv import ConvNgram
from ragged_text import map_ragged_time_sequences
from . import TEXT

BATCH_SIZE = 5
EMBEDDING_SIZE = 32
OUTPUT_DIMS = 16
POOL_SIZE = 5


word_tokenize = WordPunctTokenizer().tokenize
sent_tokenize = PunktSentenceTokenizer().tokenize
tokens = [word_tokenize(sent.lower()) for sent in sent_tokenize(TEXT)]
vocab = Counter([tok for record in tokens for tok in record])
embed = WordEmbedding(vocab=list(vocab.keys()), embedding_size=EMBEDDING_SIZE)
uni_grams = ConvNgram(ngram_size=1, output_size=OUTPUT_DIMS, pool_size=POOL_SIZE)

X = embed(tf.ragged.constant(tokens))
X = map_ragged_time_sequences(uni_grams, X)
print(X.shape)
