from collections import Counter
import tensorflow as tf
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import WordPunctTokenizer

from ragged_text.layers.word_embed import WordEmbedding
from ragged_text.layers.conv import ConvNgram
from ragged_text import map_ragged_time_sequences

BATCH_SIZE = 5
EMBEDDING_SIZE = 32
OUTPUT_DIMS = 16
POOL_SIZE = 5
# Masterpiece by Big Thief
TEXT = """
Years, days, makes no difference to me, babe
You look exactly the same to me.
Ain’t no time, crossing your legs inside the diner
Raising your coffee to your lips, the steam.

You saw the masterpiece, she looks a lot like you.
Wrapping her left arm around your right
Ready to walk you through the night.

You whisper to a restless ear, “can you get me out of here?
This place smells like piss and beer, can you get me out?”
You were asking me how to get you free
I only know the recipe to roam.

You saw the masterpiece, she looks a lot like you.
Wrapping her left arm around your right
Ready to walk you through the night.

Old stars filling up my throat
You gave em to me when I was born, now they’re coming out.
Laying there on the hospital bed, eyes were narrow, blue and red
You took a draw of breath and said to me:

You saw the masterpiece, she looks a lot like me.
Wrapping my left arm around your right
Ready to walk you through the night.

Old friends, old mothers, dogs and brothers,
There’s only so much letting go you can ask someone to do.
So I keep you by my side, I will not give you to the tide
I'll even walk you in my stride, Marie.

'Cause I saw the masterpiece, she looks a lot like you.
Wrapping your left arm around my right
Ready to walk me through the night.
"""

word_tokenize = WordPunctTokenizer().tokenize
sent_tokenize = PunktSentenceTokenizer().tokenize
tokens = [word_tokenize(sent.lower()) for sent in sent_tokenize(TEXT)]
vocab = Counter([tok for record in tokens for tok in record])
embed = WordEmbedding(vocab=list(vocab.keys()), embedding_size=EMBEDDING_SIZE)
uni_grams = ConvNgram(ngram_size=1, output_size=OUTPUT_DIMS, pool_size=POOL_SIZE)

X = embed(tf.ragged.constant(tokens))
X = map_ragged_time_sequences(uni_grams, X)
print(X.shape)
