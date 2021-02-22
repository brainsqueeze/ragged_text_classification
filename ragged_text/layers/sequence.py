import tensorflow as tf
import text2vec as t2v


class Text2VecAttentionEmbed(tf.keras.layers.Layer):
    """Attention-based layer for feature encoding from text2vec.

    Parameters
    ----------
    tokens : dict
        Vocabulary tokens, tok -> int
    embedding_size : int, optional
        Token embedding dimensionality, also the dimensionality of the output feature vectors, by default 128
    max_sequence_len : int, optional
        Maximum number of tokens in input sequences, by default 512
    input_keep_prob : float, optional
        Keep probability (dropout) for the input layer, by default 1.0
    hidden_keep_prob : float, optional
        Keep probability (dropout) for the hidden layers, by default 1.0
    sep : str, optional
        Token separator to split on, by default ' '
    """

    def __init__(self, tokens: dict, embedding_size=128, max_sequence_len=512,
                 input_keep_prob=1.0, hidden_keep_prob=1.0, sep=' '):
        super().__init__()

        with tf.name_scope('text2vec'):
            self.tokenizer = t2v.models.Tokenizer(sep)
            self.word_embedder = t2v.models.TextInput(
                token_hash=tokens,
                embedding_size=embedding_size,
                max_sequence_len=max_sequence_len
            )
            self.encoder = t2v.models.transformer.TransformerEncoder(
                max_sequence_len=max_sequence_len,
                layers=8,
                n_stacks=1,
                embedding_size=embedding_size,
                input_keep_prob=input_keep_prob,
                hidden_keep_prob=hidden_keep_prob
            )

    def __call__(self, sentences, training=False):
        with tf.name_scope("Text2VecAttentionEmbed"):
            tokens = self.tokenizer(sentences)
            sequences, mask, _ = self.word_embedder(tokens)

            if training:
                _, X = self.encoder(sequences, mask, training=training)
            else:
                X = self.encoder(sequences, mask, training=training)

            return X
