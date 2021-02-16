# ragged_text_classification

This project is intended to provide a mechanism to leverage `tf.RaggedTensor` objects **fully**
to do basic text classification tasks, while avoiding any conversions of batches to a dense, padded `tf.Tensor`. 
By maintaining a ragged structure throughout the forward pass we can hopefully maintain a lower
memory profile, reduce the number of weights in a given model and allow for any sequence length,
without bound.