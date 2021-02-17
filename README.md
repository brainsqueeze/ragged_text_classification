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