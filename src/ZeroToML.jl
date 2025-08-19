module ZeroToML

using LinearAlgebra, Random, Statistics, Logging

include("activations.jl")
include("encoding.jl")
export build_vocab, encode, decode, positional_encoding

include("transformers.jl")
export LayerNorm, ScaledDotProductAttention, MultiHeadAttention, FeedForward, TransformerBlock, Transformer,
       cross_entropy_loss, cross_entropy_loss_backward,
       backward, backward!, zero_gradients!, update!, Adam, generate, softmax

end
