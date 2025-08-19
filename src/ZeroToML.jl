module ZeroToML

using LinearAlgebra, Random, Statistics, Logging

include("encoding.jl")
export build_vocab, encode, decode

include("transformers.jl")
export LayerNorm, ScaledDotProductAttention, MultiHeadAttention, FeedForward, TransformerBlock, Transformer,
       softmax, positional_encoding, cross_entropy_loss, cross_entropy_loss_backward,
       backward, backward!, zero_gradients!, update!, Adam, generate

end
