module ZeroToML

using LinearAlgebra, Random, Statistics, Logging

include("encoding.jl")
export build_vocab, encode, decode, positional_encoding, positional_encoding_tullio

include("transformers.jl")
export LayerNorm, ScaledDotProductAttention, MultiHeadAttention, FeedForward, TransformerBlock, Transformer,
       softmax, cross_entropy_loss, cross_entropy_loss_backward,
       backward, backward!, zero_gradients!, update!, Adam, generate

end
