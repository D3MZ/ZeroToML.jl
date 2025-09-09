module ZeroToML

using Statistics, StatsBase, Random, Logging, LinearAlgebra, Zygote

# include("optimizers.jl")
# include("transformers.jl")
# export LayerNorm, ScaledDotProductAttention, MultiHeadAttention, FeedForward, TransformerBlock, Transformer,
#        cross_entropy_loss, cross_entropy_loss_backward,
#        backward, backward!, zero_gradients!, update!, Adam,
#        generate

include("decoder.jl")
export build_vocab, encode, decode, positional_encoding, parameters, forward, loss, train, generate

end
