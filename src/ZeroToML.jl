module ZeroToML

using Statistics, StatsBase, Random, Logging, LinearAlgebra, Zygote

include("decoder.jl")
include("diffusion.jl")
include("cnn.jl")

export build_vocab,
    convolution,
    DDPM,
    decode,
    Decoder,
    encode,
    forward,
    generate,
    loss,
    noise,
    noise_schedule,
    noised_sample,
    param_count,
    positional_encoding,
    remaining_signal,
    reverse_sample,
    reverse_samples,
    signal_schedule,
    train,
    train!

end
