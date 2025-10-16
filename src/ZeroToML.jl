module ZeroToML

using Statistics, StatsBase, Random, Logging, LinearAlgebra, Zygote

include("utils.jl")
include("decoder.jl")
include("diffusion.jl")
include("ppo.jl")
include("cnn.jl")

export build_vocab,
    convolution,
    DDPM,
    PPO,
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
    glorot,
    relu,
    softmax,
    policy,
    value,
    rollout,
    sgd!,
    train,
    train!

end
