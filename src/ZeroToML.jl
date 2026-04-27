module ZeroToML

using Statistics, StatsBase, Random, Logging, LinearAlgebra, Zygote

include("utils.jl")
include("decoder.jl")
include("diffusion.jl")
include("ppo.jl")
include("cnn.jl")
include("gaussian.jl")
include("kalman.jl")

export build_vocab,
    convolution,
    GaussianProcess,
    KalmanFilter,
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
    kernel_matrix,
    squared_exponential,
    predict,
    propose_next_point,
    relu,
    softmax,
    policy,
    value,
    rollout,
    sgd!,
    train,
    fit!,
    reset!,
    step!,
    simulate

end
