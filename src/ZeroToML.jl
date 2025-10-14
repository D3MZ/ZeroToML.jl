module ZeroToML

using Statistics, StatsBase, Random, Logging, LinearAlgebra, Zygote

include("decoder.jl")
include("diffusion.jl")

#Models
export DDPM, Decoder

export
    train!,
    boxes,
    build_vocab,
    decode,
    encode,
    forward,
    generate,
    loss,
    param_count,
    Decoder,
    positional_encoding,
    step,
    train,
    diffusion_step,
    diffusion_train,
    mlp_parameters,
    predict,
    noise,
    noise_schedule,
    signal_schedule,
    remaining_signal,
    noised_sample,
    reverse_sample,
    square,
    scale,
    timestep_embedding

end
