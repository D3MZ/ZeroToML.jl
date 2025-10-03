module ZeroToML

using Statistics, StatsBase, Random, Logging, LinearAlgebra, Zygote

include("decoder.jl")
include("diffusion.jl")
export
    build_vocab,
    decode,
    encode,
    forward,
    generate,
    loss,
    param_count,
    parameters,
    positional_encoding,
    step,
    train,
    # diffusion
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
