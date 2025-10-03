using Test
using Random
using Statistics

include("../src/diffusion.jl")

@testset "Diffusion Toy Driver" begin
    Random.seed!(42)
    C,H,W = 1, 16, 16
    d = C*H*W
    T = 100
    betas = noise_schedule(T)
    α = signal_schedule(betas)
    ᾱ = remaining_signal(α)
    model = parameters(d, 512)

    η = 1f-3
    losses = zeros(Float32, 100)
    for it in 1:100
        x0 = scale(square(H, W))
        t = rand(1:T)
        ε = noise(x0)
        xt = noised_sample(x0, ᾱ, t, ε)
        losses[it] = loss(model, xt, t, ε)
        model = step(model, x0, ᾱ, T; η=η)
        if it%50==0; @info "iter=$(it) loss=$(losses[it])"; end
    end
    @test mean(losses[81:100]) < mean(losses[1:20])

    xgen = reverse_sample(model, betas, α, ᾱ, T, d)
    @info "sample mean=$(mean(xgen)) std=$(std(xgen))"
    xhat = reshape(xgen, H, W)

    @test size(xhat) == (H, W)
    @test eltype(xhat) == Float32
    @test !all(iszero, xhat)
end


# Below is just a scratch pad -- will delete after
# Random.seed!(42)
# C,H,W = 1, 16, 16
# d = C*H*W
# T = 1000
# β = noise_schedule(T)
# α = signal_schedule(β)
# ᾱ = remaining_signal(α)
# model = parameters(d, 512)

# dataset = [scale(square(H, W)) for _ in 1:10_000]

# # Calculate loss before training on a sample
# x0_test = scale(square(H, W))
# ε_test = noise(x0_test)
# t_test = rand(1:T)
# xt_test = noised_sample(x0_test, ᾱ, t_test, ε_test)
# untrained_loss = loss(model, xt_test, t_test, ε_test)

# η = 1f-1
# model = train(model, ᾱ, T, η, dataset)
# # epochs = 100
# # model = train(model, ᾱ, T, η, dataset, epochs)

# # Calculate loss after training on the same sample
# trained_loss = loss(model, xt_test, t_test, ε_test)
# @info "untrained_loss=$(untrained_loss) trained_loss=$(trained_loss)"
# @test trained_loss < untrained_loss

# xgen = reverse_sample(model, β, α, ᾱ, T, d)
# @info "sample mean=$(mean(xgen)) std=$(std(xgen))"
# xhat = reshape(xgen, H, W)

# @test size(xhat) == (H, W)
# @test eltype(xhat) == Float32
# @test !all(iszero, xhat)


# using Plots

# # Make one toy image
# H, W = 16, 16
# img = scale(square(H, W))   # 256-element Vector{Float32}

# # Reshape to 2-D and plot
# heatmap(reshape(img, H, W),
#         color=:grays,
#         aspect_ratio=:equal,
#         title="Random generated square")

# # Generate one sample from the trained model
# xgen = reverse_sample(model, β, α, ᾱ, T, d)

# # Reshape to 16×16 and show as grayscale
# heatmap(reshape(xgen, H, W),
#         color=:grays,
#         aspect_ratio=:equal,
#         title="Sample from trained diffusion model")

# # x = randn(Float32, d)
# # t = rand(1:T)
# # ε̂ = predict(model, x, t)
# # μ = posterior_mean(x, ε̂, β, α, ᾱ, t)
