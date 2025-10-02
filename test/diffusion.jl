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

    # dummy dataset: e.g., small blobs
    function toy_image()
        img = zeros(Float32, H, W)
        i = rand(4:12); j = rand(4:12)
        img[i-1:i+1, j-1:j+1] .= 1f0
        return reshape(img, d)  # flatten
    end

    η = 1f-3
    losses = zeros(Float32, 100)
    for it in 1:100 # Reduced from 10_000 for testing
        x0 = toy_image()
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
