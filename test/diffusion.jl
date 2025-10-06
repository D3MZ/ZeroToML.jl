using ZeroToML
using Test
using Random
using Statistics

@testset "DDPM" begin
    Random.seed!(42)
    H,W = 16, 16
    d = H*W
    dataset = [scale(square(H, W)) for _ in 1:100]

    T = 100
    β = noise_schedule(T)
    α = signal_schedule(β)
    ᾱ = remaining_signal(α)
    model = ZeroToML.conv_parameters(d)

    # Calculate loss before training on a sample
    x0_test = scale(square(H, W))
    ε_test = noise(x0_test)
    t_test = rand(1:T)
    xt_test = noised_sample(x0_test, ᾱ, t_test, ε_test)
    untrained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)

    η = 1f-1
    model = diffusion_train(model, ᾱ, T, η, dataset)

    # Calculate loss after training on the same sample
    trained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)
    @info "untrained_loss=$(untrained_loss) trained_loss=$(trained_loss)"
    @test trained_loss < untrained_loss
end
