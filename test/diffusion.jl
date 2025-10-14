using ZeroToML
using Test
using Random
using Statistics

@testset "DDPM" begin
    @info "This is testing the paper, but the paper's code uses a more complicated model and time embedding"
    Random.seed!(42)
    H,W = 16, 16
    d = H*W
    dataset = shuffle(scale(boxes(H, W, 3)))

    T = 100
    β = noise_schedule(T)
    α = signal_schedule(β)
    ᾱ = remaining_signal(α)
    time_embedding = ᾱ
    model = DDPM()

    # Calculate loss before training on a sample
    x₀_test = rand(dataset)
    ε_test = noise(x₀_test)
    t_test = rand(1:T)
    xt_test = noised_sample(x₀_test, ᾱ, t_test, ε_test)
    untrained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)

    η = 1f-1
    model = train!(model, ᾱ, T, η, dataset, time_embedding)

    # Calculate loss after training on the same sample
    trained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)
    @info "untrained_loss=$(untrained_loss) trained_loss=$(trained_loss)"
    @test trained_loss < untrained_loss
end
