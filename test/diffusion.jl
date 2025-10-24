using ZeroToML
using Test
using Random
using Statistics

@testset "DDPM" begin
    @info "This is testing the paper, but the paper's code uses a more complicated model and time embedding"
    
    "Generate all possible h×w boxes (filled with +1f0s) in a H×W grid of -1f0s."
    boxes(H=16, W=16, h=3, w=3) = [(g = ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1.0f0; g) for i in 1:H-h+1 for j in 1:W-w+1]

    Random.seed!(42)
    H,W = 16, 16
    d = H*W
    dataset = shuffle(boxes(H,W))

    T = 100
    β = noise_schedule(T)
    α = signal_schedule(β)
    ᾱ = remaining_signal(α)
    time_embedding = ᾱ
    model = DDPM()

    # Calculate loss before training on a sample
    x₀_test = rand(dataset)
    randn(eltype(x₀_test), size(x₀_test))
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
