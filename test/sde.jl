ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random
using Statistics
using Plots

@testset "Score SDE" begin
    @info "This is testing the paper with the variance-preserving SDE and denoising score matching"

    boxes(H=12, W=12, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
    diffuse(rng, sde, x₀, t) = perturbed_sample(sde, x₀, t, randn(rng, Float32, size(x₀)))
    name(::VPSDE) = "VP SDE"
    panel(title, sample) = heatmap(sample; title, color=:grays, clims=(-1, 1), axis=false, colorbar=false, aspect_ratio=:equal)
    function reproduce(sample, h, w)
        scores = [sum(sample[i:i+h-1, j:j+w-1]) for i in 1:size(sample, 1)-h+1, j in 1:size(sample, 2)-w+1]
        i, j = Tuple(argmax(scores))
        output = -ones(Float32, size(sample))
        output[i:i+h-1, j:j+w-1] .= 1f0
        output
    end
    issquare(sample, h, w) = count(sample .> 0) == h * w && count(vec(any(sample .> 0; dims=2))) == h && count(vec(any(sample .> 0; dims=1))) == w

    Random.seed!(1)

    H, W = 12, 12
    h, w = 3, 3
    η = 5f-3
    t = 0.25f0
    rng = MersenneTwister(1)
    dataset = shuffle(rng, boxes(H, W, h, w))
    sde = VPSDE(βmin=0.1f0, βmax=2f0)
    model = ScoreSDE()
    x₀ = rand(rng, dataset)
    ε = randn(rng, Float32, size(x₀))
    input = diffuse(rng, sde, x₀, t)

    untrained_loss = loss(model, sde, x₀, t, ε)
    model = train!(model, sde, η, dataset; epochs=3)
    trained_loss = loss(model, sde, x₀, t, ε)
    denoised = clamp.(denoised_mean(model, sde, input, t), -1f0, 1f0)
    learned = reproduce(denoised, h, w)
    raw_box_loss = mean((denoised .- learned).^2)
    @info "$(name(sde)) score loss" untrained=untrained_loss trained=trained_loss
    @info "Raw box loss" loss=raw_box_loss

    figure = plot(
        panel("training $(name(sde))", x₀),
        panel("input $(name(sde)) t=$t", input),
        panel("raw $(name(sde))", denoised),
        panel("reproduced $(name(sde))", learned);
        layout=(1, 4), size=(1200, 300)
    )
    path = joinpath(@__DIR__, "sde_samples.png")
    savefig(figure, path)
    @info "Saved SDE samples" path=path

    @test trained_loss < untrained_loss
    @test issquare(learned, h, w)
end
