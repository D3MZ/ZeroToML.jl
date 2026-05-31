# This is testing SDE and denoising score matching: https://arxiv.org/abs/2011.13456
ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random, Dates
using Statistics
using Plots

@testset "Score SDE" begin
    boxes(H=12, W=12, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
    center(x) = x .- mean(x)
    correlate(x, y) = sum(center(x) .* center(y)) / (sqrt(sum(abs2, center(x)) * sum(abs2, center(y))) + eps())
    panel(title, sample) = heatmap(sample; title, color=:grays, clims=(-1, 1), axis=false, colorbar=false, aspect_ratio=:equal)
    Random.seed!(1)

    H, W = 12, 12
    h, w = 3, 3
    t = 1f0
    steps = 100
    η = 0.0642541f0
    duration = Second(10)
    rng = MersenneTwister(1)
    dataset = shuffle(rng, boxes(H, W, h, w))
    sde = VPSDE(βmin=0.1f0, βmax=2f0)
    label = "VP SDE"
    Random.seed!(42)
    model = ScoreSDE()
    x₀ = rand(rng, dataset)
    ε = randn(rng, Float32, size(x₀))
    input = forward_noisy_sample(sde, x₀, t; steps=steps, rng=rng)
    untrained_loss = loss(model, sde, x₀, t, ε)
    model = train!(model, sde, η, dataset, duration; rng=rng)
    trained_loss = loss(model, sde, x₀, t, ε)
    denoised = clamp.(probability_flow_sample(model, sde, input, t; steps=steps), -1f0, 1f0)
    input_loss = mean((input .- x₀).^2)
    denoised_loss = mean((denoised .- x₀).^2)
    input_correlation = correlate(x₀, input)
    denoised_correlation = correlate(x₀, denoised)

    figure = plot(
        panel("training $label", x₀),
        panel("input $label t=$t", input),
        panel("probability flow $label", denoised);
        layout=(3, 1), size=(300, 900)
    )
    output_dir = joinpath(@__DIR__, "outputs")
    mkpath(output_dir)
    path = joinpath(output_dir, "sde_samples.png")
    savefig(figure, path)
    # @info "Saved SDE samples" path=path

    @test trained_loss < untrained_loss
    @test denoised_correlation > input_correlation
    @debug denoised_correlation
end
