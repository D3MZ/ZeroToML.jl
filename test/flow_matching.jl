# This is testing Conditional Flow Matching with OT paths: https://arxiv.org/abs/2210.02747
ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random, Dates
using Statistics
using Plots

@testset "Flow Matching" begin
    boxes(H=12, W=12, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
    center(x) = x .- mean(x)
    correlate(x, y) = sum(center(x) .* center(y)) / (sqrt(sum(abs2, center(x)) * sum(abs2, center(y))) + eps())
    nearest_correlation(sample, dataset) = maximum(correlate(sample, x) for x in dataset)
    panel(title, sample) = heatmap(sample; title, color=:grays, clims=(-1, 1), axis=false, colorbar=false, aspect_ratio=:equal)
    Random.seed!(1)

    H, W = 12, 12
    h, w = 3, 3
    η = 0.02f0
    loss_t = 0.5f0
    t₀ = 0.3f0
    t₁ = 1f0
    steps = 100
    rng = MersenneTwister(1)
    dataset = shuffle(rng, boxes(H, W, h, w))
    path = OTFlowPath(σmin=1f-4)
    label = "OT-CFM"
    Random.seed!(42)
    model = FlowMatching()
    x₁ = rand(rng, dataset)
    x₀ = randn(rng, Float32, size(x₁))
    input = flow_sample(path, x₀, x₁, t₀)

    untrained_loss = loss(model, path, x₀, x₁, loss_t)
    model = train!(model, path, η, dataset, Second(10); rng=rng)
    trained_loss = loss(model, path, x₀, x₁, loss_t)
    generated = clamp.(flow(model, input, t₀, t₁; steps=steps), -1f0, 1f0)
    input_correlation = nearest_correlation(input, dataset)
    generated_correlation = nearest_correlation(generated, dataset)

    figure = plot(
        panel("training $label", x₁),
        panel("input $label t=$t₀", input),
        panel("flow $label t=$t₁", generated);
        layout=(3, 1), size=(300, 900)
    )
    savefig(figure, joinpath(@__DIR__, "flow_matching_samples.png"))

    @test trained_loss < untrained_loss
    @test generated_correlation > input_correlation
    @debug "Flow Matching correlations" input=input_correlation generated=generated_correlation
end
