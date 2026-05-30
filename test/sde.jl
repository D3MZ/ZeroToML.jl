ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random
using Statistics
using Plots

@testset "Score SDE" begin
    @info "This is testing SDE and denoising score matching: https://arxiv.org/abs/2011.13456"

    boxes(H=12, W=12, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
    function denoise(model, sde, x, t; steps=100)
        Δt = t / steps
        foldl(steps:-1:1; init=x) do sample, step
            τ = max(Float32(step * Δt), 1f-3)
            β = sde.βmin + τ * (sde.βmax - sde.βmin)
            sample .- (-0.5f0 .* β .* sample .- β .* forward(model, sample, τ)) .* Δt
        end
    end
    panel(title, sample) = heatmap(sample; title, color=:grays, clims=(-1, 1), axis=false, colorbar=false, aspect_ratio=:equal)
    Random.seed!(1)

    H, W = 12, 12
    h, w = 3, 3
    η = 1f-2
    t = 0.75f0
    rng = MersenneTwister(1)
    dataset = shuffle(rng, boxes(H, W, h, w))
    sde = VPSDE(βmin=0.1f0, βmax=2f0)
    label = "VP SDE"
    model = ScoreSDE()
    x₀ = rand(rng, dataset)
    ε = randn(rng, Float32, size(x₀))
    input = perturbed_sample(sde, x₀, t, ε)

    untrained_loss = loss(model, sde, x₀, t, ε)
    model = train!(model, sde, η, dataset; epochs=15)
    trained_loss = loss(model, sde, x₀, t, ε)
    denoised = clamp.(denoise(model, sde, input, t; steps=100), -1f0, 1f0)
    input_loss = mean((input .- x₀).^2)
    denoised_loss = mean((denoised .- x₀).^2)
    @info "$label score loss" untrained=untrained_loss trained=trained_loss
    @info "Denoising loss" input=input_loss denoised=denoised_loss

    figure = plot(
        panel("training $label", x₀),
        panel("input $label t=$t", input),
        panel("reverse $label", denoised);
        layout=(1, 3), size=(900, 300)
    )
    path = joinpath(@__DIR__, "sde_samples.png")
    savefig(figure, path)
    @info "Saved SDE samples" path=path

    @test trained_loss < untrained_loss
    @test denoised_loss < input_loss
end
