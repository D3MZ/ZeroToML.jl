ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random
using Statistics
using Plots

@testset "DDPM" begin
    @debug "This is testing the paper, but the paper's code uses a more complicated model and time embedding"

    "Generate all possible h×w boxes (filled with +1f0s) in a H×W grid of -1f0s."
    boxes(H=12, W=12, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1.0f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
    center(x) = x .- mean(x)
    correlate(x, y) = sum(center(x) .* center(y)) / (sqrt(sum(abs2, center(x)) * sum(abs2, center(y))) + eps())
    diffuse(rng, x₀, ᾱ, t, process) = noised_sample(x₀, ᾱ, t, noise(rng, x₀, process))
    name(::Gaussian) = "Gaussian"
    name(process::StudentT) = "StudentT ν=$(process.ν)"
    name(::Cauchy) = "Cauchy"
    rate(::Gaussian, η) = η
    rate(::StudentT, η) = η
    rate(::Cauchy, η) = η / 10
    panel(title, sample) = heatmap(sample; title, color=:grays, clims=(-1, 1), axis=false, colorbar=false, aspect_ratio=:equal)
    panels(training, inputs, denoised, processes, denoise_steps) = [panel(title, sample) for (title, sample) in vcat([("training $(name(process))", sample) for (process, sample) in zip(processes, training)], [("input $(name(process)) t=$denoise_steps", sample) for (process, sample) in zip(processes, inputs)], [("denoised $(name(process))", sample) for (process, sample) in zip(processes, denoised)])]

    Random.seed!(1)

    H, W = 12, 12
    h, w = 3, 3
    T = 100
    η = 1f-1
    denoise_steps = 100
    processes = [Gaussian(), StudentT(3), Cauchy()]
    n_samples = length(processes)
    rng = MersenneTwister(1)
    dataset = shuffle(rng, boxes(H, W, h, w))

    β = noise_schedule(T)
    α = signal_schedule(β)
    ᾱ = remaining_signal(α)
    time_embedding = ᾱ
    training_sample = rand(rng, dataset)
    training = fill(training_sample, n_samples)
    inputs = [diffuse(rng, training_sample, ᾱ, denoise_steps, process) for process in processes]

    losses = map(processes) do process
        model = DDPM()
        ε_test = noise(rng, training_sample, process)
        xt_test = noised_sample(training_sample, ᾱ, denoise_steps, ε_test)
        untrained_loss = loss(model, xt_test, denoise_steps, ε_test, ᾱ)
        model = train!(model, ᾱ, T, rate(process, η), dataset, time_embedding; process=process)
        trained_loss = loss(model, xt_test, denoise_steps, ε_test, ᾱ)
        @debug "$(name(process)) loss" untrained=untrained_loss trained=trained_loss
        (model, untrained_loss, trained_loss)
    end

    denoised = [clamp.(denoise(model, sample, β, α, ᾱ, denoise_steps, time_embedding), -1f0, 1f0) for ((model, _, _), sample) in zip(losses, inputs)]
    input_correlations = [correlate(training_sample, sample) for sample in inputs]
    denoised_correlations = [correlate(training_sample, sample) for sample in denoised]
    @debug "DDPM correlations" input=input_correlations denoised=denoised_correlations

    figure = plot(panels(training, inputs, denoised, processes, denoise_steps)...; layout=(3, n_samples), size=(900, 900))
    output_dir = joinpath(@__DIR__, "outputs")
    mkpath(output_dir)
    path = joinpath(output_dir, "ddpm_samples.png")
    savefig(figure, path)
    @debug "Saved DDPM samples" path=path

    @test all(loss -> loss[3] < loss[2], losses)
    @test mean(denoised_correlations) > mean(input_correlations)
end
