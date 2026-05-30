ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random
using Statistics
using Plots

@testset "DDPM" begin
    @info "This is testing the paper, but the paper's code uses a more complicated model and time embedding"
    
    "Generate all possible h×w boxes (filled with +1f0s) in a H×W grid of -1f0s."
    boxes(H=16, W=16, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1.0f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
    diffuse(rng, x₀, ᾱ, t, process) = noised_sample(x₀, ᾱ, t, noise(rng, x₀, process))
    name(::Gaussian) = "Gaussian"
    name(process::StudentT) = "StudentT ν=$(process.ν)"
    name(::Cauchy) = "Cauchy"
    rate(::Gaussian, η) = η
    rate(::StudentT, η) = η
    rate(::Cauchy, η) = η / 10
    panels(training, inputs, denoised, processes, denoise_steps) = [heatmap(sample; title, color=:grays, clims=(-1, 1), axis=false, colorbar=false, aspect_ratio=:equal) for (title, sample) in vcat([("training $(name(process))", sample) for (process, sample) in zip(processes, training)], [("input $(name(process)) t=$denoise_steps", sample) for (process, sample) in zip(processes, inputs)], [("denoised $(name(process))", sample) for (process, sample) in zip(processes, denoised)])]
    
    Random.seed!(1)

    H, W = 16, 16
    h, w = 3, 3
    T = 100
    η = 1f-1
    denoise_steps = 100
    processes = [Gaussian(), StudentT(3), Cauchy()]
    n_samples = length(processes)
    image_size = (900, 900)

    rng = MersenneTwister(1)
    d = H * W
    dataset = shuffle(rng, boxes(H, W, h, w))

    β = noise_schedule(T)
    α = signal_schedule(β)
    ᾱ = remaining_signal(α)
    time_embedding = ᾱ
    x₀_test = rand(rng, dataset)
    t_test = rand(rng, 1:T)
    training_sample = rand(rng, dataset)
    training = fill(training_sample, n_samples)
    inputs = map(processes) do process
        diffuse(rng, training_sample, ᾱ, denoise_steps, process)
    end

    losses = map(processes) do process
        model = DDPM()
        ε_test = noise(rng, x₀_test, process)
        xt_test = noised_sample(x₀_test, ᾱ, t_test, ε_test)
        untrained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)
        model = train!(model, ᾱ, T, rate(process, η), dataset, time_embedding; process=process)
        trained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)
        @info "$(name(process)) loss" untrained=untrained_loss trained=trained_loss
        (model, untrained_loss, trained_loss)
    end

    denoised = [clamp.(denoise(model, sample, β, α, ᾱ, denoise_steps, time_embedding), -1f0, 1f0) for ((model, _, _), sample) in zip(losses, inputs)]
    input_losses = [mean((sample .- training_sample).^2) for sample in inputs]
    denoised_losses = [mean((sample .- training_sample).^2) for sample in denoised]
    @info "Denoising loss" input=input_losses denoised=denoised_losses
    figure = plot(panels(training, inputs, denoised, processes, denoise_steps)...; layout=(3, n_samples), size=image_size)
    path = joinpath(@__DIR__, "ddpm_samples.png")
    savefig(figure, path)
    @info "Saved DDPM samples" path=path

    @test all(loss -> loss[3] < loss[2], losses)
    @test mean(denoised_losses) < mean(input_losses)
end
