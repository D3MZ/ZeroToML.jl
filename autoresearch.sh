#!/usr/bin/env bash
set -euo pipefail

julia --project --startup-file=no <<'JL'
using ZeroToML, Random, Statistics

boxes(H=16, W=16, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1.0f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
input(rng, x₀, ᾱ, t, process) = noised_sample(x₀, ᾱ, t, noise(rng, x₀, process))
denoise(model, x, β, α, ᾱ, t, time_embedding) = foldl(t:-1:1; init=x) do sample, step
    ε̂ = forward(model, sample, step, time_embedding)
    (sample .- 0.9f0 .* (β[step] / sqrt(1 - ᾱ[step])) .* ε̂) ./ sqrt(α[step])
end
function reproduce(sample, h, w)
    scores = [sum(sample[i:i+h-1, j:j+w-1]) for i in 1:size(sample, 1)-h+1, j in 1:size(sample, 2)-w+1]
    i, j = Tuple(argmax(scores))
    output = -ones(Float32, size(sample))
    output[i:i+h-1, j:j+w-1] .= 1f0
    output
end
rate(::Gaussian, η) = η
rate(::StudentT, η) = η
rate(::Cauchy, η) = η / 10

Random.seed!(1)
H, W = 16, 16
h, w = 3, 3
T = 100
η = 1f-1
denoise_steps = 100
processes = (Gaussian(), StudentT(3), Cauchy())
dataset = shuffle(MersenneTwister(1), boxes(H, W, h, w))
β = noise_schedule(T)
α = signal_schedule(β)
ᾱ = remaining_signal(α)
time_embedding = ᾱ
evaluation = dataset[[1, 29, 57, 85, 113, 141, 169, 196]]

for process in processes
    train!(DDPM(), ᾱ, T, rate(process, η), first(dataset, 2), time_embedding; process=process);
end

training_times = Float64[]
noise_losses = Float32[]
raw_box_losses = Float32[]

for (index, process) in enumerate(processes)
    Random.seed!(100 + index)
    model = DDPM()
    seconds = @elapsed train!(model, ᾱ, T, rate(process, η), dataset, time_embedding; process=process)
    push!(training_times, seconds)

    rng = MersenneTwister(200 + index)
    x₀ = dataset[37 * index]
    ε = noise(rng, x₀, process)
    t = 13 * index + 7
    xt = noised_sample(x₀, ᾱ, t, ε)
    push!(noise_losses, loss(model, xt, t, ε, time_embedding))

    for sample in evaluation
        noisy = input(rng, sample, ᾱ, denoise_steps, process)
        raw = clamp.(denoise(model, noisy, β, α, ᾱ, denoise_steps, time_embedding), -1f0, 1f0)
        push!(raw_box_losses, mean((raw .- reproduce(raw, h, w)).^2))
    end
end

println("METRIC mean_raw_box_loss=", mean(raw_box_losses))
println("METRIC max_training_s=", maximum(training_times))
println("METRIC mean_training_s=", mean(training_times))
println("METRIC noise_loss=", mean(noise_losses))
JL
