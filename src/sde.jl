using Base: @kwdef
using Random, Statistics, Zygote
using NNlib: conv
using .ZeroToML: glorot, relu, sgd!

@kwdef struct VPSDE
    βmin = 0.1f0
    βmax = 20f0
end

@kwdef struct ScoreSDE
    W₁ = glorot(3, 3, 1, 16)
    b₁ = zeros(Float32, 1, 1, 16, 1)
    W₂ = glorot(3, 3, 16, 32)
    b₂ = zeros(Float32, 1, 1, 32, 1)
    W₃ = glorot(3, 3, 32, 16)
    b₃ = zeros(Float32, 1, 1, 16, 1)
    W₄ = glorot(3, 3, 16, 1)
    b₄ = zeros(Float32, 1, 1, 1, 1)
    Wₜ = reshape(glorot(16, 1), 1, 1, 16, 1)
end

β(sde::VPSDE, t) = sde.βmin + t * (sde.βmax - sde.βmin)
∫β(sde::VPSDE, t) = sde.βmin * t + 0.5f0 * (sde.βmax - sde.βmin) * t^2
@fastmath drift(sde::VPSDE, x, t) = -0.5f0 * β(sde, t) .* x
@fastmath diffusion(sde::VPSDE, t) = sqrt(β(sde, t))
@fastmath marginal_mean(sde::VPSDE, x, t) = exp(-0.5f0 * ∫β(sde, t)) .* x
@fastmath marginal_std(sde::VPSDE, t) = sqrt(1f0 - exp(-∫β(sde, t)))
@fastmath perturbed_sample(sde::VPSDE, x₀, t, ε) = marginal_mean(sde, x₀, t) .+ marginal_std(sde, t) .* ε
@fastmath score_target(sde::VPSDE, t, ε) = .-ε ./ marginal_std(sde, t)

function forward(m::ScoreSDE, x, t)
    H, W = size(x)
    h = reshape(x, H, W, 1, 1)
    padding = (size(m.W₁, 1) - 1) ÷ 2

    h = conv(h, m.W₁; pad=padding) .+ m.b₁ .+ m.Wₜ .* t
    h = relu(h)
    h = conv(h, m.W₂; pad=padding) .+ m.b₂
    h = relu(h)
    h = conv(h, m.W₃; pad=padding) .+ m.b₃
    h = relu(h)
    h = conv(h, m.W₄; pad=padding) .+ m.b₄

    reshape(h, H, W)
end

loss(m::ScoreSDE, sde::VPSDE, x, t, ε) = mean((marginal_std(sde, t) .* forward(m, perturbed_sample(sde, x, t, ε), t) .+ ε).^2)

function step!(m::ScoreSDE, sde::VPSDE, x₀; t=rand(Float32), η=1f-3)
    t = max(t, 1f-3)
    ε = randn(Float32, size(x₀))
    (∇,) = gradient(θ -> loss(θ, sde, x₀, t, ε), m)
    sgd!(m, ∇, η)
    return m
end

function train!(model::ScoreSDE, sde::VPSDE, η, dataset; epochs=1)
    foldl(1:epochs; init=model) do m, epoch
        trained = foldl((θ, x₀) -> step!(θ, sde, x₀; η=η), dataset; init=m)
        x₀ = rand(dataset)
        t = max(rand(Float32), 1f-3)
        ℓ = loss(trained, sde, x₀, t, randn(Float32, size(x₀)))
        @info "epoch=$(epoch) score loss=$(ℓ)"
        trained
    end
end

@fastmath denoised_mean(m::ScoreSDE, sde::VPSDE, x, t) = (x .+ marginal_std(sde, t)^2 .* forward(m, x, t)) ./ exp(-0.5f0 * ∫β(sde, t))
denoise(m::ScoreSDE, sde::VPSDE, x, t) = denoised_mean(m, sde, x, t)

function reverse_predictor(m::ScoreSDE, sde::VPSDE, x, t, Δt; rng=Random.default_rng())
    score = forward(m, x, t)
    dx = drift(sde, x, t) .- diffusion(sde, t)^2 .* score
    x .- dx .* Δt .+ diffusion(sde, t) * sqrt(Δt) .* randn(rng, Float32, size(x))
end

function langevin_corrector(m::ScoreSDE, x, t; snr=0.16f0, steps=1, rng=Random.default_rng())
    foldl(1:steps; init=x) do sample, _
        score = forward(m, sample, t)
        noise = randn(rng, Float32, size(sample))
        score_norm = sqrt(mean(score .^ 2))
        score_norm += eps(score_norm)
        noise_norm = sqrt(mean(noise .^ 2))
        η = 2f0 * (snr * noise_norm / score_norm)^2
        sample .+ η .* score .+ sqrt(2f0 * η) .* noise
    end
end

function reverse_denoise(m::ScoreSDE, sde::VPSDE, x, t; steps=100, corrector_steps=0, snr=0.16f0, rng=Random.default_rng())
    Δt = t / steps
    sample = foldl(steps:-1:1; init=x) do sample, step
        τ = max(Float32(step * Δt), 1f-3)
        predicted = reverse_predictor(m, sde, sample, τ, Δt; rng=rng)
        langevin_corrector(m, predicted, τ; snr=snr, steps=corrector_steps, rng=rng)
    end
    denoise(m, sde, sample, 1f-3)
end

function probability_flow_sample(m::ScoreSDE, sde::VPSDE, x, t; steps=100)
    Δt = t / steps
    sample = foldl(steps:-1:1; init=x) do sample, step
        τ = max(Float32(step * Δt), 1f-3)
        score = forward(m, sample, τ)
        dx = drift(sde, sample, τ) .- 0.5f0 * diffusion(sde, τ)^2 .* score
        sample .- dx .* Δt
    end
    denoise(m, sde, sample, 1f-3)
end

function reverse_sample(m::ScoreSDE, sde::VPSDE, d; steps=100, corrector_steps=0, snr=0.16f0, rng=Random.default_rng())
    H = W = isqrt(d)
    x = randn(rng, Float32, H, W)
    reverse_denoise(m, sde, x, 1f0; steps=steps, corrector_steps=corrector_steps, snr=snr, rng=rng)
end

predictor_corrector_sample(m::ScoreSDE, sde::VPSDE, d; steps=100, corrector_steps=1, snr=0.16f0, rng=Random.default_rng()) = reverse_sample(m, sde, d; steps=steps, corrector_steps=corrector_steps, snr=snr, rng=rng)

function reverse_samples(m::ScoreSDE, sde::VPSDE, d, N; steps=100, corrector_steps=0, snr=0.16f0)
    samples = Vector{Matrix{Float32}}(undef, N)
    Threads.@threads for i in eachindex(samples)
        samples[i] = reverse_sample(m, sde, d; steps=steps, corrector_steps=corrector_steps, snr=snr)
    end
    samples
end
