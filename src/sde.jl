using Base: @kwdef
using Random, Statistics, Zygote, Dates
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

"VP-SDE linear noise schedule β(t), source: https://arxiv.org/abs/2011.13456"
β(sde::VPSDE, t) = sde.βmin + t * (sde.βmax - sde.βmin)
"Integrated VP-SDE noise schedule ∫₀ᵗβ(s)ds, source: https://arxiv.org/abs/2011.13456"
∫β(sde::VPSDE, t) = sde.βmin * t + 0.5f0 * (sde.βmax - sde.βmin) * t^2
"VP-SDE drift f(x,t)=-β(t)x/2, source: https://arxiv.org/abs/2011.13456"
@fastmath drift(sde::VPSDE, x, t) = -0.5f0 * β(sde, t) .* x
"VP-SDE diffusion coefficient g(t)=√β(t), source: https://arxiv.org/abs/2011.13456"
@fastmath diffusion(sde::VPSDE, t) = sqrt(β(sde, t))
"VP-SDE perturbation mean α(t)x₀, source: https://arxiv.org/abs/2011.13456"
@fastmath marginal_mean(sde::VPSDE, x, t) = exp(-0.5f0 * ∫β(sde, t)) .* x
"VP-SDE perturbation standard deviation σ(t), source: https://arxiv.org/abs/2011.13456"
@fastmath marginal_std(sde::VPSDE, t) = sqrt(1f0 - exp(-∫β(sde, t)))
"Samples xₜ from the VP-SDE perturbation kernel p₀ₜ(xₜ|x₀), source: https://arxiv.org/abs/2011.13456"
@fastmath perturbed_sample(sde::VPSDE, x₀, t, ε) = marginal_mean(sde, x₀, t) .+ marginal_std(sde, t) .* ε
"Conditional perturbation score ∇ₓₜ log p₀ₜ(xₜ|x₀), source: https://arxiv.org/abs/2011.13456"
@fastmath score_target(sde::VPSDE, t, ε) = .-ε ./ marginal_std(sde, t)

"Score network sθ(xₜ,t) for SDE denoising score matching, source: https://arxiv.org/abs/2011.13456"
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

"Denoising score matching loss with λ(t)=σ(t)², source: https://arxiv.org/abs/2011.13456"
loss(m::ScoreSDE, sde::VPSDE, x, t, ε) = mean((marginal_std(sde, t) .* forward(m, perturbed_sample(sde, x, t, ε), t) .+ ε).^2)

"One stochastic gradient step for SDE denoising score matching, source: https://arxiv.org/abs/2011.13456"
function step!(m::ScoreSDE, sde::VPSDE, x₀; t=rand(Float32), η=1f-3)
    t = max(t, 1f-3)
    ε = randn(Float32, size(x₀))
    (∇,) = gradient(θ -> loss(θ, sde, x₀, t, ε), m)
    sgd!(m, ∇, η)
    return m
end

"Trains a score model for N epochs across random continuous SDE times, source: https://arxiv.org/abs/2011.13456"
function train!(model::ScoreSDE, sde::VPSDE, η, dataset, epochs::Int=1)
    foldl(1:epochs; init=model) do m, _
        trained = foldl((θ, x₀) -> step!(θ, sde, x₀; η=η), dataset; init=m)
        x₀ = rand(dataset)
        t = max(rand(Float32), 1f-3)
        ℓ = loss(trained, sde, x₀, t, randn(Float32, size(x₀)))
        trained
    end
end

"Trains a score model for a time budget, completing full epochs, source: https://arxiv.org/abs/2011.13456"
function train!(model::ScoreSDE, sde::VPSDE, η, dataset, duration::Dates.Period)
    target_s = duration isa Dates.Second ? Dates.value(duration) :
                duration isa Dates.Minute ? Dates.value(duration) * 60 :
                duration isa Dates.Hour ? Dates.value(duration) * 3600 :
                Dates.value(duration)
    t₀ = time()
    while true
        time() - t₀ >= target_s && break
        model = foldl((θ, x₀) -> step!(θ, sde, x₀; η=η), dataset; init=model)
    end
    model
end

"Final denoised estimate x̂₀ from the score, source: https://arxiv.org/abs/2011.13456"
@fastmath denoised_mean(m::ScoreSDE, sde::VPSDE, x, t) = (x .+ marginal_std(sde, t)^2 .* forward(m, x, t)) ./ exp(-0.5f0 * ∫β(sde, t))
"Alias for the final SDE denoised estimate, source: https://arxiv.org/abs/2011.13456"
denoise(m::ScoreSDE, sde::VPSDE, x, t) = denoised_mean(m, sde, x, t)

"Euler-Maruyama predictor for the reverse-time SDE, source: https://arxiv.org/abs/2011.13456"
function reverse_predictor(m::ScoreSDE, sde::VPSDE, x, t, Δt; rng=Random.default_rng())
    score = forward(m, x, t)
    dx = drift(sde, x, t) .- diffusion(sde, t)^2 .* score
    x .- dx .* Δt .+ diffusion(sde, t) * sqrt(Δt) .* randn(rng, Float32, size(x))
end

"SNR-controlled Langevin corrector for predictor-corrector sampling, source: https://arxiv.org/abs/2011.13456"
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

"Runs reverse-time SDE denoising with optional Langevin corrector and final denoising, source: https://arxiv.org/abs/2011.13456"
function reverse_denoise(m::ScoreSDE, sde::VPSDE, x, t; steps=100, corrector_steps=0, snr=0.16f0, rng=Random.default_rng())
    Δt = t / steps
    sample = foldl(steps:-1:1; init=x) do sample, step
        τ = max(Float32(step * Δt), 1f-3)
        predicted = reverse_predictor(m, sde, sample, τ, Δt; rng=rng)
        langevin_corrector(m, predicted, τ; snr=snr, steps=corrector_steps, rng=rng)
    end
    denoise(m, sde, sample, 1f-3)
end

"Deterministic probability-flow ODE sampler with the 1/2 score coefficient, source: https://arxiv.org/abs/2011.13456"
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

"Samples from the VP-SDE prior and solves the reverse-time SDE, source: https://arxiv.org/abs/2011.13456"
function reverse_sample(m::ScoreSDE, sde::VPSDE, d; steps=100, corrector_steps=0, snr=0.16f0, rng=Random.default_rng())
    H = W = isqrt(d)
    x = randn(rng, Float32, H, W)
    reverse_denoise(m, sde, x, 1f0; steps=steps, corrector_steps=corrector_steps, snr=snr, rng=rng)
end

"Predictor-corrector sampler using reverse SDE prediction and Langevin correction, source: https://arxiv.org/abs/2011.13456"
predictor_corrector_sample(m::ScoreSDE, sde::VPSDE, d; steps=100, corrector_steps=1, snr=0.16f0, rng=Random.default_rng()) = reverse_sample(m, sde, d; steps=steps, corrector_steps=corrector_steps, snr=snr, rng=rng)

"Draws multiple reverse-time SDE samples, source: https://arxiv.org/abs/2011.13456"
function reverse_samples(m::ScoreSDE, sde::VPSDE, d, N; steps=100, corrector_steps=0, snr=0.16f0)
    samples = Vector{Matrix{Float32}}(undef, N)
    Threads.@threads for i in eachindex(samples)
        samples[i] = reverse_sample(m, sde, d; steps=steps, corrector_steps=corrector_steps, snr=snr)
    end
    samples
end
