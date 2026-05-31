using Base: @kwdef
using Random, Statistics, Zygote, Dates
using NNlib: conv
using .ZeroToML: glorot, relu, sgd!

@kwdef struct OTFlowPath
    σmin = 1f-4
end

@kwdef struct FlowMatching
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

"Conditional OT path standard deviation σₜ = 1 - (1 - σmin)t from Flow Matching, Lipman et al. (2022)."
@fastmath flow_std(path::OTFlowPath, t) = 1f0 - (1f0 - path.σmin) * t

"Conditional OT sample xₜ = σₜx₀ + tx₁, where x₀ is prior noise and x₁ is data."
@fastmath flow_sample(path::OTFlowPath, x₀, x₁, t) = flow_std(path, t) .* x₀ .+ t .* x₁

"Conditional OT vector-field target uₜ = ∂ₜxₜ = x₁ - (1 - σmin)x₀."
@fastmath flow_target(path::OTFlowPath, x₀, x₁) = x₁ .- (1f0 - path.σmin) .* x₀

"Velocity network vθ(xₜ,t) for Conditional Flow Matching."
function forward(m::FlowMatching, x, t)
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

"Conditional Flow Matching loss E‖vθ(xₜ,t) - uₜ(xₜ|x₁)‖²."
function loss(m::FlowMatching, path::OTFlowPath, x₀, x₁, t)
    xt = flow_sample(path, x₀, x₁, t)
    ut = flow_target(path, x₀, x₁)
    mean((forward(m, xt, t) .- ut).^2)
end

"One stochastic-gradient step for OT Conditional Flow Matching."
function step!(m::FlowMatching, path::OTFlowPath, x₁; t=rand(Float32), η=1f-3, rng=Random.default_rng())
    t = clamp(t, 1f-3, 1f0)
    x₀ = randn(rng, Float32, size(x₁))
    (∇,) = gradient(θ -> loss(θ, path, x₀, x₁, t), m)
    sgd!(m, ∇, η)
    return m
end

"Trains Flow Matching for N epochs over a dataset."
function train!(model::FlowMatching, path::OTFlowPath, η, dataset, epochs::Int=1; rng=Random.default_rng())
    foldl(1:epochs; init=model) do m, _
        foldl((θ, x₁) -> step!(θ, path, x₁; η=η, rng=rng), dataset; init=m)
    end
end

"Trains Flow Matching for a time budget, completing full dataset passes."
function train!(model::FlowMatching, path::OTFlowPath, η, dataset, duration::Dates.Period; rng=Random.default_rng())
    target_s = seconds(duration)
    t₀ = time()
    while true
        time() - t₀ >= target_s && break
        model = foldl((θ, x₁) -> step!(θ, path, x₁; η=η, rng=rng), dataset; init=model)
    end
    model
end

"Euler ODE solve dx/dt = vθ(x,t) from t₀ to t₁."
function flow(m::FlowMatching, x, t₀=0f0, t₁=1f0; steps=100)
    Δt = (t₁ - t₀) / steps
    foldl(1:steps; init=x) do sample, step
        t = Float32(t₀ + (step - 1) * Δt)
        sample .+ Float32(Δt) .* forward(m, sample, t)
    end
end

"Generates one sample by integrating the learned CNF from Gaussian prior noise to data time."
function reverse_sample(m::FlowMatching, d; steps=100, rng=Random.default_rng())
    H = W = isqrt(d)
    x₀ = randn(rng, Float32, H, W)
    flow(m, x₀, 0f0, 1f0; steps=steps)
end

"Draws multiple Flow Matching samples."
function reverse_samples(m::FlowMatching, d, N; steps=100)
    samples = Vector{Matrix{Float32}}(undef, N)
    Threads.@threads for i in eachindex(samples)
        samples[i] = reverse_sample(m, d; steps=steps)
    end
    samples
end
