using Random, Statistics, Zygote, Tullio, LoopVectorization
using NNlib: conv

@kwdef struct DDPM
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

"model's forward process: ε̂ = ϵθ(xt,t)"
function forward(m::DDPM, x, t, time_embedding)
    H, W = size(x)
    h = reshape(x, H, W, 1, 1)
    padding = (size(m.W₁, 1) - 1) ÷ 2

    # Layer 1 with time embedding injection
    h = conv(h, m.W₁; pad=padding) .+ m.b₁ .+ m.Wₜ .* time_embedding[t]
    h = relu(h)

    # Layer 2
    h = conv(h, m.W₂; pad=padding) .+ m.b₂
    h = relu(h)

    # Layer 3
    h = conv(h, m.W₃; pad=padding) .+ m.b₃
    h = relu(h)
    
    # Layer 4 (Final layer)
    h = conv(h, m.W₄; pad=padding) .+ m.b₄
    
    return reshape(h, H, W)
end

abstract type Noise end

struct Gaussian <: Noise end
struct StudentT <: Noise
    ν
end
struct Cauchy <: Noise end

"Generates a box of the same type and size with random values"
noise(x) = noise(x, Gaussian())
noise(x, process::Noise) = noise(Random.default_rng(), x, process)
noise(rng, x, ::Gaussian) = randn(rng, eltype(x), size(x))
noise(rng, x, process::StudentT) = clamp.(randn(rng, eltype(x), size(x)) ./ sqrt.(dropdims(sum(abs2, randn(rng, eltype(x), size(x)..., Int(process.ν)); dims=ndims(x)+1); dims=ndims(x)+1) ./ process.ν), -eltype(x)(10), eltype(x)(10))
noise(rng, x, ::Cauchy) = clamp.(tan.(eltype(x)(π) .* (rand(rng, eltype(x), size(x)) .- eltype(x)(0.5))), -eltype(x)(10), eltype(x)(10))
"The entire noise variance schedule via β_t = β_min + (β_max - β_min) * (t-1)/(T-1)"
noise_schedule(T; βmin=1f-4, βmax=0.02f0) = range(βmin, βmax; length=T)
"Entire signal variance schedule: α_t = 1 - β_t"
signal_schedule(β::AbstractRange) = 1 .- β
"the total remaining signal variance is the cumprod of the signal_schedule"
remaining_signal(α::AbstractRange) = cumprod(α)
"Log Signal to Noise Ratio"
snr(ᾱ) = log.(ᾱ ./ (1 .- ᾱ))
"Conditional marginal mean E[xₜ | x₀] for the forward diffusion process q(xₜ | x₀)"
@fastmath marginal_mean(x, ᾱ, t) = sqrt(ᾱ[t]) .* x
"Conditional marginal noise for the forward diffusion marginal q(xₜ | x₀). This is the random Gaussian noise part added to the deterministic mean √ᾱₜ · x₀."
@fastmath marginal_noise(ᾱ, t, ε) = sqrt(1-ᾱ[t]).*ε
"Forward noise sample q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε, with ε ~ N(0, I)"
@fastmath noised_sample(x₀, ᾱ, t, ε) = marginal_mean(x₀, ᾱ, t) .+ (sqrt(1-ᾱ[t]) .* ε)
"Mean Squared Error (MSE) loss used for DDPM training: Lₛᵢₘₚₗₑ(θ) := 𝐄ₜ,ₓ₀,ϵ ‖ϵ − ϵθ(√ᾱₜ·x₀ + √(1−ᾱₜ)·ϵ, t)‖²"
loss(θ::DDPM, x, t, ε, time_embedding) = mean((ε .- forward(θ, x, t, time_embedding)).^2)
"Performs one training step: adds noise xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε and updates model by gradient of the loss (ε̂, ε)"
function step!(m::DDPM, x₀, ᾱ, T, time_embedding; t=rand(1:T), η=1e-3f0, process=Gaussian())
    ε  = noise(x₀, process)
    xt = noised_sample(x₀, ᾱ, t, ε)
    (∇,) = gradient(θ -> loss(θ, xt, t, ε, time_embedding), m)
    sgd!(m, ∇, η)
    return m
end

"Computes μₜ = (xₜ − (βₜ / √(1−ᾱₜ))·ε̂) / √αₜ for the reverse diffusion mean"
@fastmath posterior_mean(x, ε̂, β, α, ᾱ, t) = (x .- (β[t]/sqrt(1-ᾱ[t])) .* ε̂) ./ sqrt(α[t])

"Draws a sample xₜ₋₁ ~ μ + √βₜ · N(0, I) from the reverse diffusion step"
@fastmath latent(μ, β, t, x) = μ .+ sqrt(β[t]) .* randn(eltype(x), size(x))

"Generates ~x0 by iteratively sampling xₜ₋₁ = μₜ(xₜ, ε̂) + √βₜ·z for t = T,…,0, starting from x_T ~ N(0,I). "
function reverse_sample(m::DDPM, β, α, ᾱ, T, d, time_embedding)
    H = W = isqrt(d)
    x = randn(Float32, H, W)
    μ = similar(x)
    for t in T:-1:2
        ε̂ = forward(m, x, t, time_embedding)
        μ = posterior_mean(x, ε̂, β, α, ᾱ, t)
        x = latent(μ, β, t, x)
    end
    
    t = 1
    ε̂ = forward(m, x, t, time_embedding)
    posterior_mean(x, ε̂, β, α, ᾱ, t)
end

function reverse_samples(m::DDPM, β, α, ᾱ, T, d, time_embedding, N)                                                                                                                             
    samples = Vector{Matrix{Float32}}(undef, N)                                                                     
     Threads.@threads for i in eachindex(samples)                                                                   
        samples[i] = reverse_sample(m, β, α, ᾱ, T, d, time_embedding)                                              
     end                                                                                                            
    return samples                                                                                                  
end 

"Trains the diffusion model over the dataset by repeatedly applying one training step"
function train!(model::DDPM, ᾱ, T, η, dataset, time_embedding; process=Gaussian())
    trained = foldl(1:3; init=model) do m, epoch
        foldl((θ, x₀) -> step!(θ, x₀, ᾱ, T, time_embedding; η=1.25f0 * η, process=process), dataset; init=m)
    end
    foldl((θ, x₀) -> step!(θ, x₀, ᾱ, T, time_embedding; η=0.25f0 * η, process=process), first(dataset, 32); init=trained)
end
"Trains for E epochs by folding `train!(model, ᾱ, T, η, dataset, time_embedding)` over epochs: mₑ = foldl((m,_)->train!(m, ᾱ, T, η, dataset, time_embedding), 1:E; init=model)"
function train!(model, ᾱ, T, η, dataset, time_embedding, epochs; process=Gaussian())
    foldl(1:epochs; init=model) do m, epoch
        trained = train!(m, ᾱ, T, η, dataset, time_embedding; process=process)
        x₀ = rand(dataset)
        ε = noise(x₀, process)
        t = rand(1:T)
        xt = noised_sample(x₀, ᾱ, t, ε)
        ℓ = loss(trained, xt, t, ε, time_embedding)
        @info "epoch=$(epoch) loss=$(ℓ)"
        trained
    end
end
