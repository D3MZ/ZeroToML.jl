using Random, Statistics, Test, Zygote

"Relu Activation function"
relu(x::AbstractArray) = max.(x, zero(eltype(x)))
relu(x::Number)        = max(x, zero(x))

"Glorot/Xavier uniform initialization: Wᵢⱼ ~ U[-√(6/(m+n)), √(6/(m+n))] via https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
glorot(m, n) = rand(Float32, m, n) .* (2f0*sqrt(6f0/(m+n))) .- sqrt(6f0/(m+n))

"Sinusoidal timestep embedding for a single timestep `t` and dimension `d`"
function timestep_embedding(t, d)
    half_d = d ÷ 2
    emb = log(10000.0f0) / (half_d - 1)
    emb = exp.(range(0, stop=half_d-1) .* -emb)
    emb = t .* emb
    emb = vcat(sin.(emb), cos.(emb))
    if d % 2 == 1
        emb = vcat(emb, 0.0f0)
    end
    return emb
end

"Initialize MLP parameters for dimension d -> d (noise prediction)"
function parameters(d, h=1024)
    W1 = glorot(h, d); b1 = zeros(Float32, h)
    W2 = glorot(d, h); b2 = zeros(Float32, d)
    W_temb = glorot(h, d)
    return (W1=W1, b1=b1, W2=W2, b2=b2, W_temb=W_temb)
end

"ε̂ = ϵθ(xt,t)"
function predict(m, x, t)
    temb_d = size(m.W_temb, 2)
    temb = timestep_embedding(t, temb_d)
    h1 = relu(m.W1*x .+ m.W_temb*temb .+ m.b1)
    m.W2*h1 .+ m.b2
end

noise(x) = randn(eltype(x), size(x))
"The entire noise variance schedule via β_t = β_min + (β_max - β_min) * (t-1)/(T-1)"
noise_schedule(T; βmin=1f-4, βmax=0.02f0) = range(βmin, βmax; length=T)
"Entire signal variance schedule: α_t = 1 - β_t"
signal_schedule(β::AbstractRange) = 1 .- β
"the total remaining signal variance is the cumprod of the signal_schedule"
remaining_signal(α::AbstractRange) = cumprod(α)
"Conditional marginal mean E[xₜ | x₀] for the forward diffusion process q(xₜ | x₀)"
marginal_mean(x, ᾱ, t) = sqrt(ᾱ[t]) .* x
"Conditional marginal noise for the forward diffusion marginal q(xₜ | x₀). This is the random Gaussian noise part added to the deterministic mean √ᾱₜ · x₀."
marginal_noise(ᾱ, t, ε) = sqrt(1-ᾱ[t]).*ε
"Forward noise sample q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε, with ε ~ N(0, I)"
noised_sample(x0, ᾱ, t, ε) = marginal_mean(x0, ᾱ, t) .+ (sqrt(1-ᾱ[t]) .* ε)
"Mean Squared Error (MSE) loss used for DDPM training: Lₛᵢₘₚₗₑ(θ) := 𝐄ₜ,ₓ₀,ϵ ‖ϵ − ϵθ(√ᾱₜ·x₀ + √(1−ᾱₜ)·ϵ, t)‖²"
loss(θ, x, t, y) = mean((y .- predict(θ, x, t)).^2)
"Stochastic Gradient Descent (SGD). m, ∇, η are parameters, gradients, and learning rate respectively"
sgd(m, ∇, η) = map((p, g) -> p .- η .* g, m, ∇)

"Performs one training step: adds noise xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε and updates model by gradient of the loss (ε̂, ε)"
function step(m, x0, ᾱ, T; t=rand(1:T), η=1e-3f0)
    ε  = noise(x0)
    xt = noised_sample(x0, ᾱ, t, ε)
    (∇,) = gradient(θ -> loss(θ, xt, t, ε), m)
    sgd(m, ∇, η)
end

"Computes μₜ = (xₜ − (βₜ / √(1−ᾱₜ))·ε̂) / √αₜ for the reverse diffusion mean"
posterior_mean(x, ε̂, β, α, ᾱ, t) = (x .- (β[t]/sqrt(1-ᾱ[t])) .* ε̂) ./ sqrt(α[t])

"Draws a sample xₜ₋₁ ~ μ + √βₜ · N(0, I) from the reverse diffusion step"
latent(μ, β, t, x) = μ .+ sqrt(β[t]) .* randn(eltype(x), size(x))

"Generates ~x0 by iteratively sampling xₜ₋₁ = μₜ(xₜ, ε̂) + √βₜ·z for t = T,…,1, starting from x_T ~ N(0,I). "
function reverse_sample(m, β, α, ᾱ, T, d)
    x = randn(Float32, d)
    μ = similar(x)
    for t in T:-1:2
        ε̂ = predict(m, x, t)
        μ = posterior_mean(x, ε̂, β, α, ᾱ, t)
        x = latent(μ, β, t, x)
    end
    
    t = 1
    ε̂ = predict(m, x, t)
    posterior_mean(x, ε̂, β, α, ᾱ, t)
end

"Trains the diffusion model over the dataset by repeatedly applying one training step"
train(model, ᾱ, T, η, dataset) = foldl((m, x0) -> step(m, x0, ᾱ, T; η=η), dataset; init=model)
"Trains for E epochs by folding `train(model, ᾱ, T, η, dataset)` over epochs: mₑ = foldl((m,_)->train(m, ᾱ, T, η, dataset), 1:E; init=model)"
train(model, ᾱ, T, η, dataset, epochs) = foldl((m, _) -> train(m, ᾱ, T, η, dataset), 1:epochs; init=model)

"Generates a square of 255s against a 0s background"
function square(h, w)
    d = h * w
    img = zeros(Int, h, w)
    i = rand(4:12); j = rand(4:12)
    img[i-1:i+1, j-1:j+1] .= 255
    return reshape(img, d)
end

"Scales an image from [0, 255] to [-1, 1]"
scale(img) = (2.0f0 .* Float32.(img) ./ 255.0f0) .- 1.0f0

Random.seed!(42)
C,H,W = 1, 16, 16
d = C*H*W
T = 1000
β = noise_schedule(T)
α = signal_schedule(β)
ᾱ = remaining_signal(α)
model = parameters(d, 512)

dataset = [scale(square(H, W)) for _ in 1:10_000]

# Calculate loss before training on a sample
x0_test = scale(square(H, W))
ε_test = noise(x0_test)
t_test = rand(1:T)
xt_test = noised_sample(x0_test, ᾱ, t_test, ε_test)
untrained_loss = loss(model, xt_test, t_test, ε_test)

η = 1f-1
model = train(model, ᾱ, T, η, dataset)
# epochs = 100
# model = train(model, ᾱ, T, η, dataset, epochs)

# Calculate loss after training on the same sample
trained_loss = loss(model, xt_test, t_test, ε_test)
@info "untrained_loss=$(untrained_loss) trained_loss=$(trained_loss)"
@test trained_loss < untrained_loss

xgen = reverse_sample(model, β, α, ᾱ, T, d)
@info "sample mean=$(mean(xgen)) std=$(std(xgen))"
xhat = reshape(xgen, H, W)

@test size(xhat) == (H, W)
@test eltype(xhat) == Float32
@test !all(iszero, xhat)


using Plots

# Make one toy image
H, W = 16, 16
img = scale(square(H, W))   # 256-element Vector{Float32}

# Reshape to 2-D and plot
heatmap(reshape(img, H, W),
        color=:grays,
        aspect_ratio=:equal,
        title="Random generated square")

# Generate one sample from the trained model
xgen = reverse_sample(model, β, α, ᾱ, T, d)

# Reshape to 16×16 and show as grayscale
heatmap(reshape(xgen, H, W),
        color=:grays,
        aspect_ratio=:equal,
        title="Sample from trained diffusion model")

# x = randn(Float32, d)
# t = rand(1:T)
# ε̂ = predict(model, x, t)
# μ = posterior_mean(x, ε̂, β, α, ᾱ, t)
