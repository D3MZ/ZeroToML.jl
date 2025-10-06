using Random, Statistics, Zygote, NNlib

"Relu Activation function"
relu(x::AbstractArray) = max.(x, zero(eltype(x)))
relu(x::Number)        = max(x, zero(x))

"Glorot/Xavier uniform initialization: Wᵢⱼ ~ U[-√(6/(m+n)), √(6/(m+n))] via https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
glorot(m, n) = rand(Float32, m, n) .* (2f0*sqrt(6f0/(m+n))) .- sqrt(6f0/(m+n))
glorot_conv(w, h, c_in, c_out) = (rand(Float32, w, h, c_in, c_out) .* 2f0 .- 1f0) .* sqrt(6f0 / (w * h * (c_in + c_out)))


"Initialize fully convolutional network parameters for image-to-image noise prediction"
function conv_parameters(d)
    kernel_size = 3
    channels = [1, 16, 32, 16, 1]
    c_out = channels[2]
    W_alpha_bar = reshape(glorot(c_out, 1), 1, 1, c_out, 1)
    layers = []
    for i in 1:length(channels)-1
        push!(layers, (
            W=glorot_conv(kernel_size, kernel_size, channels[i], channels[i+1]),
            b=zeros(Float32, 1, 1, channels[i+1], 1)
        ))
    end
    return (layers=layers, W_alpha_bar=W_alpha_bar)
end

"forward process; ε̂ = ϵθ(xt,t)"
function predict(m, x, t, ᾱ)
    H, W = size(x)
    h = reshape(x, H, W, 1, 1)
    padding = (size(first(m.layers).W, 1) - 1) ÷ 2

    # First layer with ᾱ injection
    h = conv(h, m.layers[1].W; pad=padding) .+ m.layers[1].b .+ m.W_alpha_bar .* ᾱ[t]
    h = relu(h)

    # Hidden layers
    for layer in m.layers[2:end-1]
        h = conv(h, layer.W; pad=padding) .+ layer.b
        h = relu(h)
    end
    
    # Final layer
    h = conv(h, m.layers[end].W; pad=padding) .+ m.layers[end].b
    
    return reshape(h, H, W)
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
loss(θ, x, t, y, ᾱ) = mean((y .- predict(θ, x, t, ᾱ)).^2)
"Stochastic Gradient Descent (SGD). m, ∇, η are mlp_parameters, gradients, and learning rate respectively"
function sgd(m, ∇, η)
    layers = map(m.layers, ∇.layers) do layer, grad
        map((p, g) -> p .- η .* g, layer, grad)
    end
    W_alpha_bar = m.W_alpha_bar .- η .* ∇.W_alpha_bar
    (layers=layers, W_alpha_bar=W_alpha_bar)
end

"Performs one training step: adds noise xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε and updates model by gradient of the loss (ε̂, ε)"
function diffusion_step(m, x0, ᾱ, T; t=rand(1:T), η=1e-3f0)
    ε  = noise(x0)
    xt = noised_sample(x0, ᾱ, t, ε)
    (∇,) = gradient(θ -> loss(θ, xt, t, ε, ᾱ), m)
    sgd(m, ∇, η)
end

"Computes μₜ = (xₜ − (βₜ / √(1−ᾱₜ))·ε̂) / √αₜ for the reverse diffusion mean"
posterior_mean(x, ε̂, β, α, ᾱ, t) = (x .- (β[t]/sqrt(1-ᾱ[t])) .* ε̂) ./ sqrt(α[t])

"Draws a sample xₜ₋₁ ~ μ + √βₜ · N(0, I) from the reverse diffusion step"
latent(μ, β, t, x) = μ .+ sqrt(β[t]) .* randn(eltype(x), size(x))

"Generates ~x0 by iteratively sampling xₜ₋₁ = μₜ(xₜ, ε̂) + √βₜ·z for t = T,…,1, starting from x_T ~ N(0,I). "
function reverse_sample(m, β, α, ᾱ, T, d)
    H = W = isqrt(d)
    x = randn(Float32, H, W)
    μ = similar(x)
    for t in T:-1:2
        ε̂ = predict(m, x, t, ᾱ)
        μ = posterior_mean(x, ε̂, β, α, ᾱ, t)
        x = latent(μ, β, t, x)
    end
    
    t = 1
    ε̂ = predict(m, x, t, ᾱ)
    posterior_mean(x, ε̂, β, α, ᾱ, t)
end

"Trains the diffusion model over the dataset by repeatedly applying one training step"
diffusion_train(model, ᾱ, T, η, dataset) = foldl((m, x0) -> diffusion_step(m, x0, ᾱ, T; η=η), dataset; init=model)
"Trains for E epochs by folding `diffusion_train(model, ᾱ, T, η, dataset)` over epochs: mₑ = foldl((m,_)->diffusion_train(m, ᾱ, T, η, dataset), 1:E; init=model)"
diffusion_train(model, ᾱ, T, η, dataset, epochs) = foldl((m, _) -> diffusion_train(m, ᾱ, T, η, dataset), 1:epochs; init=model)

"Generates a square of 255s against a 0s background at a specified location"
function square(h, w, i, j)
    img = zeros(Int, h, w)
    img[i-1:i+1, j-1:j+1] .= 255
    return img
end

"Generates all possible unique squares on a black background"
squares(h, w) = [square(h, w, i, j) for i in 2:h-1 for j in 2:w-1]

"Scales an image from [0, 255] to [-1, 1]"
scale(img) = (2.0f0 .* Float32.(img) ./ 255.0f0) .- 1.0f0

# Below is just a scratch pad -- will delete after
using Test, Plots
Random.seed!(42)
H,W = 16, 16
d = H*W
all_squares = squares(H, W)
plot(rand(all_squares))
# dataset = [scale(rand(all_squares)) for _ in 1:100_000]

# T = 1_000
# β = noise_schedule(T)
# α = signal_schedule(β)
# ᾱ = remaining_signal(α)
# model = conv_parameters(d)

# # Calculate loss before training on a sample
# x0_test = scale(rand(all_squares))
# ε_test = noise(x0_test)
# t_test = rand(1:T)
# xt_test = noised_sample(x0_test, ᾱ, t_test, ε_test)
# untrained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)

# η = 1f-3
# @time model = diffusion_train(model, ᾱ, T, η, dataset)
# # epochs = 1
# # @code_warntype diffusion_train(model, ᾱ, T, η, dataset, epochs)
# # using BenchmarkTools
# # @benchmark diffusion_train(model, ᾱ, T, η, dataset, epochs)
# # @time model = diffusion_train(model, ᾱ, T, η, dataset, epochs)

# # # Calculate loss after training on the same sample
# trained_loss = loss(model, xt_test, t_test, ε_test, ᾱ)
# @info "untrained_loss=$(untrained_loss) trained_loss=$(trained_loss)"
# @test trained_loss < untrained_loss

# # # Reshape to 2-D and plot
# heatmap(first(dataset),
#         color=:grays,
#         aspect_ratio=:equal,
#         title="Random generated square")

# # Generate a 5×2 grid (10 samples) from the trained model
# samples = [reverse_sample(model, β, α, ᾱ, T, d) for _ in 1:10]
# plots = [heatmap(samples[i],
#                  color=:grays,
#                  aspect_ratio=1,
#                  axis=false,
#                  framestyle=:none,
#                  xticks=false,
#                  yticks=false,
#                  colorbar=false) for i in 1:length(samples)]
# plot(plots...;
#      layout=(5,2),
#      size=(300,500))
