using Random, Statistics, Zygote, NNlib, Tullio, LoopVectorization, CUDA, Flux

"Relu Activation function"
relu(x::AbstractArray) = max.(x, zero(eltype(x)))
relu(x::Number)        = max(x, zero(x))

"Glorot/Xavier uniform initialization: Wᵢⱼ ~ U[-√(6/(m+n)), √(6/(m+n))] via https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
glorot(m, n) = rand(Float32, m, n) .* (2f0*sqrt(6f0/(m+n))) .- sqrt(6f0/(m+n))
"Glorot/Xavier for convolution"
glorot(w, h, c_in, c_out) = (rand(Float32, w, h, c_in, c_out) .* 2f0 .- 1f0) .* sqrt(6f0 / (w * h * (c_in + c_out)))

"Initialize fully convolutional network parameters for image-to-image noise forwardion"
function parameters()
    (
        W₁ = glorot(3, 3, 1, 16),
        b₁ = zeros(Float32, 1, 1, 16, 1),
        W₂ = glorot(3, 3, 16, 32),
        b₂ = zeros(Float32, 1, 1, 32, 1),
        W₃ = glorot(3, 3, 32, 16),
        b₃ = zeros(Float32, 1, 1, 16, 1),
        W₄ = glorot(3, 3, 16, 1),
        b₄ = zeros(Float32, 1, 1, 1, 1),
        W_time_embedding = reshape(glorot(16, 1), 1, 1, 16, 1)
    )
end

"model's forward process: ε̂ = ϵθ(xt,t)"
function forward(m, x, t, time_embedding)
    H, W = size(x)
    h = reshape(x, H, W, 1, 1)
    padding = (size(m.W₁, 1) - 1) ÷ 2

    # Layer 1 with time embedding injection
    h = conv(h, m.W₁; pad=padding) .+ m.b₁ .+ m.W_time_embedding .* time_embedding[t]
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

noise(x) = randn!(similar(x))
"The entire noise variance schedule via β_t = β_min + (β_max - β_min) * (t-1)/(T-1)"
noise_schedule(T; βmin=1f-4, βmax=0.02f0) = range(βmin, βmax; length=T)
"Entire signal variance schedule: α_t = 1 - β_t"
signal_schedule(β::AbstractRange) = 1 .- β
"the total remaining signal variance is the cumprod of the signal_schedule"
remaining_signal(α::AbstractRange) = cumprod(α)
"Log Signal to Noise Ratio"
snr(ᾱ) = log.(ᾱ ./ (1 .- ᾱ))
"Conditional marginal mean E[xₜ | x₀] for the forward diffusion process q(xₜ | x₀)"
marginal_mean(x, ᾱ, t) = sqrt(ᾱ[t]) .* x
"Conditional marginal noise for the forward diffusion marginal q(xₜ | x₀). This is the random Gaussian noise part added to the deterministic mean √ᾱₜ · x₀."
marginal_noise(ᾱ, t, ε) = sqrt(1-ᾱ[t]).*ε
"Forward noise sample q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε, with ε ~ N(0, I)"
noised_sample(x0, ᾱ, t, ε) = marginal_mean(x0, ᾱ, t) .+ (sqrt(1-ᾱ[t]) .* ε)
"Mean boxd Error (MSE) loss used for DDPM training: Lₛᵢₘₚₗₑ(θ) := 𝐄ₜ,ₓ₀,ϵ ‖ϵ − ϵθ(√ᾱₜ·x₀ + √(1−ᾱₜ)·ϵ, t)‖²"
loss(θ, x, t, y, time_embedding) = mean((y .- forward(θ, x, t, time_embedding)).^2)
"Stochastic Gradient Descent (SGD). m, ∇, η are mlp_parameters, gradients, and learning rate respectively"
sgd!(m, ∇, η) = foreach((w, dw) -> w .-= η .* dw, m, ∇)

"Performs one training step: adds noise xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε and updates model by gradient of the loss (ε̂, ε)"
function diffusion_step!(m, x0, ᾱ, T, time_embedding; t=rand(1:T), η=1e-3f0)
    ε  = noise(x0)
    xt = noised_sample(x0, ᾱ, t, ε)
    (∇,) = gradient(θ -> loss(θ, xt, t, ε, time_embedding), m)
    sgd!(m, ∇, η)
    return m
end

"Computes μₜ = (xₜ − (βₜ / √(1−ᾱₜ))·ε̂) / √αₜ for the reverse diffusion mean"
posterior_mean(x, ε̂, β, α, ᾱ, t) = (x .- (β[t]/sqrt(1-ᾱ[t])) .* ε̂) ./ sqrt(α[t])

"Draws a sample xₜ₋₁ ~ μ + √βₜ · N(0, I) from the reverse diffusion step"
latent(μ, β, t, x) = μ .+ sqrt(β[t]) .* randn!(similar(x))

"Generates ~x0 by iteratively sampling xₜ₋₁ = μₜ(xₜ, ε̂) + √βₜ·z for t = T,…,0, starting from x_T ~ N(0,I). "
function reverse_sample(m, β, α, ᾱ, T, d, time_embedding)
    H = W = isqrt(d)
    x = randn!(similar(first(m), Float32, H, W))
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

function reverse_samples(m, β, α, ᾱ, T, d, time_embedding, N)                                                                                                                             
    samples = Vector{Matrix{Float32}}(undef, N)                                                                     
     Threads.@threads for i in eachindex(samples)                                                                   
        samples[i] = cpu(reverse_sample(m, β, α, ᾱ, T, d, time_embedding))                                              
     end                                                                                                            
    return samples                                                                                                  
end 

"Trains the diffusion model over the dataset by repeatedly applying one training step"
train!(model, ᾱ, T, η, dataset, time_embedding) = foldl((m, x0) -> diffusion_step!(m, x0, ᾱ, T, time_embedding; η=η), dataset; init=model)
# "Trains for E epochs by folding `train(model, ᾱ, T, η, dataset)` over epochs: mₑ = foldl((m,_)->train(m, ᾱ, T, η, dataset), 1:E; init=model)"
# train(model, ᾱ, T, η, dataset, epochs) = foldl((m, _) -> train(m, ᾱ, T, η, dataset), 1:epochs; init=model)
function train!(model, ᾱ, T, η, dataset, epochs, time_embedding)
    losses = Float32[]
    for _ in 1:epochs
        train!(model, ᾱ, T, η, dataset, time_embedding)
        push!(losses, cpu(loss(model, xt_test, t_test, ε_test, time_embedding)))
    end
    display(plot(losses))
    return model
end

"Creates an h×w zero matrix for a blank image"  
img(h, w) = zeros(Int, h, w)
"Paints a blocksize×blocksize block of 255s centered at (i, j) into an image (mutates)"
function addbox!(img, i, j, blocksize)
    r = (blocksize - 1) ÷ 2
    img[i-r:i+r, j-r:j+r] .= 255
    img
end
"Generates an h×w image with a blocksize×blocksize white box at (i, j)"
box(h, w, i, j, blocksize) = addbox!(img(h, w), i, j, blocksize)
"Generates all possible unique boxes on a black background"
function boxes(h, w, blocksize)
    r = (blocksize - 1) ÷ 2
    [box(h, w, i, j, blocksize) for i in 1+r:h-r for j in 1+r:w-r]
end
"Scales an image (array) from [0,255] to [-1,1] via y = (2/255)*x - 1"
scale(img::Matrix) = (2 .* Float32.(img) ./ 255) .- 1
"Scales a vector of images by mapping `scale` over elements"
scale(imgs::AbstractVector) = map(scale, imgs)

# Below is just a scratch pad -- will delete after
using Test, Plots, BenchmarkTools

Random.seed!(42)
H,W = 32, 32
d = H*W
whiteboxsize = 9
dataset = gpu.(shuffle(scale(boxes(H, W, whiteboxsize))))

T = 1_000
β = noise_schedule(T) |> gpu
α = signal_schedule(β)
ᾱ = remaining_signal(α)
s = snr(ᾱ)
time_embedding = 2 .* (s .- minimum(s)) ./ (maximum(s) - minimum(s)) .- 1
model = parameters() |> gpu

# Why log.(ᾱ ./ (1 .- ᾱ) and not ᾱ 
# plot(ᾱ, label = "ᾱ[t]") # Signal is compressed near 0 after 500 steps
# plot(ᾱ ./ (1 .- ᾱ), label = "SNR(t) = ᾱ[t] / (1 - ᾱ[t])") # Large dynamic range, but signal explodes on each end
# plot(log.(ᾱ ./ (1 .- ᾱ)), label = "log SNR(t) = log(ᾱ[t] / (1 - ᾱ[t]))") # Signal is compressed when SNR is low, and gives a large dynamic range afterwards

# Calculate loss before training on a sample
x0_test = rand(dataset)
ε_test = noise(x0_test)
t_test = rand(1:T)
xt_test = noised_sample(x0_test, ᾱ, t_test, ε_test)
untrained_loss = loss(model, xt_test, t_test, ε_test, time_embedding)

epochs = 10

@time model = train!(model, ᾱ, T, 1f-1, shuffle(dataset), epochs, time_embedding)
# model = train!(model, ᾱ, T, 1f-2, shuffle(dataset), epochs, time_embedding)
# model = train!(model, ᾱ, T, 1f-3, shuffle(dataset), epochs, time_embedding)


@time samples = reverse_samples(model, β, α, ᾱ, T, d, time_embedding, 100)
plots = [heatmap(samples[i],
                 color=:grays,
                 aspect_ratio=1,
                 axis=false,
                 framestyle=:none,
                 xticks=false,
                 yticks=false,
                 colorbar=false) for i in 1:lastindex(samples)]
p = plot(plots...;
         layout = (10,10),
         size   = (500,500))
savefig(p, "samples-epochs$epochs-$(H)x$(W)-whiteboxsize=$whiteboxsize.png")

# H = W = isqrt(d)
# x = randn(Float32, H, W)
# μ = similar(x)
# m = model
# μs = []

# for t in T:-1:1
#     ε̂ = forward(m, x, t, ᾱ)
#     μ = posterior_mean(x, ε̂, β, α, ᾱ, t)
#     push!(μs,μ)
#     x = latent(μ, β, t, x)
# end

# N = length(μs)
# frames = 120
# idxs = round.(Int, N .- (N-1) .* exp.(-5.0 .* range(0,1,length=frames)))
# idxs = unique(idxs)
# idxs[end] = N
# anim = @animate for i in idxs
#     t = T - i + 1
#     heatmap(μs[i];
#             title = "t = $t",
#             color=:grays,
#             axis=false,
#             framestyle=:none,
#             xticks=false,
#             yticks=false,
#             aspect_ratio = :equal)
# end

# mp4(anim, "denoising.mp4", fps = 30)
