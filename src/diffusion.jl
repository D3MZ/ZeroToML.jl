using Random, Statistics, Test, Zygote

# -------------------------
# Tiny MLP noise predictor ε_θ(x_t, t)
# -------------------------
# Initialize MLP parameters for dimension d -> d (noise prediction)
function parameters(d, h=1024)
    W1 = 0.02f0*randn(Float32, h, d); b1 = zeros(Float32, h)
    W2 = 0.02f0*randn(Float32, d, h); b2 = zeros(Float32, d)
    return (W1=W1, b1=b1, W2=W2, b2=b2)
end

relu(x::AbstractArray) = max.(x, zero(eltype(x)))
relu(x::Number)        = max(x, zero(x))

# forward: returns ε̂
function predict(m, x::Vector{Float32})
    h1 = relu(m.W1*x .+ m.b1)
    y  = m.W2*h1 .+ m.b2
    return y
end

noise(x) = randn(eltype(x), size(x))
"The entire noise variance schedule via β_t = β_min + (β_max - β_min) * (t-1)/(T-1)"
noise_schedule(T; βmin=1f-4, βmax=0.02f0) = range(βmin, βmax; length=T)
"Entire signal variance schedule: α_t = 1 - β_t"
signal_schedule(β::AbstractRange) = 1 .- β
"the total remaining signal variance"
remaining_signal(α::AbstractRange) = cumprod(α)
"Conditional marginal mean E[xₜ | x₀] for the forward diffusion process q(xₜ | x₀)"
marginal_mean(x, ᾱ, t) = sqrt(ᾱ[t]) .* x
"Conditional marginal noise for the forward diffusion marginal q(xₜ | x₀). This is the random Gaussian noise part added to the deterministic mean √ᾱₜ · x₀."
marginal_noise(ᾱ, t, ε) = sqrt(1-ᾱ[t]).*ε
"Forward noise sample q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε, with ε ~ N(0, I)"
noised_sample(x0, ᾱ, t, ε) = marginal_mean(x0, ᾱ, t) .+ (sqrt(1-ᾱ[t]) .* ε)
"MSE Loss function"
loss(θ, x, y) = mean((predict(θ, x) .- y).^2)

sgd(m, ∇, η) = map((p, g) -> p .- η .* g, m, ∇)

function step(m, x0::Vector{Float32}, ᾱ, T; t=rand(1:T), η=1e-3f0)
    ε  = noise(x0)
    xt = noised_sample(x0, ᾱ, t, ε)
    (∇,) = gradient(θ -> loss(θ, xt, ε), m)
    sgd(m, ∇, η)
end

# -------------------------
# Reverse sampling (unconditional)
# x_T ~ N(0, I); iterate to x_0
# -------------------------
function reverse_sample(m, betas, α, ᾱ, T, d; σ_type=:fixed)
    x = randn(Float32, d)
    for t in T:-1:1
        # predict ε
        ε̂ = predict(m, x)

        # μ_θ
        μ = (x .- (betas[t]/sqrt(1-ᾱ[t])).*ε̂) ./ sqrt(α[t])

        if t>1
            σt = σ_type==:fixed ? sqrt(betas[t]) : sqrt(((1-ᾱ[t-1])/(1-ᾱ[t]))*betas[t])
            x = μ .+ σt.*randn(eltype(x), size(x))
        else
            x = μ
        end
    end
    return x
end

@testset "Diffusion Toy Driver" begin
    Random.seed!(42)
    C,H,W = 1, 16, 16
    d = C*H*W
    T = 100
    β = noise_schedule(T)
    α = signal_schedule(β)
    ᾱ = remaining_signal(α)
    model = parameters(d, 512)

    # dummy dataset: e.g., small blobs
    function toy_image()
        img = zeros(Float32, H, W)
        i = rand(4:12); j = rand(4:12)
        img[i-1:i+1, j-1:j+1] .= 1f0
        return reshape(img, d)  # flatten
    end

    # Calculate loss before training on a sample
    x0_test = toy_image()
    ε_test = noise(x0_test)
    t_test = rand(1:T)
    xt_test = noised_sample(x0_test, ᾱ, t_test, ε_test)
    untrained_loss = loss(model, xt_test, ε_test)

    η = 1f-1
    for it in 1:1000
        x0 = toy_image()
        model = step(model, x0, ᾱ, T; η=η)
    end

    # Calculate loss after training on the same sample
    trained_loss = loss(model, xt_test, ε_test)
    @info "untrained_loss=$(untrained_loss) trained_loss=$(trained_loss)"
    @test trained_loss < untrained_loss

    xgen = reverse_sample(model, β, α, ᾱ, T, d)
    @info "sample mean=$(mean(xgen)) std=$(std(xgen))"
    xhat = reshape(xgen, H, W)

    @test size(xhat) == (H, W)
    @test eltype(xhat) == Float32
    @test !all(iszero, xhat)
end
