using Random, Statistics, Test

# -------------------------
# Tiny MLP noise predictor ε_θ(x_t, t)
# (manual forward + backward for MSE)
# -------------------------
struct MLP
    W1::Array{Float32,2}; b1::Vector{Float32}
    W2::Array{Float32,2}; b2::Vector{Float32}
end

# Initialize MLP for dimension d -> d (noise prediction)
function init_mlp(d, h=1024)
    W1 = 0.02f0*randn(Float32, h, d); b1 = zeros(Float32, h)
    W2 = 0.02f0*randn(Float32, d, h); b2 = zeros(Float32, d)
    return MLP(W1, b1, W2, b2)
end

relu(x::AbstractArray) = max.(x, zero(eltype(x)))
relu(x::Number)        = max(x, zero(x))
sgd(param, grad, η) = (param .- η.*grad)

# forward: returns (ε̂, cache)
function mlp_forward(m::MLP, x::Vector{Float32})
    h1 = relu(m.W1*x .+ m.b1)
    y  = m.W2*h1 .+ m.b2
    return y, (x, h1)
end

function mlp_backward(m::MLP, cache, resid, η)
    x, h1 = cache
    N = length(resid)
    dL_dy = (2f0/N).*resid

    # y = W2*h1 + b2
    dL_dW2 = dL_dy*h1'
    dL_db2 = dL_dy

    # h1 = relu(W1*x + b1)
    dh1 = m.W2' * dL_dy
    dz1 = dh1 .* (h1 .> 0f0)

    dL_dW1 = dz1 * x'
    dL_db1 = dz1

    # SGD updates
    W1_new = sgd(m.W1, dL_dW1, η)
    b1_new = sgd(m.b1, dL_db1, η)
    W2_new = sgd(m.W2, dL_dW2, η)
    b2_new = sgd(m.b2, dL_db2, η)
    return MLP(W1_new, b1_new, W2_new, b2_new)
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


random_step(T) = rand(1:T)

function train_step(m::MLP, x0::Vector{Float32}, ᾱ, T; η=1e-3f0)
    t = random_step(T)
    ε  = noise(x0)
    xt = noised_sample(x0, ᾱ, t, ε)

    ε̂, cache = mlp_forward(m, xt)
    resid = ε̂ .- ε
    loss = mean(resid.^2)
    m_new = mlp_backward(m, cache, resid, η)
    return m_new, loss
end

# -------------------------
# Reverse sampling (unconditional)
# x_T ~ N(0, I); iterate to x_0
# -------------------------
function reverse_sample(m::MLP, betas, α, ᾱ, T, d; σ_type=:fixed)
    x = randn(Float32, d)
    for t in T:-1:1
        # predict ε
        ε̂, _ = mlp_forward(m, x)

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
    model = init_mlp(d, 512)

    # dummy dataset: e.g., small blobs
    function toy_image()
        img = zeros(Float32, H, W)
        i = rand(4:12); j = rand(4:12)
        img[i-1:i+1, j-1:j+1] .= 1f0
        return reshape(img, d)  # flatten
    end

    η = 1f-1
    losses = zeros(Float32, 100)
    for it in 1:100
        x0 = toy_image()
        model, losses[it] = train_step(model, x0, ᾱ, T; η=η)
        if it%50==0; @info "iter=$(it) loss=$(losses[it])"; end
    end
    @test mean(losses[81:100]) < mean(losses[1:20])

    xgen = reverse_sample(model, β, α, ᾱ, T, d)
    @info "sample mean=$(mean(xgen)) std=$(std(xgen))"
    xhat = reshape(xgen, H, W)

    @test size(xhat) == (H, W)
    @test eltype(xhat) == Float32
    @test !all(iszero, xhat)
end
