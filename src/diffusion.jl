using Random, Statistics, Test

randn_like(x) = randn(eltype(x), size(x))

# Time embedding (simple scalar scaling; replace with sinusoidal if you wish)
time_embed(t, T) = Float32(t)/Float32(T)

# -------------------------
# Beta schedule (linear)
# -------------------------
make_betas(T; βmin=1f-4, βmax=0.02f0) = range(βmin, βmax; length=T) |> collect

function make_alphas(betas)
    α = 1 .- betas
    ᾱ = cumprod(α)
    return α, ᾱ
end

# -------------------------
# Forward sampler q(x_t | x_0)
# -------------------------
q_sample(x0, t, ᾱ) = sqrt(ᾱ[t]).*x0 .+ sqrt(1-ᾱ[t]).*randn_like(x0)

# -------------------------
# Tiny MLP noise predictor ε_θ(x_t, t)
# (manual forward + backward for MSE)
# -------------------------
struct MLP
    W1::Array{Float32,2}; b1::Vector{Float32}
    W2::Array{Float32,2}; b2::Vector{Float32}
end

relu(x) = max.(x, 0f0)

# forward: returns (ε̂, cache)
function mlp_forward(m::MLP, x::Vector{Float32}, t, T)
    h1 = relu(m.W1*x .+ m.b1 .+ time_embed(t, T))
    y  = m.W2*h1 .+ m.b2
    return y, (x, h1)
end

function mlp_backward!(m::MLP, cache, resid, η)
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
    sgd!(m.W2, dL_dW2, η)
    sgd!(m.b2, dL_db2, η)
    sgd!(m.W1, dL_dW1, η)
    sgd!(m.b1, dL_db1, η)
end

# simple SGD update
sgd!(param, grad, η) = (param .-= η.*grad)

# Initialize MLP for dimension d -> d (noise prediction)
function init_mlp(d, h=1024)
    W1 = 0.02f0*randn(Float32, h, d); b1 = zeros(Float32, h)
    W2 = 0.02f0*randn(Float32, d, h); b2 = zeros(Float32, d)
    return MLP(W1, b1, W2, b2)
end

# -------------------------
# Training step: one batch = one image here (extend to minibatches easily)
# Loss: ||ε - ε̂||^2
# Manual backprop for this 2-layer MLP
# -------------------------
function train_step!(m::MLP, x0::Vector{Float32}, betas, α, ᾱ, T; η=1e-3f0)
    t = rand(1:T)
    ε  = randn_like(x0)
    xt = sqrt(ᾱ[t]).*x0 .+ sqrt(1-ᾱ[t]).*ε

    ε̂, cache = mlp_forward(m, xt, t, T)
    resid = ε̂ .- ε
    loss = mean(resid.^2)
    mlp_backward!(m, cache, resid, η)
    return loss
end

# -------------------------
# Reverse sampling (unconditional)
# x_T ~ N(0, I); iterate to x_0
# -------------------------
function reverse_sample(m::MLP, betas, α, ᾱ, T, d; σ_type=:fixed)
    x = randn(Float32, d)
    for t in T:-1:1
        # predict ε
        ε̂, _ = mlp_forward(m, x, t, T)

        # μ_θ
        μ = (x .- (betas[t]/sqrt(1-ᾱ[t])).*ε̂) ./ sqrt(α[t])

        if t>1
            σt = σ_type==:fixed ? sqrt(betas[t]) : sqrt(((1-ᾱ[t-1])/(1-ᾱ[t]))*betas[t])
            x = μ .+ σt.*randn_like(x)
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
    betas = Float32.(make_betas(T))
    α, ᾱ = make_alphas(betas)
    model = init_mlp(d, 512)

    # dummy dataset: e.g., small blobs
    function toy_image()
        img = zeros(Float32, H, W)
        i = rand(4:12); j = rand(4:12)
        img[i-1:i+1, j-1:j+1] .= 1f0
        return reshape(img, d)  # flatten
    end

    η = 1f-3
    losses = zeros(Float32, 100)
    for it in 1:100 # Reduced from 10_000 for testing
        x0 = toy_image()
        losses[it] = train_step!(model, x0, betas, α, ᾱ, T; η=η)
        if it%50==0; @info "iter=$(it) loss=$(losses[it])"; end
    end
    @test mean(losses[81:100]) < mean(losses[1:20])

    xgen = reverse_sample(model, betas, α, ᾱ, T, d)
    @info "sample mean=$(mean(xgen)) std=$(std(xgen))"
    xhat = reshape(xgen, H, W)

    @test size(xhat) == (H, W)
    @test eltype(xhat) == Float32
    @test !all(iszero, xhat)
end
