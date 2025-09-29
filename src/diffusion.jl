using Random, Statistics

# -------------------------
# Utilities
# -------------------------
randn_like(x) = randn(eltype(x), size(x)...)  # one-liner

# Time embedding (simple scalar scaling; replace with sinusoidal if you wish)
time_embed(t, T) = Float32(t)/Float32(T)

# -------------------------
# Beta schedule (linear)
# -------------------------
make_betas(T; βmin=1f-4, βmax=0.02f0) = range(βmin, βmax; length=T) |> collect

function make_alphas(betas)
    α = 1 .- betas
    ᾱ = collect(Iterators.accumulate(*, α))
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
function mlp_forward(m::MLP, x::Vector{Float32})
    h1 = relu(m.W1*x .+ m.b1)
    y  = m.W2*h1 .+ m.b2
    return y, (x, h1)
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

    ε̂, (x, h1) = mlp_forward(m, xt)
    resid = ε̂ .- ε
    loss = mean(resid.^2)

    # Backprop (dL/dy = 2*(y-ε)/N)
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
    dL_dx  = m.W1' * dz1  # not used here

    # SGD updates
    sgd!(m.W2, dL_dW2, η)
    sgd!(m.b2, dL_db2, η)
    sgd!(m.W1, dL_dW1, η)
    sgd!(m.b1, dL_db1, η)

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
        ε̂, _ = mlp_forward(m, x)

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

