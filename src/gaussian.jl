using LinearAlgebra

"Gaussian process with radial basis function kernel and noisy observations"
@kwdef mutable struct GaussianProcess
    kernel = squared_exponential
    noise = 1f-5
    X = Matrix{Float32}(undef, 0, 0)
    y = Float32[]
    L = Matrix{Float32}(undef, 0, 0)
    α = Float32[]
end

"Squared exponential kernel κ(x, x̃) = σ² · exp(-½ ‖(x - x̃) ./ ℓ‖²)"
function squared_exponential(x, x̃; ℓ=1f0, σ=1f0)
    δ = x .- x̃
    scaled = δ ./ ℓ
    σ² = σ * σ
    σ² * exp(-0.5f0 * (scaled ⋅ scaled))
end

"Construct kernel matrix K_ij = κ(xᵢ, yⱼ)"
kernel_matrix(kernel, X, Y) = [kernel(X[i, :], Y[j, :]) for i in axes(X, 1), j in axes(Y, 1)]

"Condition the process on data (X, y) by computing the Cholesky factor and α = K⁻¹y"
function fit!(gp::GaussianProcess, X, y)
    gp.X = X
    gp.y = y

    K = kernel_matrix(gp.kernel, X, X)
    jitter = gp.noise + eps(eltype(K))
    factor = nothing
    for attempt in 1:6
        try
            K_reg = Symmetric(K + jitter .* I)
            factor = cholesky(K_reg)
            break
        catch err
            if err isa PosDefException && attempt < 6
                jitter *= 10f0
            else
                rethrow(err)
            end
        end
    end

    gp.L = factor.L
    gp.α = factor \ y
    gp
end

"Posterior mean μ and covariance Σ for query inputs X̃"
function predict(gp::GaussianProcess, X̃)
    Kₛ = kernel_matrix(gp.kernel, X̃, gp.X)
    μ = Kₛ * gp.α
    v = gp.L \ Kₛ'
    Kₛₛ = kernel_matrix(gp.kernel, X̃, X̃)
    Σ_latent = Kₛₛ .- (v' * v)
    Σ_latent = (Σ_latent + Σ_latent') .* 0.5f0
    for i in 1:size(Σ_latent, 1)
        if Σ_latent[i, i] < 0f0
            Σ_latent[i, i] = 0f0
        end
    end
    Σ = Symmetric(Σ_latent)
    μ, Σ
end

"Upper Confidence Bound acquisition function"
upper_confidence_bound(μ, σ; κ=1.0f0) = μ .+ κ .* σ

"Find the next point to sample by maximizing the acquisition function"
function propose_next_point(gp::GaussianProcess, X_search; κ=1.0f0)
    μ, Σ = predict(gp, X_search)
    σ = sqrt.(clamp.(diag(Matrix(Σ)), 0f0, Inf))
    ucb_values = upper_confidence_bound(μ, σ; κ=κ)
    best_idx = argmax(ucb_values)
    X_search[best_idx:best_idx, :]
end

objective(x) = sin(x * 2.5f0) * exp(-0.1f0 * x^2) * 2.0f0    
x_range = -5f0:0.1f0:5f0
X_search = reshape(collect(x_range), :, 1)

X_data = Matrix{Float32}(undef, 0, 1)
y_data = Float32[]

gp = GaussianProcess()
kernel = squared_exponential
noise = 1f-5
X = Matrix{Float32}(undef, 0, 0)
y = Float32[]
L = Matrix{Float32}(undef, 0, 0)
α = Float32[]

# fit!(gp, X_data, y_data)
K = kernel_matrix(gp.kernel, X, X)
