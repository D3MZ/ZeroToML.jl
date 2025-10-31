using LinearAlgebra

"Gaussian process with radial basis function kernel and noisy observations"
@kwdef mutable struct GaussianProcess
    kernel = squared_exponential_kernel()
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

"Closure-producing helper for squared exponential kernel with fixed hyperparameters"
function squared_exponential_kernel(; ℓ=1f0, σ=1f0)
    (x, x̃) -> squared_exponential(x, x̃; ℓ=ℓ, σ=σ)
end

"Construct covariance matrix K_ij = κ(xᵢ, yⱼ)"
function covariance(kernel, X, Y)
    K = similar(X, size(X, 1), size(Y, 1))
    @views for i in axes(X, 1)
        xᵢ = X[i, :]
        for j in axes(Y, 1)
            yⱼ = Y[j, :]
            K[i, j] = kernel(xᵢ, yⱼ)
        end
    end
    K
end

"Condition the process on data (X, y) by computing the Cholesky factor and α = K⁻¹y"
function fit!(gp::GaussianProcess, X, y)
    gp.X = X
    gp.y = y
    if isempty(y)
        T = eltype(X)
        gp.L = Matrix{T}(undef, 0, 0)
        gp.α = similar(y, 0)
        return gp
    end

    K = covariance(gp.kernel, X, X)
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
    if isempty(gp.y)
        μ = zeros(Float32, size(X̃, 1))
        Σ = covariance(gp.kernel, X̃, X̃)
        return μ, Symmetric(Σ)
    end

    Kₛ = covariance(gp.kernel, X̃, gp.X)
    μ = Kₛ * gp.α
    v = gp.L \ Kₛ'
    Kₛₛ = covariance(gp.kernel, X̃, X̃)
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
