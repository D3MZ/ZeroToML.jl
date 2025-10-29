using LinearAlgebra

"Gaussian process with radial basis function kernel and noisy observations"
@kwdef mutable struct GaussianProcess
    kernel = squared_exponential_kernel()
    noise = 1f-6
    X = Matrix{Float32}(undef, 0, 0)
    y = Float32[]
    L = Matrix{Float32}(undef, 0, 0)
    α = Float32[]
end

"Squared exponential kernel κ(x, x̃) = σ² · exp(-½ ‖(x - x̃) ./ ℓ‖²)"
@fastmath function squared_exponential(x, x̃; ℓ=1f0, σ=1f0)
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
function train!(gp::GaussianProcess, X, y)
    gp.X = X
    gp.y = y
    K = covariance(gp.kernel, X, X)
    K = Symmetric(K + gp.noise .* I)
    factor = cholesky(K)
    gp.L = factor.L
    gp.α = factor \ y
    gp
end

"Posterior mean μ and covariance Σ for query inputs X̃"
function predict(gp::GaussianProcess, X̃)
    Kₛ = covariance(gp.kernel, X̃, gp.X)
    μ = Kₛ * gp.α
    v = gp.L \ Kₛ'
    Kₛₛ = covariance(gp.kernel, X̃, X̃)
    Σ = Symmetric(Kₛₛ .- (v' * v))
    μ, Σ
end
