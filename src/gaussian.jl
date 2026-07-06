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
    for i in axes(Σ_latent, 1)
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


# Build K_y and its explicit inverse (NAÏVE)
function fit_naive!(gp::GaussianProcess, X, y)
    gp.X = X
    gp.y = y
    K = kernel_matrix(gp.kernel, X, X)
    σ²n = gp.noise
    Ky = K .+ (σ²n .* I)                       # regularized Gram
    Ky = Symmetric(Ky)                          # keep symmetry
    Ky_inv = inv(Matrix(Ky))                    # <-- explicit inverse
    gp.L = Ky_inv                               # stash here just to hold it
    gp.α = Ky_inv * y                           # α = K_y^{-1} y
    gp
end

# Predict using explicit inverse
function predict_naive(gp::GaussianProcess, X̃)
    K_star  = kernel_matrix(gp.kernel, X̃, gp.X)       # K_*
    K_star2 = kernel_matrix(gp.kernel, X̃, X̃)         # K_{**}
    μ = K_star * gp.α
    Ky_inv = gp.L                                      # stored inverse
    Σ_latent = K_star2 .- (K_star * Ky_inv * K_star')  # Σ_*
    Σ_latent = (Σ_latent .+ Σ_latent') .* 0.5f0        # symmetrize
    Σ = Symmetric(Σ_latent)
    μ, Σ
end

# Optional: log marginal likelihood using explicit inverse
function logml_naive(gp::GaussianProcess)
    n = length(gp.y)
    Ky_inv = gp.L
    K = kernel_matrix(gp.kernel, gp.X, gp.X)
    Ky = Symmetric(K .+ (gp.noise .* I))
    quad = dot(gp.y, Ky_inv * gp.y)
    sgn, logabsdetKy = logdet(Matrix(Ky))              # uses generic logdet
    -0.5f0*quad - 0.5f0*logabsdetKy - 0.5f0*n*log(2f0*pi)
end

# Objective
objective(x::Float32) = sin(x * 2.5f0) * exp(-0.1f0 * x^2) * 2.0f0

# Hyperparameters
ℓ   = 1f0
σ   = 1f0           # signal std
σ²  = σ * σ
σ²n = 1f-5            # observation noise variance (also acts as jitter)

using Random
Random.seed!(42)
Xtrain = Float32.(-5 .+ 10 .* rand(5))              # n = 5
ytrain = Float32[objective(x) for x in Xtrain]
n = length(Xtrain)

# RBF kernel (scalar, explicit)
function k_rbf(x::Float32, xp::Float32, ℓ::Float32, σ²::Float32)
    r = (x - xp) / ℓ
    return σ² * exp(-0.5f0 * r * r)
end

# Build K (n×n) explicitly with loops
K = Matrix{Float32}(undef, n, n)
@inbounds for i in 1:n
    xi = Xtrain[i]
    @inbounds for j in 1:n
        K[i, j] = k_rbf(xi, Xtrain[j], ℓ, σ²)
    end
end

# Regularize and invert (NAIVE)
Ky     = Matrix(Symmetric(K + σ²n * I))   # make a plain Matrix for inv
Ky_inv = inv(Ky)                           # <-- explicit inverse
α      = Ky_inv * ytrain                   # α = (K + σ²n I)^{-1} y

# Prediction grid
Xstar = Float32.(collect(-5.0:0.1:5.0))
m = length(Xstar)

# Build K_* (m×n) and K_{**} (m×m) explicitly
Kstar = Matrix{Float32}(undef, m, n)
@inbounds for j in 1:m
    xj = Xstar[j]
    @inbounds for i in 1:n
        Kstar[j, i] = k_rbf(xj, Xtrain[i], ℓ, σ²)
    end
end

# For RBF, k(x*,x*) = σ² (constant on the diagonal)
Kss_diag = fill(σ², m)

# Posterior mean μ_* = K_* α
μstar = Kstar * α

# Posterior variance diag: Σ_* = K_{**} - K_* (K + σ²n I)^{-1} K_*'
# Naive full multiplication to get diag: diag(K_* Ky_inv K_*')
M = Kstar * Ky_inv * Kstar'            # m×m
diagΣ = similar(Kss_diag)
@inbounds for j in 1:m
    v = Kss_diag[j] - M[j, j]
    diagΣ[j] = v > 0f0 ? v : 0f0
end
σstar = sqrt.(diagΣ)

# Optional diagnostics (RMSE against true objective on grid)
ytrue = Float32[objective(x) for x in Xstar]
rmse  = sqrt(sum((μstar .- ytrue) .^ 2) / m)



# using Plots
# plot(Xstar, μstar; ribbon=σstar, label="GP mean ±1σ", xlabel="x", ylabel="f(x)")
# scatter!(Xtrain, ytrain; color=:red, label="train")
# plot!(Xstar, ytrue; label="true objective", lw=2)

2
# Plot difference: GP mean minus true objective
# diff = μstar .- ytrue
# plot(Xstar, diff; label="mean - true", xlabel="x", ylabel="μ(x) - f(x)")
# hline!([0.0]; linestyle=:dash, color=:black, label="0")

# objective(x) = sin(x * 2.5f0) * exp(-0.1f0 * x^2) * 2.0f0    

# kernel = squared_exponential
# noise = 1f-5
# X = Matrix{Float32}(undef, 0, 0)
# y = Float32[]
# L = Matrix{Float32}(undef, 0, 0)
# α = Float32[]

# x_range = -5f0:0.1f0:5f0
# X_search = reshape(collect(x_range), :, 1)

# X_data = Matrix{Float32}(undef, 0, size(X_search, 2))  # 0×d
# y_data = Float32[]

# μ, Σ = predict(gp, X_search)
# σ = sqrt.(clamp.(diag(Matrix(Σ)), 0f0, Inf))

# candidates = findall(x -> x == best_val, ucb_values)
# best_idx = rand(candidates)
# X_search[best_idx:best_idx, :]

# ucb_values = upper_confidence_bound(μ, σ)
# best_idx = argmax(ucb_values)
# X_search[best_idx:best_idx, :]


# kernel = squared_exponential
# noise = 1f-5
# X = Matrix{Float32}(undef, 0, 0)
# y = Float32[]
# L = Matrix{Float32}(undef, 0, 0)
# α = Float32[]

# # fit!(gp, X_data, y_data)
# K = kernel_matrix(gp.kernel, X, X)
# μ = K * α

# jitter = gp.noise + eps(eltype(K))
# factor = nothing
# K_reg = Symmetric(K + jitter .* I)
# factor = cholesky(K_reg)

# next_x = propose_next_point(gp, X_search; κ=2.0f0)
# next_y = objective(first(next_x))
# X_data = vcat(X_data, next_x)
# y_data = vcat(y_data, next_y)