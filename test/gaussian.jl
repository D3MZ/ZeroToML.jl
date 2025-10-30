using ZeroToML
using Test
using LinearAlgebra
using Statistics
using Plots

# @testset "GaussianProcess" begin
    x_range = -5f0:0.1f0:5f0
    X_prior = reshape(collect(x_range), :, 1)
    gp_prior = GaussianProcess()
    μ, Σ = predict(gp_prior, X_prior)
    std = sqrt.(diag(Σ))
    L = cholesky(Σ + gp_prior.noise * I).L
    p = plot(X_prior, μ; ribbon=2 .* std, label="Mean prediction", title="Four samples from GP prior")
    for i in 1:4
        y_sample = μ + L * randn(Float32, length(μ))
        plot!(p, X_prior, y_sample; label="sample $i", linestyle=:dash)
    end
    display(p)

    # Posterior with two data points
    X_data = reshape([-4f0,2,4f0], :, 1)
    y_data = [1f0,0, -1f0]

    gp_posterior = GaussianProcess()
    train!(gp_posterior, X_data, y_data)

    μ_post, Σ_post = predict(gp_posterior, X_prior)

    std_post = sqrt.(diag(Σ_post))
    p_post = plot(X_prior, μ_post; ribbon=2 .* std_post, label="Mean prediction", linestyle=:solid, title="Posterior with two data points")
    scatter!(p_post, X_data, y_data; label="Observed data")

    L_post = cholesky(Σ_post + gp_posterior.noise * I).L
    for i in 1:4
        y_sample = μ_post + L_post * randn(Float32, length(μ_post))
        plot!(p_post, X_prior, y_sample; label="sample $i", linestyle=:dash)
    end
    display(p_post)

#     x = collect(range(-1f0, 1f0; length=15))
#     X = reshape(x, :, 1)
#     y = sin.(2f0 * π .* x)

#     gp = GaussianProcess()
#     train!(gp, X, y)

#     μ, Σ = predict(gp, X)
#     @test mean(abs.(μ .- y)) < 0.15f0
#     @test Σ isa Symmetric
#     @test maximum(abs.(diag(Matrix(Σ)))) < 1f-2

#     kernel = squared_exponential_kernel(; ℓ=0.5f0, σ=1f0)
#     gp_custom = GaussianProcess(; kernel=kernel, noise=1f-6)
#     x̃ = Float32[-1, 0, 1]
#     X̃ = reshape(x̃, :, 1)
#     ỹ = Float32[0, 1, 0]
#     train!(gp_custom, X̃, ỹ)

#     Xₛ = reshape(Float32[-0.5, 0.5], :, 1)
#     μₛ, Σₛ = predict(gp_custom, Xₛ)
#     @test length(μₛ) == 2

#     Kₛ = covariance(gp_custom.kernel, Xₛ, gp_custom.X)
#     @test size(Kₛ) == (2, 3)

#     eigenvalues = eigvals(Matrix(Σₛ))
#     @test all(eigenvalues .>= -1f-5)
# # end
