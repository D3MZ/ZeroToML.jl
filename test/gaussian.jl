using ZeroToML
using Test
using LinearAlgebra
using Statistics
using Plots

function plot_gp_posterior(X_data, y_data, X_prior; n_samples=4)
    gp = GaussianProcess()
    train!(gp, X_data, y_data)

    μ, Σ = predict(gp, X_prior)
    std = sqrt.(diag(Σ))

    n_pts = size(X_data, 1)
    p = plot(X_prior, μ; ribbon=2 .* std, label="Mean", title="$n_pts data points", legend=false, ylims=(-3,3))
    
    if !isempty(y_data)
        scatter!(p, X_data, y_data; label="Data", markersize=3)
    end

    L = cholesky(Σ + (gp.noise + 1f-6) * I).L
    for _ in 1:n_samples
        y_sample = μ + L * randn(Float32, length(μ))
        plot!(p, X_prior, y_sample; label="", linestyle=:dash)
    end
    return p
end

# @testset "GaussianProcess" begin
    x_range = -5f0:0.1f0:5f0
    X_prior = reshape(collect(x_range), :, 1)

    # Full dataset with 8 points
    X_data_full = reshape(Float32[-4, -3, -2, -1, 1, 2, 3, 4], :, 1)
    y_data_full = Float32[1.5, -1, 1, -1.5, 1.5, -1, 1, -1.5]

    plot_counts = [0, 1, 2, 3, 4, 5, 6, 8]
    plots_list = []
    for n_pts in plot_counts
        X_d = X_data_full[1:n_pts, :]
        y_d = y_data_full[1:n_pts]
        p = plot_gp_posterior(X_d, y_d, X_prior; n_samples=4)
        push!(plots_list, p)
    end

    final_plot = plot(plots_list...; layout=(4,2), size=(800, 1600))
    display(final_plot)

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
