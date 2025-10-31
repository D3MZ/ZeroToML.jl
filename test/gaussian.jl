include("../src/gaussian.jl")
using Test
using LinearAlgebra
using Statistics
using Plots

function plot_gp_posterior(X_data, y_data, X_prior, objective_fn; n_samples=4)
    gp = GaussianProcess()
    fit!(gp, X_data, y_data)

    μ, Σ = predict(gp, X_prior)
    σ = sqrt.(diag(Σ))

    n_pts = size(X_data, 1)
    p = plot(X_prior, objective_fn.(X_prior); label="Objective", color=:black, linestyle=:dot, title="$n_pts data points", legend=false, ylims=(-3,3))
    plot!(p, X_prior, μ; ribbon=2 .* σ, label="Mean")
    
    if !isempty(y_data)
        scatter!(p, X_data, y_data; label="Data", markersize=3)
    end

    L = cholesky(Symmetric(Σ) + (gp.noise + 1f-5) * I).L
    for _ in 1:n_samples
        y_sample = μ + L * randn(Float32, length(μ))
        plot!(p, X_prior, y_sample; label="", linestyle=:dash)
    end
    return p
end

function bayesian_optimization_demo()
    objective(x) = sin(x * 2.5f0) * exp(-0.1f0 * x^2) * 2.0f0
    
    x_range = -5f0:0.1f0:5f0
    X_search = reshape(collect(x_range), :, 1)

    X_data = Matrix{Float32}(undef, 0, 1)
    y_data = Float32[]

    plots_list = []
    
    for i in 1:8
        target_n_points = 2^i
        while size(X_data, 1) < target_n_points
            gp = GaussianProcess()
            fit!(gp, X_data, y_data)
            next_x = propose_next_point(gp, X_search; κ=2.0f0)
            next_y = objective(first(next_x))
            X_data = vcat(X_data, next_x)
            y_data = vcat(y_data, next_y)
        end
        p = plot_gp_posterior(X_data, y_data, X_search, objective; n_samples=4)
        push!(plots_list, p)
    end

    final_plot = plot(plots_list...; layout=(4,2), size=(800, 1600))
    if isinteractive()
        display(final_plot)
    end
    return final_plot
end

@testset "GaussianProcess" begin
    x = collect(range(-1f0, 1f0; length=15))
    X = reshape(x, :, 1)
    y = sin.(2f0 * π .* x)

    gp = GaussianProcess()
    fit!(gp, X, y)

    μ, Σ = predict(gp, X)
    @test mean(abs.(μ .- y)) < 0.16f0
    @test Σ isa Symmetric
    @test maximum(abs.(diag(Matrix(Σ)))) < 1f-2

    kernel = (x, x̃) -> squared_exponential(x, x̃; ℓ=0.5f0, σ=1f0)
    gp_custom = GaussianProcess(; kernel=kernel, noise=1f-6)
    x̃ = Float32[-1, 0, 1]
    X̃ = reshape(x̃, :, 1)
    ỹ = Float32[0, 1, 0]
    fit!(gp_custom, X̃, ỹ)

    Xₛ = reshape(Float32[-0.5, 0.5], :, 1)
    μₛ, Σₛ = predict(gp_custom, Xₛ)
    @test length(μₛ) == 2

    Kₛ = kernel_matrix(gp_custom.kernel, Xₛ, gp_custom.X)
    @test size(Kₛ) == (2, 3)

    eigenvalues = eigvals(Matrix(Σₛ))
    @test all(eigenvalues .>= -1f-5)
end

bayesian_optimization_demo()
