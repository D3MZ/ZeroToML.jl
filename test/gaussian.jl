using ZeroToML
using Test
using LinearAlgebra
using Statistics

@testset "GaussianProcess" begin
    x = collect(range(-1f0, 1f0; length=15))
    X = reshape(x, :, 1)
    y = sin.(2f0 * π .* x)

    gp = GaussianProcess()
    train!(gp, X, y)

    μ, Σ = predict(gp, X)
    @test mean(abs.(μ .- y)) < 0.15f0
    @test Σ isa Symmetric
    @test maximum(abs.(diag(Matrix(Σ)))) < 1f-2

    kernel = squared_exponential_kernel(; ℓ=0.5f0, σ=1f0)
    gp_custom = GaussianProcess(; kernel=kernel, noise=1f-6)
    x̃ = Float32[-1, 0, 1]
    X̃ = reshape(x̃, :, 1)
    ỹ = Float32[0, 1, 0]
    train!(gp_custom, X̃, ỹ)

    Xₛ = reshape(Float32[-0.5, 0.5], :, 1)
    μₛ, Σₛ = predict(gp_custom, Xₛ)
    @test length(μₛ) == 2

    Kₛ = covariance(gp_custom.kernel, Xₛ, gp_custom.X)
    @test size(Kₛ) == (2, 3)

    eigenvalues = eigvals(Matrix(Σₛ))
    @test all(eigenvalues .>= -1f-5)
end
