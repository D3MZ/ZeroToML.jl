using Test
using LinearAlgebra
using Random

Random.seed!(42)

@testset "KalmanFilter" begin
    @testset "1D trivial — position stays still" begin
        Φ = Float32[1 0; 0 1]
        M = Float32[1 0]         # 1×2
        Q = 1f-6 * I
        x₀ = Float32[42, 0]
        P₀ = Float32[1 0; 0 1]

        kf = KalmanFilter(Φ, M, Q, Float32[0.01;;], x₀, P₀)

        for _ in 1:10
            step!(kf, Float32[42])
        end
        @test kf.x[1] ≈ 42f0 atol=0.1
    end

    @testset "2D constant velocity tracking" begin
        dt = 1f0
        Φ = Float32[1 dt; 0 1]
        M = Float32[1 0]
        Q = 1f-4 * I
        x₀ = Float32[0, 1]
        P₀ = Float32[1 0; 0 1]

        kf = KalmanFilter(Φ, M, Q, Float32[0.01;;], x₀, P₀)

        T = 30
        xs = Matrix{Float32}(undef, 2, T)
        ys = Matrix{Float32}(undef, 1, T)
        x = copy(x₀)
        for t in 1:T
            x = Φ * x
            ys[:, t] = M * x
            xs[:, t] = x
        end

        for t in 1:T
            step!(kf, ys[:, t])
        end

        @test kf.x[1] ≈ xs[1, end] atol=1.5
        @test kf.x[2] ≈ xs[2, end] atol=1.5
    end

    @testset "covariance shrinks with more observations" begin
        Φ = Float32[1 0; 0 1]
        M = Float32[1 0]
        Q = 1f-4 * I
        x₀ = Float32[0, 0]
        P₀ = Float32[10 0; 0 10]

        kf = KalmanFilter(Φ, M, Q, Float32[0.25;;], x₀, P₀)
        P_initial = kf.P[1, 1]
        for _ in 1:50
            step!(kf, Float32[randn()])
        end
        @test kf.P[1, 1] < P_initial / 2
    end
end

@testset "simulate" begin
    Φ = Float32[1 1; 0 1]
    M = Float32[1 0]
    Q = Float32[1 0; 0 1] .* 1f-2
    x₀ = Float32[0, 0]
    T = 20

    xs, ys = simulate(Φ, M, Q, x₀, T)
    @test size(xs) == (2, T)
    @test size(ys) == (1, T)
    @test all(isfinite, xs)
    @test all(isfinite, ys)

    kf = KalmanFilter(Φ, M, Q, Float32[0.25;;], x₀, Float32[10 0; 0 10])
    for t in 1:T
        step!(kf, ys[:, t])
    end
    @test all(isfinite, kf.x)
    @test all(isfinite, kf.P)
end
