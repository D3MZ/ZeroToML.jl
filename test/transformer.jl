using ZeroToML
using Test
using Random
using Zygote

@testset "Generic Transformer" begin
    @testset "attention masks keys per query" begin
        Q = zeros(Float32, 2, 2)
        K = zeros(Float32, 2, 2)
        V = Float32[1 10; 2 20]

        unrestricted = scaled_dot_product_attention(Q, K, V)
        @test unrestricted ≈ Float32[5.5 5.5; 11 11]

        identity_allowed = Bool[1 0; 0 1]
        restricted = scaled_dot_product_attention(Q, K, V; allowed = identity_allowed)
        @test restricted ≈ V

        @test causal_mask(3) == Bool[1 1 1; 0 1 1; 0 0 1]
    end

    @testset "encoder is shape- and gradient-safe" begin
        Random.seed!(7)
        enc = TransformerEncoder()
        X = randn(Float32, 8, 5)
        Y = forward(enc, X)
        @test size(Y) == size(X)
        @test all(isfinite, Y)

        (∇,) = gradient(e -> sum(abs2, forward(e, X)), enc)
        @test ∇.Wq !== nothing
        @test size(∇.Wq) == size(enc.Wq)
        @test all(isfinite, ∇.Wq)
    end
end
