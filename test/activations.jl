using ZeroToML
using Test
import NNlib: softmax as nn_softmax

@testset "Activations" begin
    x = randn(5, 4)
    @testset "softmax ≈ NNlib.softmax" begin
        @test softmax(x, dims=2) ≈ nn_softmax(x, dims=2)
    end
end