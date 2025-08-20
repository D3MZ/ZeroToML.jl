using ZeroToML
using Test
import NNlib: softmax as nn_softmax

@testset "Activations" begin
    x = randn(5, 4)
    @testset "softmax ≈ NNlib.softmax" begin
        @testset "column-wise (dims=1)" begin
            @test softmax(x, dims=1) ≈ nn_softmax(x, dims=1)
        end
        @testset "row-wise (dims=2)" begin
            @test softmax(x, dims=2) ≈ nn_softmax(x, dims=2)
        end
    end
end