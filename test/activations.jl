using ZeroToML
using Test
using LinearAlgebra, Random # Added for potential future use or consistency

@testset "Activations" begin
    @testset "softmax" begin
        @testset "Simple 1D array" begin
            x1 = [1.0, 2.0, 3.0]
            expected1 = exp.(x1) ./ sum(exp.(x1))
            @test softmax(x1) ≈ expected1
        end

        @testset "2D array, column-wise softmax" begin
            x2 = [1.0 2.0; 3.0 4.0]
            expected2 = [exp(1.0) exp(2.0); exp(3.0) exp(4.0)]
            expected2[:, 1] ./= sum(expected2[:, 1])
            expected2[:, 2] ./= sum(expected2[:, 2])
            @test softmax(x2, dims=1) ≈ expected2
        end

        @testset "2D array, row-wise softmax" begin
            x3 = [1.0 2.0; 3.0 4.0]
            expected3 = [exp(1.0) exp(2.0); exp(3.0) exp(4.0)]
            expected3[1, :] ./= sum(expected3[1, :])
            expected3[2, :] ./= sum(expected3[2, :])
            @test softmax(x3, dims=2) ≈ expected3
        end

        @testset "Array with negative values" begin
            x4 = [-1.0, 0.0, 1.0]
            expected4 = exp.(x4) ./ sum(exp.(x4))
            @test softmax(x4) ≈ expected4
        end

        @testset "Large values stability" begin
            x5 = [100.0, 101.0, 102.0]
            expected5 = exp.(x5 .- maximum(x5)) ./ sum(exp.(x5 .- maximum(x5)))
            @test softmax(x5) ≈ expected5
        end
    end
end
