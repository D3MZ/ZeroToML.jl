using ZeroToML

using Test

@testset "ZeroToML.jl" begin
    @testset "Transformers" begin
        include("decoder.jl")
    end

    @testset "Recursive Reasoning" begin
        include("trm.jl")
    end

    @testset "Diffusion" begin
        include("ddpm.jl")
    end

    @testset "Neural Network Primitives" begin
        include("convolution.jl")
    end

    @testset "Reinforcement Learning" begin
        include("ppo.jl")
    end

    @testset "Gaussian Processes" begin
        include("gaussian.jl")
    end

    @testset "Kalman Filter" begin
        include("kalman.jl")
    end
end
