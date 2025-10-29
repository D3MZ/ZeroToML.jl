using ZeroToML

using Test

@testset "ZeroToML.jl" begin
    @testset "Transformers" begin
        include("decoder.jl")
    end

    @testset "Diffusion" begin
        include("diffusion.jl")
    end

    @testset "Neural Network Primitives" begin
        include("cnn.jl")
    end

    @testset "Reinforcement Learning" begin
        include("ppo.jl")
    end

    @testset "Gaussian Processes" begin
        include("gaussian.jl")
    end
end
