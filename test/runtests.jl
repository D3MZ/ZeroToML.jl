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
end
