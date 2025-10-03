using ZeroToML

using Test

@testset "ZeroToML.jl" begin
    @testset "Transformers" begin
        include("decoder.jl")
        include("diffusion.jl")
    end
end
