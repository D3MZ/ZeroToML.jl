using ZeroToML
using Test

@testset "ZeroToML.jl" begin
    @testset "Activations" begin
        include("activations.jl")
    end
    # @testset "Encoding" begin
    #     include("encoding.jl")
    # end
    # @testset "Optimizers" begin
    #     include("optimizers.jl")
    # end
    @testset "Transformers" begin
        include("decoder.jl")
    end
end
