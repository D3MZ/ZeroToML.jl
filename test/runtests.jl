using ZeroToML
using Test

@testset "ZeroToML.jl" begin
    @testset "Transformers" begin
        # include("decoder.jl")
        include("decoder-shakespeare.jl")
    end
end
