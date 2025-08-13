using ZeroToML
using Test

@testset "ZeroToML.jl" begin
    @testset "Transformers" begin
        include("transformers.jl")
    end
end
