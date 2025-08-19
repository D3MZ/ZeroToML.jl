using ZeroToML
using Test

@testset "ZeroToML.jl" begin
    @testset "Encoding" begin
        include("encoding.jl")
    end
    @testset "Transformers" begin
        include("transformers.jl")
    end

end
