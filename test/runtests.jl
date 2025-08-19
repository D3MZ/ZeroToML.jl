using ZeroToML
using Test

@testset "ZeroToML.jl" begin
    @testset "Tokenizer" begin
        include("tokenizer.jl")
    end
    @testset "Transformers" begin
        include("transformers.jl")
    end

end
