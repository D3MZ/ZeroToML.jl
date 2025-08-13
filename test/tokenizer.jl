text = "hello world"
vocab = build_vocab(text)

@testset "build_vocab" begin
    @test vocab == [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
end

@testset "encode" begin
    @test encode(text, vocab) == [4, 3, 5, 5, 6, 1, 8, 6, 7, 5, 2]
end

@testset "decode" begin
    encoded = [4, 3, 5, 5, 6, 1, 8, 6, 7, 5, 2]
    @test decode(encoded, vocab) == "hello world"
end

@testset "encode/decode roundtrip" begin
    @test decode(encode(text, vocab), vocab) == text
    
    text2 = "another test"
    vocab2 = build_vocab(text2)
    @test decode(encode(text2, vocab2), vocab2) == text2
end
