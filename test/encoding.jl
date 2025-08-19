using ZeroToML
using Test

@testset "Tokenizer Functions" begin
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
end

@testset "Positional Encoding Functions" begin
    @testset "positional_encoding" begin
        seq_len = 50
        embed_size = 20
        pe = positional_encoding(seq_len, embed_size)

        @test size(pe) == (embed_size, seq_len)
        @test all(x -> -1 <= x <= 1, pe)

        # Test with smaller values and check calculations
        seq_len_small = 10
        embed_size_small = 4
        pe_small = positional_encoding(seq_len_small, embed_size_small)

        div_term_calc = exp.((0:2:(embed_size_small-1)) .* -(log(10000.0) / embed_size_small))

        for pos in 1:seq_len_small
            for i in 0:div(embed_size_small, 2)-1
                @test pe_small[2i+1, pos] ≈ sin(pos * div_term_calc[i+1])
                @test pe_small[2i+2, pos] ≈ cos(pos * div_term_calc[i+1])
            end
        end
    end

    @testset "positional_encoding_tullio" begin
        seq_len = 50
        embed_size = 20
        @test positional_encoding(seq_len, embed_size) ≈ positional_encoding_tullio(seq_len, embed_size)

        seq_len_small = 10
        embed_size_small = 4
        @test positional_encoding(seq_len_small, embed_size_small) ≈ positional_encoding_tullio(seq_len_small, embed_size_small)
    end
end
