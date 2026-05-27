using ZeroToML
using Test
using Statistics
using BenchmarkTools
using Random

@testset "Decoder" begin
    text = "A quick brown fox jumps over the lazy dog. " ^ 2
    vocab = build_vocab(text)
    x = encode(text[1:end-1], vocab)
    y = encode(text[2:end], vocab)

    @testset "absolute positions" begin
        learning_rate = 9f-1
        epochs = 1_000
        model = Decoder()
        model = train(model, x, y, length(text), learning_rate, epochs; position=absolute)
        ℓ = loss(model, x, y, absolute(eachindex(x)))
        @info "Post-train absolute loss" loss=ℓ
        @test ℓ < 1e-3

        n_generate = 40
        seed_len = 10
        seed_text = text[begin:seed_len]
        generated = generate(model, vocab, seed_text; n=n_generate, position=absolute, choose=argmax)
        actual_text = text[begin:seed_len+n_generate]
        @info "Generated absolute" seed=seed_text generated=generated actual=actual_text
        @test generated == actual_text
    end

    @testset "relative sequences" begin
        learning_rate = 5f-2
        epochs = 300
        n_generate = 40
        seed_len = 10
        sequence_len = seed_len + n_generate
        start_idx = rand(1:(length(text) - sequence_len))
        model = Decoder()
        model = train(model, x, y, sequence_len, learning_rate, epochs; position=relative, stride=1)
        ℓ = loss(model, x[begin:sequence_len], y[begin:sequence_len], relative(1:sequence_len))
        @info "Post-train relative loss" loss=ℓ
        @test ℓ < 5f-2

        seed_text = text[start_idx:start_idx+seed_len-1]
        generated = generate(model, vocab, seed_text; n=n_generate, position=relative, choose=argmax)
        actual_text = text[start_idx:start_idx+sequence_len-1]
        @info "Generated relative" start=start_idx seed=seed_text generated=generated actual=actual_text
        @test generated == actual_text
    end
end
