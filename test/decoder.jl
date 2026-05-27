using ZeroToML
using Test
using Statistics
using BenchmarkTools
using Random

@testset "Decoder" begin
    Random.seed!(0xBAADF00D)
    text = "A quick brown fox jumps over the lazy dog. " ^ 2
    vocab = build_vocab(text)
    x = encode(text[1:end-1], vocab)
    y = encode(text[2:end], vocab)

    learning_rate = 9f-1
    epochs  = 1_000
    max_seq_len = length(text)
    model = Decoder()
    model = train(model, x, y, max_seq_len, learning_rate, epochs)
    ℓ = loss(model, x, y)
    @info "Post-train loss" loss=ℓ
    @test ℓ < 1e-3

    n_generate = 40
    seed_len = 10
    seed_text = text[begin:seed_len]
    generated = generate(model, vocab, seed_text; n=n_generate)
    actual_text = text[begin:seed_len+n_generate]
    @info "Generated" seed=seed_text generated=generated actual=actual_text
    @test generated == actual_text
end
