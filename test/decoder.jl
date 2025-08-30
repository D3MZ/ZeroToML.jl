using ZeroToML
using Test
using Statistics
using BenchmarkTools
using Random

@testset "Decoder" begin
    Random.seed!(0xBAADF00D)
    text = "A quick brown fox jumps over the lazy dog. " ^ 5
    vocab = build_vocab(text)
    x = encode(text[1:end-1], vocab)
    y = encode(text[2:end], vocab)

    learning_rate = 9f-1
    epochs  = 1_000
    model = parameters(vocab; max_seq_len=length(text))
    model = train!(model, x, y, epochs, learning_rate)
    ℓ = loss(model, x, y)
    @info "Post-train loss" loss=ℓ
    @test ℓ < 1e-3

    n_generate = 50
    start_idx = rand(1:(length(text) - n_generate))
    generated = generate(model, vocab, text[start_idx]; n=n_generate)
    @info "Generated" seed=text[start_idx] generated=generated actual=text[start_idx:start_idx+n_generate]
    # @test sample == text[start_idx:start_idx+n_generate]
    
    # @btime train!($model, $x, $y, 1, $learning_rate)
end
