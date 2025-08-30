using ZeroToML
using Test
using Statistics
using BenchmarkTools

@testset "Decoder" begin
    text = "A quick brown fox jumps over the lazy dog. " ^ 5
    vocab = build_vocab(text)
    x = encode(text[1:end-1], vocab)
    y = encode(text[2:end], vocab)

    learning_rate = 1f-1
    epochs  = 1_000
    model = Parameters(vocab; max_seq_len=length(text))
    model = train!(model, x, y, epochs, learning_rate)
    start_idx = rand(1:length(text)-1)
    sample = generate(model, vocab, text[start_idx]; n=length(text)-start_idx)
    @info "Generated sample" seed=text[start_idx] sample=sample
    @test sample == text[start_idx:end]
    @test loss(model, x, y) < 1
    
    # @btime train!($model, $x, $y, 1, $learning_rate)
end
