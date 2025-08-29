using ZeroToML
using Test

@testset "Decoder" begin
    text = "ABABAABBAAABBB"
    vocab = build_vocab(text)
    vocab_idx = Dict(c => i for (i, c) in enumerate(vocab))
    x = encode(text[1:end-1], vocab)
    y = encode(text[2:end], vocab)

    learning_rate = 1f-2
    epochs  = 500
    model = Parameters(vocab)
    losses, model = train!(model, x, y, epochs, learning_rate)
    
    @info "Sample: $(generate(model, 'A', length(text)))"
    @test generate(model, 'A', length(text)) == text
    @test losses[end] <= 1e-10
end