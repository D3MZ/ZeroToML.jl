using ZeroToML
using Test

@testset "Decoder" begin
    text = "ABABAABBAAABBB"
    vocab = build_vocab(text)
    x = encode(text[1:end-1], vocab)
    y = encode(text[2:end], vocab)

    learning_rate = 1f-2
    epochs  = 1000
    model = Parameters(vocab)
    losses, model = train!(model, x, y, epochs, learning_rate)
    sample = generate(model, vocab, 'A'; n=length(text)-1)
    @info sample
    @test quantile(losses, 0.25) <= quantile(losses, 0.75)
end