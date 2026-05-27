using ZeroToML
using Test
using Random

@testset "TRM" begin
    Random.seed!(1)

    text = "A quick brown fox jumps over the lazy dog. "
    vocab = build_vocab(text)
    context = 12
    x = encode(text[begin:context], vocab)
    y = encode(text[begin+1:context+1], vocab)

    model = TRM(vocab=length(vocab), context=context, width=20)
    logits, answer, latent = forward(x, model; n=2, T=2)

    @test size(logits) == (length(vocab), context)
    @test size(answer) == (20, context)
    @test size(latent) == (20, context)
    @test length(predict(model, x; n=2, T=2)) == length(y)
    @test 0 ≤ halt(model, answer) ≤ 1

    initial = loss(model, x, y; n=2, T=2)
    trained = train(model, x, y, 0.05f0, 200; n=2, T=2)
    final = loss(trained, x, y; n=2, T=2)
    generated = decode(predict(trained, x; n=2, T=2), vocab)
    actual = text[begin+1:context+1]

    @info "Post-train TRM loss" loss=final
    @info "Generated TRM" input=text[begin:context] generated=generated actual=actual
    @test final < initial
    @test final < 1f-2
    @test generated == actual
end
