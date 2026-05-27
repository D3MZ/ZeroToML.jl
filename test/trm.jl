using ZeroToML
using Test
using Random

@testset "TRM" begin
    Random.seed!(1)
    model = TRM(vocab=3, context=4, width=6)
    x = [1, 2, 3, 1]
    y = [1, 2, 3, 1]

    logits, answer, latent = forward(x, model; n=2, T=2)
    @test size(logits) == (3, 4)
    @test size(answer) == (6, 4)
    @test size(latent) == (6, 4)
    @test length(predict(model, x; n=2, T=2)) == length(y)
    @test 0 ≤ halt(model, answer) ≤ 1

    initial = loss(model, x, y; n=2, T=1)
    trained = train(model, x, y, 0.05f0, 40; n=2, T=1)
    final = loss(trained, x, y; n=2, T=1)
    @test final < initial
end
