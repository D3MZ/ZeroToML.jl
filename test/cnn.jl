using Test, Random, Statistics, NNlib, Tullio, LoopVectorization, Zygote

@testset "Convolution" begin
    for stride in [1, 2]
        @testset "stride=$stride" begin
            x = rand(Float32, 10, 10, 3, 2)
            w = rand(Float32, 3, 3, 3, 4)
            
            y_manual = convolution(x, w, stride)
            y_nnlib = NNlib.conv(x, w; stride=stride)
            
            @test y_manual ≈ y_nnlib

            y_true = rand(eltype(y_manual), size(y_manual))
            
            loss(x, w) = sum(abs2, convolution(x, w, stride) - y_true)
            gs_manual = Zygote.gradient(loss, x, w)

            loss_nnlib(x, w) = sum(abs2, NNlib.conv(x, w; stride=stride) - y_true)
            gs_nnlib = Zygote.gradient(loss_nnlib, x, w)

            @test gs_manual[1] ≈ gs_nnlib[1]
            @test gs_manual[2] ≈ gs_nnlib[2]
        end
    end
end
