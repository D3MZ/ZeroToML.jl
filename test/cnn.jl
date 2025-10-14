using Test, Random, Statistics, NNlib, Tullio, LoopVectorization, BenchmarkTools, Zygote

# "Apply convolution filter w to input x. x and w are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. x and w may have real or complex element types."

convolution(x, w) = convolution(x, w, 1, 0)
convolution(x, w, stride) = convolution(x, w, stride, 0)

function convolution(x, w, stride, pad)
    Wx, Hx, C_in, N = size(x)
    Ww, Hw, C_in_w, C_out = size(w)
    @assert C_in == C_in_w "Input channels must match kernel input channels"

    if pad > 0
        x_padded = zeros(eltype(x), Wx + 2*pad, Hx + 2*pad, C_in, N)
        x_padded[pad+1:end-pad, pad+1:end-pad, :, :] = x
        x = x_padded
    end

    Wo = (size(x, 1) - Ww) ÷ stride + 1
    Ho = (size(x, 2) - Hw) ÷ stride + 1

    w_flipped = @view w[end:-1:1, end:-1:1, :, :]
    @tullio y[wo, ho, co, n] := x[(wo-1)*$stride+kw, (ho-1)*$stride+kh, ci, n] * w_flipped[kw, kh, ci, co] (wo in 1:Wo, ho in 1:Ho, kw in 1:Ww, kh in 1:Hw)
end

@testset "convolution" begin
    for (stride, pad) in [(1, 0), (2, 1), (2, 0)]
        @testset "stride=$stride, pad=$pad" begin
            x = rand(Float32, 10, 10, 3, 2)
            w = rand(Float32, 3, 3, 3, 4)
            
            y_manual = convolution(x, w, stride, pad)
            y_nnlib = NNlib.conv(x, w; stride=stride, pad=pad)
            
            @test y_manual ≈ y_nnlib

            b_manual = @benchmark convolution($x, $w, $stride, $pad) samples=3
            b_nnlib = @benchmark NNlib.conv($x, $w; stride=$stride, pad=$pad) samples=3
            @info "Benchmark (stride=$stride, pad=$pad):" convolution=median(b_manual) NNlib.conv=median(b_nnlib)

            y_true = rand(eltype(y_manual), size(y_manual))
            
            loss(x, w) = sum(abs2, convolution(x, w, stride, pad) - y_true)
            gs_manual = Zygote.gradient(loss, x, w)

            loss_nnlib(x, w) = sum(abs2, NNlib.conv(x, w; stride=stride, pad=pad) - y_true)
            gs_nnlib = Zygote.gradient(loss_nnlib, x, w)

            @test gs_manual[1] ≈ gs_nnlib[1]
            @test gs_manual[2] ≈ gs_nnlib[2]
        end
    end
end
