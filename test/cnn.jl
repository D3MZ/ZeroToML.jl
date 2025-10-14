using Test, Random, Statistics, NNlib, Tullio, LoopVectorization, BenchmarkTools

# "Apply convolution filter w to input x. x and w are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. x and w may have real or complex element types."

function manual_conv(x, w; stride = 1, pad = 0)
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

    @tullio y[wo, ho, co, n] := x[(wo-1)*$stride+kw, (ho-1)*$stride+kh, ci, n] * w[kw, kh, ci, co] (wo in 1:Wo, ho in 1:Ho, kw in 1:Ww, kh in 1:Hw)
end

@testset "manual_conv" begin
    # Test case 1: stride=1, pad=0
    x = rand(Float32, 10, 10, 3, 2)
    w = rand(Float32, 3, 3, 3, 4)
    
    y_manual = manual_conv(x, w)
    y_nnlib = NNlib.conv(x, w)
    
    @test y_manual ≈ y_nnlib

    # Test case 2: stride=2, pad=1
    stride = 2
    pad = 1
    y_manual_2 = manual_conv(x, w; stride=stride, pad=pad)
    y_nnlib_2 = NNlib.conv(x, w; stride=stride, pad=pad)
    
    @test y_manual_2 ≈ y_nnlib_2
end
