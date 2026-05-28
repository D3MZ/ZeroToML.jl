# "Apply convolution filter w to input x. x and w are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. x and w may have real or complex element types."

function convolution(x, w, stride=1)
    @assert size(x, 3) == size(w, 3) "Input channels must match kernel input channels"
    Ww, Hw = size(w, 1), size(w, 2)
    Wo = (size(x, 1) - Ww) ÷ stride + 1
    Ho = (size(x, 2) - Hw) ÷ stride + 1
    w_flipped = @view w[end:-1:1, end:-1:1, :, :]
    @tullio y[wo, ho, co, n] := x[(wo-1)*$stride+kw, (ho-1)*$stride+kh, ci, n] * w_flipped[kw, kh, ci, co] (wo in 1:Wo, ho in 1:Ho, kw in 1:Ww, kh in 1:Hw)
end
