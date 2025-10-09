using Tullio, LoopVectorization, Flux, Test

"Convolution with padding via Tullio"
convolution(x,k) = @tullio y[i+_, j+_] := x[i+a, j+b] * k[a,b]
convolution(x,k;p=padding) = @tullio y[i+_, j+_] := x[pad(i-a,p), pad(j-b,p)] * k[a,b]

