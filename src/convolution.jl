using Tullio, LoopVectorization, Flux, Test, BenchmarkTools

"Tullio Convolution"
convolution(x,k) = @tullio y[i+_, j+_] := x[i+a, j+b] * k[a,b]
"Tullio Convolution with Padding"
convolution(x,k;p=padding) = @tullio y[i+_, j+_] := x[pad(i-a,p), pad(j-b,p)] * k[a,b]

