using Test, Random, Statistics, NNlib, Tullio, LoopVectorization, BenchmarkTools

# "Apply convolution filter w to input x. x and w are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. x and w may have real or complex element types."