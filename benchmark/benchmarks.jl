using ZeroToML
using BenchmarkTools
using NNlib, Random

SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# Write your benchmarks here.

Random.seed!(0)

convolution_group = BenchmarkGroup()
for stride in (1, 2)
    x = rand(Float32, 10, 10, 3, 2)
    w = rand(Float32, 3, 3, 3, 4)
    stride_group = BenchmarkGroup()
    stride_group["ZeroToML"] = @benchmarkable convolution($x, $w, $stride)
    stride_group["NNlib"] = @benchmarkable NNlib.conv($x, $w; stride=$stride)
    convolution_group["stride=$(stride)"] = stride_group
end
SUITE["convolution"] = convolution_group
