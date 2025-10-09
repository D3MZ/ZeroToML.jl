using Tullio, LoopVectorization, Flux, Test, BenchmarkTools

"Tullio Convolution"
convolution(x,k) = @tullio y[i+_, j+_] := x[i+a, j+b] * k[a,b]
"Tullio Convolution with Padding"
convolution(x,k;p=0) = @tullio y[i+_, j+_] := x[pad(i-a,p), pad(j-b,p)] * k[a,b]

@testset "Convolution" begin
    x = rand(Float32, 10, 10)
    k = rand(Float32, 3, 3)

    y_tullio = convolution(x, k)

    x_flux = reshape(x, size(x)..., 1, 1)
    k_flux = reshape(k, size(k)..., 1, 1)
    y_flux = Flux.conv(x_flux, k_flux, pad = 0) |> x -> dropdims(x, dims=(3,4))

    @test y_tullio ≈ y_flux

    @info "Tullio convolution benchmark:"
    tullio_bench = @benchmark convolution($x, $k)

    @info "Flux convolution benchmark:"
    flux_bench = @benchmark Flux.conv($x_flux, $k_flux, pad = 0)

    @test tullio_bench < flux_bench
end

