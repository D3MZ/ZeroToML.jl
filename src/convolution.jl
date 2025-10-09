using Tullio, LoopVectorization, Flux, Test, BenchmarkTools, Statistics

"Tullio Convolution"
convolution(x,k) = @tullio y[i+_, j+_] := x[i+a, j+b] * k[a,b]
"Tullio Convolution with Padding"
convolution(x,k;p=0) = @tullio y[i+_, j+_] := x[pad(i-a,p), pad(j-b,p)] * k[a,b]

"Tullio Convolution with channels"
convolution_channels(x,k) = @tullio y[i+_, j+_, c_out] := x[i+a, j+b, c_in] * k[end-a+1, end-b+1, c_in, c_out]

function convolution_manual(x, k)
    (img_rows, img_cols) = size(x)
    (kernel_rows, kernel_cols) = size(k)
    output_rows = img_rows - kernel_rows + 1
    output_cols = img_cols - kernel_cols + 1
    output = zeros(eltype(x), output_rows, output_cols)

    for i in 1:output_rows
        for j in 1:output_cols
            accumulator = zero(eltype(x))
            for ki in 1:kernel_rows
                for kj in 1:kernel_cols
                    accumulator += x[i + ki - 1, j + kj - 1] * k[kernel_rows - ki + 1, kernel_cols - kj + 1]
                end
            end
            output[i, j] = accumulator
        end
    end
    return output
end

function convolution_manual_channels(x, k)
    (img_rows, img_cols, in_channels) = size(x)
    (kernel_rows, kernel_cols, _, out_channels) = size(k)
    output_rows = img_rows - kernel_rows + 1
    output_cols = img_cols - kernel_cols + 1
    output = zeros(eltype(x), output_rows, output_cols, out_channels)

    for cout in 1:out_channels
        for i in 1:output_rows
            for j in 1:output_cols
                accumulator = zero(eltype(x))
                for cin in 1:in_channels
                    for ki in 1:kernel_rows
                        for kj in 1:kernel_cols
                            accumulator += x[i + ki - 1, j + kj - 1, cin] * k[kernel_rows - ki + 1, kernel_cols - kj + 1, cin, cout]
                        end
                    end
                end
                output[i, j, cout] = accumulator
            end
        end
    end
    return output
end

@testset "Convolution" begin
    x = rand(Float32, 10, 10)
    k = rand(Float32, 3, 3)

    y_tullio = convolution(x, k)
    y_manual = convolution_manual(x, k)

    x_flux = reshape(x, size(x)..., 1, 1)
    k_flux = reshape(k, size(k)..., 1, 1)
    y_flux = Flux.conv(x_flux, k_flux, pad = 0) |> x -> dropdims(x, dims=(3,4))

    @test y_flux ≈ y_tullio 
    @test y_flux ≈ y_manual 

    @info "Tullio convolution benchmark:"
    tullio_bench = @benchmark convolution($x, $k)

    @info "Manual convolution benchmark:"
    @btime convolution_manual($x, $k)

    @info "Flux convolution benchmark:"
    flux_bench = @benchmark Flux.conv($x_flux, $k_flux, pad = 0)

    @test median(tullio_bench).time < median(flux_bench).time
end

@testset "Convolution with Channels" begin
    in_channels = 3
    out_channels = 4
    x = rand(Float32, 10, 10, in_channels)
    k = rand(Float32, 3, 3, in_channels, out_channels)

    y_manual_channels = convolution_manual_channels(x, k)
    y_tullio_channels = convolution_channels(x,k)

    x_flux = reshape(x, size(x)..., 1)
    y_flux = Flux.conv(x_flux, k, pad = 0) |> x -> dropdims(x, dims=4)

    @test y_flux ≈ y_manual_channels
    @test y_flux ≈ y_tullio_channels

    @info "Tullio convolution with channels benchmark:"
    @btime convolution_channels($x, $k)

    @info "Manual convolution with channels benchmark:"
    @btime convolution_manual_channels($x, $k)

    @info "Flux convolution with channels benchmark:"
    @btime Flux.conv($x_flux, $k, pad = 0)
end

