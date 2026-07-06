using Test
using Random
using Zygote
using ZeroToML

@testset "ParaRNN" begin
    Random.seed!(7)

    @testset "diagonal parallel scan solves affine recurrence" begin
        A = randn(Float32, 3, 9) .* 0.1f0
        b = randn(Float32, 3, 9)
        δ = pararnn_scan_diag(A, b)
        ref = similar(b)
        ref[:, 1] = b[:, 1]
        for l in axes(b, 2)[begin+1:end]
            ref[:, l] = A[:, l] .* ref[:, l - 1] .+ b[:, l]
        end
        @test δ ≈ ref atol=1f-5
        @test pararnn_scan_diag(A, b; threaded = true) ≈ ref atol=1f-5
    end

    @testset "2x2 block parallel scan solves affine recurrence" begin
        H, L = 2, 8
        Jcc = randn(Float32, H, L) .* 0.1f0
        Jch = randn(Float32, H, L) .* 0.1f0
        Jhc = randn(Float32, H, L) .* 0.1f0
        Jhh = randn(Float32, H, L) .* 0.1f0
        b = randn(Float32, 2H, L)
        δ = pararnn_scan_block2(Jcc, Jch, Jhc, Jhh, b)
        ref = similar(b)
        ref[:, 1] = b[:, 1]
        for l in 2:L
            c = Jcc[:, l] .* ref[1:H, l - 1] .+ Jch[:, l] .* ref[H+1:2H, l - 1] .+ b[1:H, l]
            h = Jhc[:, l] .* ref[1:H, l - 1] .+ Jhh[:, l] .* ref[H+1:2H, l - 1] .+ b[H+1:2H, l]
            ref[:, l] = vcat(c, h)
        end
        @test δ ≈ ref atol=1f-5
        @test pararnn_scan_block2(Jcc, Jch, Jhc, Jhh, b; threaded = true) ≈ ref atol=1f-5
    end

    @testset "ParaGRU Newton application matches sequential recurrence" begin
        cell = ParaGRU(input_dim = 4, hidden_dim = 5)
        X = randn(Float32, 4, 12)
        @test forward(cell, X; newton_iters = 6, threaded = false) ≈ forward_sequential(cell, X) atol=1f-4 rtol=1f-4
        @test forward(cell, X; newton_iters = 6, threaded = true) ≈ forward_sequential(cell, X) atol=1f-4 rtol=1f-4
        g = gradient(c -> sum(forward(c, X; newton_iters = 2, threaded = false)), cell)[1]
        @test g !== nothing
        @test size(g.B_z) == size(cell.B_z)
    end

    @testset "ParaLSTM Newton application matches sequential recurrence" begin
        cell = ParaLSTM(input_dim = 3, hidden_dim = 4)
        X = randn(Float32, 3, 10)
        @test forward(cell, X; newton_iters = 6, threaded = false) ≈ forward_sequential(cell, X) atol=1f-4 rtol=1f-4
        @test forward(cell, X; newton_iters = 6, threaded = true) ≈ forward_sequential(cell, X) atol=1f-4 rtol=1f-4
        g = gradient(c -> sum(forward(c, X; newton_iters = 2, threaded = false)), cell)[1]
        @test g !== nothing
        @test size(g.B_f) == size(cell.B_f)
    end

    @testset "ParaRNN language model lowers quick-brown-fox loss" begin
        text = "A quick brown fox jumps over the lazy dog. " ^ 2
        vocab = build_vocab(text)
        x = encode(text[1:end-1], vocab)
        y = encode(text[2:end], vocab)

        model = ParaRNNLanguageModel(
            E = glorot(16, length(vocab); gain = 0.5f0),
            cell = ParaGRU(input_dim = 16, hidden_dim = 16),
            W_out = glorot(length(vocab), 16; gain = 0.5f0),
            b_out = zeros(Float32, length(vocab), 1),
        )

        initial_loss = loss(model, x, y; newton_iters = 3)
        train!(model, x, y, 0.1f0, 100; newton_iters = 3)
        final_loss = loss(model, x, y; newton_iters = 3)
        @debug "Post-train quick-brown-fox ParaRNN loss" initial_loss final_loss

        @test final_loss < initial_loss
        @test final_loss < 3.1f0
        @test final_loss / initial_loss < 0.92f0
    end
end
