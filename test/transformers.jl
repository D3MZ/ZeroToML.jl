using ZeroToML
using Test
using LinearAlgebra, Random

@testset "Transformer Components" begin
    embed_size = 32
    seq_len = 8
    vocab_size = 10
    num_heads = 4
    num_layers = 2
    ff_hidden_size = 4 * embed_size

    @testset "Forward Pass" begin
        model = Transformer(vocab_size, embed_size, seq_len, num_heads, num_layers, ff_hidden_size)
        x = rand(1:vocab_size, seq_len)
        logits, cache = model(x)
        @test size(logits) == (vocab_size, seq_len)
    end

    @testset "Backward Pass" begin
        model = Transformer(vocab_size, embed_size, seq_len, num_heads, num_layers, ff_hidden_size)
        x = rand(1:vocab_size, seq_len)
        y = rand(1:vocab_size, seq_len)

        # Forward pass
        logits, cache = model(x)
        loss = cross_entropy_loss(logits, y)
        @test loss isa Float64

        # Backward pass
        zero_gradients!(model)
        dlogits = cross_entropy_loss_backward(logits, y)
        backward!(model, dlogits, cache)

        # Check if gradients are computed (not zero)
        @test !iszero(model.∇lm_head)
        @test !iszero(model.∇token_embedding)
        @test any(!iszero(block.mha.∇W_q) for block in model.blocks)
    end

    @testset "Full Training Step" begin
        model = Transformer(vocab_size, embed_size, seq_len, num_heads, num_layers, ff_hidden_size)
        x = rand(1:vocab_size, seq_len)
        y = rand(1:vocab_size, seq_len)

        # Initial loss
        logits_before, _ = model(x)
        loss_before = cross_entropy_loss(logits_before, y)

        # One step of training
        zero_gradients!(model)
        logits, cache = model(x)
        loss = cross_entropy_loss(logits, y)
        dlogits = cross_entropy_loss_backward(logits, y)
        backward!(model, dlogits, cache)
        update!(model, 1e-3)

        # Loss after one step
        logits_after, _ = model(x)
        loss_after = cross_entropy_loss(logits_after, y)

        @test loss_after isa Float64
    end
end
