using ZeroToML
using Test
using LinearAlgebra, Random

const generate = ZeroToML.generate

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
        optimizer = Adam(lr=1e-3)
        update!(model, optimizer)

        # Loss after one step
        logits_after, _ = model(x)
        loss_after = cross_entropy_loss(logits_after, y)

        @test loss_after isa Float64
    end

    @testset "Training on a simple sequence" begin
        input_text = repeat("AB", 1_000)
        vocab = build_vocab(input_text)
        data = encode(input_text, vocab)

        # Model parameters
        vocab_size = length(vocab)
        embed_size = 32
        block_size = 8
        num_heads = 2
        num_layers = 2
        ff_hidden_size = 4 * embed_size

        max_pos = length(data)
        model = Transformer(vocab_size, embed_size, max_pos, num_heads, num_layers, ff_hidden_size)

        # Training parameters
        learning_rate = 1e-2
        num_steps = 100
        optimizer = Adam(lr=learning_rate)

        # Calculate initial loss
        loss_before = 0.0
        for _ in 1:10
            t = rand(1:(length(data) - block_size))
            x = data[t:(t + block_size - 1)]
            y = data[(t + 1):(t + block_size)]
            logits, _ = model(x; start_pos=t)
            loss_before += cross_entropy_loss(logits, y)
        end
        loss_before /= 10

        # Training loop
        for _ in 1:num_steps
            t = rand(1:(length(data) - block_size))
            x = data[t:(t + block_size - 1)]
            y = data[(t + 1):(t + block_size)]

            zero_gradients!(model)
            logits, cache = model(x; start_pos=t)
            dlogits = cross_entropy_loss_backward(logits, y)
            backward!(model, dlogits, cache)
            update!(model, optimizer)
        end
        
        # Calculate final loss
        loss_after = 0.0
        for _ in 1:10
            t = rand(1:(length(data) - block_size))
            x = data[t:(t + block_size - 1)]
            y = data[(t + 1):(t + block_size)]
            logits, _ = model(x; start_pos=t)
            loss_after += cross_entropy_loss(logits, y)
        end
        loss_after /= 10
        
        @test loss_after < loss_before
    end

    # @testset "Full training and generation" begin
    #     input_text = "The quick brown fox jumps over the lazy dog. " ^ 100
    #     vocab = build_vocab(input_text)
    #     data = encode(input_text, vocab)

    #     # Model parameters
    #     vocab_size = length(vocab)
    #     embed_size = 32
    #     block_size = 8
    #     num_heads = 2
    #     num_layers = 2
    #     ff_hidden_size = 4 * embed_size

    #     max_pos = length(data)
    #     model = Transformer(vocab_size, embed_size, max_pos, num_heads, num_layers, ff_hidden_size)

    #     # Training parameters
    #     learning_rate = 3e-4
    #     num_steps = 150_000
    #     optimizer = Adam(lr=learning_rate)

    #     # Training loop
    #     best_loss = Inf
    #     for step in 1:num_steps
    #         t = rand(1:(length(data) - block_size))
    #         x = data[t:(t + block_size - 1)]
    #         y = data[(t + 1):(t + block_size)]

    #         zero_gradients!(model)
    #         logits, cache = model(x; start_pos=t)
    #         dlogits = cross_entropy_loss_backward(logits, y)
    #         backward!(model, dlogits, cache)
    #         update!(model, optimizer)

    #         if step % 1000 == 0
    #             loss = cross_entropy_loss(logits, y)
    #             if loss < best_loss
    #                 best_loss = loss
    #                 @info "Step $step, best loss = $best_loss"
    #             end
    #         end
    #     end

    #     # Generation
    #     # start_index = rand(1:(length(data) - block_size))
    #     start_index = 1
    #     context_indices = data[start_index:(start_index + block_size - 1)]
    #     context_str = decode(context_indices, vocab)
        
    #     num_generate = 50
    #     generated_indices = generate(model, context_indices, num_generate; start_pos=start_index)
    #     generated_text = decode(generated_indices, vocab)

    #     @info "Full training test generation" context=context_str generated=generated_text
        
    #     expected_indices = data[start_index:(start_index + block_size - 1 + num_generate)]
    #     @test generated_indices == expected_indices
    # end
end

@testset "Tokenizer Functions" begin
    text = "hello world"
    vocab = build_vocab(text)

    @testset "build_vocab" begin
        @test vocab == [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
    end

    @testset "encode" begin
        @test encode(text, vocab) == [4, 3, 5, 5, 6, 1, 8, 6, 7, 5, 2]
    end

    @testset "decode" begin
        encoded = [4, 3, 5, 5, 6, 1, 8, 6, 7, 5, 2]
        @test decode(encoded, vocab) == "hello world"
    end

    @testset "encode/decode roundtrip" begin
        @test decode(encode(text, vocab), vocab) == text
        
        text2 = "another test"
        vocab2 = build_vocab(text2)
        @test decode(encode(text2, vocab2), vocab2) == text2
    end
end
