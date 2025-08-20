# --- Model Components ---

mutable struct LayerNorm
    gamma::Vector{Float64}
    beta::Vector{Float64}
    epsilon::Float64
    ∇gamma::Vector{Float64}
    ∇beta::Vector{Float64}

    m_gamma::Vector{Float64}
    v_gamma::Vector{Float64}
    m_beta::Vector{Float64}
    v_beta::Vector{Float64}

    function LayerNorm(embed_size::Int; epsilon=1e-5)
        gamma = ones(embed_size)
        beta = zeros(embed_size)
        ∇gamma = zeros(embed_size)
        ∇beta = zeros(embed_size)
        m_gamma = zeros(embed_size)
        v_gamma = zeros(embed_size)
        m_beta = zeros(embed_size)
        v_beta = zeros(embed_size)
        new(gamma, beta, epsilon, ∇gamma, ∇beta, m_gamma, v_gamma, m_beta, v_beta)
    end
end

function (ln::LayerNorm)(x)
    μ = mean(x, dims=1)
    σ² = var(x, dims=1, corrected=false)
    x_norm = (x .- μ) ./ sqrt.(σ² .+ ln.epsilon)
    out = ln.gamma .* x_norm .+ ln.beta
    cache = (x, x_norm, μ, σ²)
    return out, cache
end

struct ScaledDotProductAttention
end

function (sdpa::ScaledDotProductAttention)(Q, K, V; mask=nothing)
    d_k = size(Q, 1)
    scores = (K' * Q) ./ sqrt(d_k)
    
    if mask !== nothing
        scores = scores .+ mask
    end
    
    p_attn = softmax(scores, dims=1)
    out = V * p_attn
    cache = (Q, K, V, p_attn)
    return out, cache
end

mutable struct MultiHeadAttention
    embed_size::Int
    num_heads::Int
    head_dim::Int
    
    W_q::Matrix{Float64}
    W_k::Matrix{Float64}
    W_v::Matrix{Float64}
    W_o::Matrix{Float64}
    
    ∇W_q::Matrix{Float64}
    ∇W_k::Matrix{Float64}
    ∇W_v::Matrix{Float64}
    ∇W_o::Matrix{Float64}
    
    m_W_q::Matrix{Float64}
    v_W_q::Matrix{Float64}
    m_W_k::Matrix{Float64}
    v_W_k::Matrix{Float64}
    m_W_v::Matrix{Float64}
    v_W_v::Matrix{Float64}
    m_W_o::Matrix{Float64}
    v_W_o::Matrix{Float64}

    attention::ScaledDotProductAttention

    function MultiHeadAttention(embed_size::Int, num_heads::Int)
        @assert embed_size % num_heads == 0
        head_dim = embed_size ÷ num_heads
        
        limit = sqrt(3.0 / embed_size)
        W_q = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit
        W_k = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit
        W_v = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit
        W_o = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit

        ∇W_q = zeros(size(W_q))
        ∇W_k = zeros(size(W_k))
        ∇W_v = zeros(size(W_v))
        ∇W_o = zeros(size(W_o))

        m_W_q = zeros(size(W_q)); v_W_q = zeros(size(W_q))
        m_W_k = zeros(size(W_k)); v_W_k = zeros(size(W_k))
        m_W_v = zeros(size(W_v)); v_W_v = zeros(size(W_v))
        m_W_o = zeros(size(W_o)); v_W_o = zeros(size(W_o))
        
        new(embed_size, num_heads, head_dim, W_q, W_k, W_v, W_o, ∇W_q, ∇W_k, ∇W_v, ∇W_o, 
            m_W_q, v_W_q, m_W_k, v_W_k, m_W_v, v_W_v, m_W_o, v_W_o, ScaledDotProductAttention())
    end
end

function (mha::MultiHeadAttention)(x; mask=nothing)
    seq_len = size(x, 2)
    
    Q_proj = mha.W_q * x
    K_proj = mha.W_k * x
    V_proj = mha.W_v * x

    Q = permutedims(reshape(Q_proj, mha.head_dim, mha.num_heads, seq_len), (1, 3, 2))
    K = permutedims(reshape(K_proj, mha.head_dim, mha.num_heads, seq_len), (1, 3, 2))
    V = permutedims(reshape(V_proj, mha.head_dim, mha.num_heads, seq_len), (1, 3, 2))
    
    attended_values = similar(V)
    sdpa_caches = Vector{Any}(undef, mha.num_heads)
    for i in 1:mha.num_heads
        head_q = Q[:, :, i]
        head_k = K[:, :, i]
        head_v = V[:, :, i]
        attended_values[:, :, i], sdpa_caches[i] = mha.attention(head_q, head_k, head_v, mask=mask)
    end
    
    concatenated = reshape(permutedims(attended_values, (1, 3, 2)), mha.embed_size, seq_len)
    
    output = mha.W_o * concatenated
    
    cache = (x, Q_proj, K_proj, V_proj, Q, K, V, attended_values, concatenated, sdpa_caches)
    return output, cache
end

mutable struct FeedForward
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    ∇W1::Matrix{Float64}
    ∇b1::Vector{Float64}
    ∇W2::Matrix{Float64}
    ∇b2::Vector{Float64}

    m_W1::Matrix{Float64}
    v_W1::Matrix{Float64}
    m_b1::Vector{Float64}
    v_b1::Vector{Float64}
    m_W2::Matrix{Float64}
    v_W2::Matrix{Float64}
    m_b2::Vector{Float64}
    v_b2::Vector{Float64}

    function FeedForward(embed_size::Int, hidden_size::Int)
        limit1 = sqrt(6.0 / (embed_size + hidden_size))
        W1 = rand(Float64, hidden_size, embed_size) .* 2 .* limit1 .- limit1
        b1 = zeros(hidden_size)
        
        limit2 = sqrt(6.0 / (hidden_size + embed_size))
        W2 = rand(Float64, embed_size, hidden_size) .* 2 .* limit2 .- limit2
        b2 = zeros(embed_size)
        
        ∇W1 = zeros(size(W1)); ∇b1 = zeros(size(b1))
        ∇W2 = zeros(size(W2)); ∇b2 = zeros(size(b2))
        m_W1 = zeros(size(W1)); v_W1 = zeros(size(W1))
        m_b1 = zeros(size(b1)); v_b1 = zeros(size(b1))
        m_W2 = zeros(size(W2)); v_W2 = zeros(size(W2))
        m_b2 = zeros(size(b2)); v_b2 = zeros(size(b2))

        new(W1, b1, W2, b2, ∇W1, ∇b1, ∇W2, ∇b2, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2)
    end
end

function (ff::FeedForward)(x)
    pre_activation = ff.W1 * x .+ ff.b1
    hidden = max.(0, pre_activation)
    out = ff.W2 * hidden .+ ff.b2
    cache = (x, pre_activation, hidden)
    return out, cache
end

function cross_entropy_loss(logits, targets)
    # logits: (vocab_size, seq_len)
    # targets: (seq_len,)
    probs = softmax(logits, dims=1)
    indices = CartesianIndex.(targets, eachindex(targets))
    p = probs[indices]
    loss = -mean(log.(p .+ 1e-9))
    return loss
end

function cross_entropy_loss_backward(logits, targets)
    probs = softmax(logits, dims=1)
    
    dlogits = copy(probs)
    indices = CartesianIndex.(targets, eachindex(targets))
    dlogits[indices] .-= 1.0
    
    dlogits ./= length(targets)
    
    return dlogits
end

struct TransformerBlock
    mha::MultiHeadAttention
    ln1::LayerNorm
    ff::FeedForward
    ln2::LayerNorm

    function TransformerBlock(embed_size::Int, num_heads::Int, ff_hidden_size::Int)
        mha = MultiHeadAttention(embed_size, num_heads)
        ln1 = LayerNorm(embed_size)
        ff = FeedForward(embed_size, ff_hidden_size)
        ln2 = LayerNorm(embed_size)
        new(mha, ln1, ff, ln2)
    end
end

function (block::TransformerBlock)(x; mask=nothing)
    attn_output, mha_cache = block.mha(x, mask=mask)
    ln1_input = x + attn_output
    x_norm1, ln1_cache = block.ln1(ln1_input)

    ff_output, ff_cache = block.ff(x_norm1)
    ln2_input = x_norm1 + ff_output
    x_norm2, ln2_cache = block.ln2(ln2_input)

    cache = (mha_cache, ln1_cache, x_norm1, ff_cache, ln2_cache)
    return x_norm2, cache
end

# --- Transformer Model ---
mutable struct Transformer
    token_embedding::Matrix{Float64}
    pos_encoding::Matrix{Float64}
    blocks::Vector{TransformerBlock}
    ln_final::LayerNorm
    lm_head::Matrix{Float64}

    ∇token_embedding::Matrix{Float64}
    ∇lm_head::Matrix{Float64}

    m_token_embedding::Matrix{Float64}
    v_token_embedding::Matrix{Float64}
    m_lm_head::Matrix{Float64}
    v_lm_head::Matrix{Float64}

    function Transformer(vocab_size::Int, embed_size::Int, seq_len::Int, num_heads::Int, num_layers::Int, ff_hidden_size::Int)
        token_embedding = randn(Float64, embed_size, vocab_size) .* 0.02
        pos_encoding = positional_encoding(seq_len, embed_size)
        blocks = [TransformerBlock(embed_size, num_heads, ff_hidden_size) for _ in 1:num_layers]
        ln_final = LayerNorm(embed_size)
        
        limit_head = sqrt(6.0 / (embed_size + vocab_size))
        lm_head = rand(Float64, vocab_size, embed_size) .* 2 .* limit_head .- limit_head
        
        ∇token_embedding = zeros(size(token_embedding))
        ∇lm_head = zeros(size(lm_head))

        m_token_embedding = zeros(size(token_embedding))
        v_token_embedding = zeros(size(token_embedding))
        m_lm_head = zeros(size(lm_head))
        v_lm_head = zeros(size(lm_head))
        
        new(token_embedding, pos_encoding, blocks, ln_final, lm_head, ∇token_embedding, ∇lm_head, m_token_embedding, v_token_embedding, m_lm_head, v_lm_head)
    end
end

function (model::Transformer)(x_indices; start_pos=1)
    current_seq_len = length(x_indices)
    
    x_emb = model.token_embedding[:, x_indices]
    pos_indices = start_pos:(start_pos + current_seq_len - 1)
    x = x_emb .+ model.pos_encoding[:, pos_indices]
    
    mask = triu(fill(-Inf, current_seq_len, current_seq_len), 1)'

    block_outputs = Vector{Any}(undef, length(model.blocks))
    block_caches = Vector{Any}(undef, length(model.blocks))
    block_input = x
    for (i, block) in enumerate(model.blocks)
        block_outputs[i], block_caches[i] = block(block_input, mask=mask)
        block_input = block_outputs[i]
    end
    
    ln_final_output, ln_final_cache = model.ln_final(block_outputs[end])
    logits = model.lm_head * ln_final_output
    
    cache = (x_indices, x_emb, x, block_outputs, block_caches, ln_final_output, ln_final_cache)
    return logits, cache
end

function generate(model::Transformer, idx, max_new_tokens; greedy::Bool=true, start_pos=1)
    original_pos_encoding = model.pos_encoding
    embed_size, original_seq_len = size(original_pos_encoding)
    max_len_needed = start_pos + length(idx) + max_new_tokens - 1
    if max_len_needed > original_seq_len
        model.pos_encoding = positional_encoding(max_len_needed, embed_size)
    end

    for _ in 1:max_new_tokens
        logits, _ = model(idx; start_pos=start_pos)
        logits = logits[:, end]
        probs = softmax(logits)

        if greedy
            idx_next = argmax(probs)
        else
            u = rand()
            cdf = cumsum(vec(probs))
            idx_next = findfirst(>=(u), cdf)
            if isnothing(idx_next)
                idx_next = length(probs)
            end
        end
        idx = vcat(idx, idx_next)
    end

    model.pos_encoding = original_pos_encoding
    return idx
end

# --- Backpropagation ---

function backward!(ff::FeedForward, d_out, cache)
    x, pre_activation, hidden = cache
    
    ff.∇W2 .+= d_out * hidden'
    ff.∇b2 .+= sum(d_out, dims=2)[:]
    
    d_hidden = ff.W2' * d_out
    d_pre_activation = d_hidden .* (pre_activation .> 0)
    
    ff.∇W1 .+= d_pre_activation * x'
    ff.∇b1 .+= sum(d_pre_activation, dims=2)[:]
    
    d_x = ff.W1' * d_pre_activation
    return d_x
end

function backward!(ln::LayerNorm, d_out, cache)
    x, x_norm, mu, sigma_sq = cache
    embed_size, _ = size(x)
    
    d_x_norm = d_out .* ln.gamma
    d_sigma_sq = sum(d_x_norm .* (x .- mu) .* (-0.5) .* (sigma_sq .+ ln.epsilon).^(-1.5), dims=1)
    
    d_mu = sum(d_x_norm .* (-1 ./ sqrt.(sigma_sq .+ ln.epsilon)), dims=1) .+ 
           d_sigma_sq .* sum(-2 .* (x .- mu), dims=1) ./ embed_size

    d_x = (d_x_norm ./ sqrt.(sigma_sq .+ ln.epsilon)) .+ 
          (d_sigma_sq .* 2 .* (x .- mu) ./ embed_size) .+ 
          (d_mu ./ embed_size)
          
    ln.∇gamma .+= sum(d_out .* x_norm, dims=2)[:]
    ln.∇beta .+= sum(d_out, dims=2)[:]
    
    return d_x
end

function backward(sdpa::ScaledDotProductAttention, d_out, cache)
    Q, K, V, p_attn = cache
    d_k = size(Q, 1)

    d_V = d_out * p_attn'
    d_p_attn = V' * d_out
    
    d_scores = p_attn .* d_p_attn - p_attn .* sum(p_attn .* d_p_attn, dims=1)

    d_scores ./= sqrt(d_k)
    
    d_Q = K * d_scores
    d_K = Q * d_scores'
    
    return d_Q, d_K, d_V
end

function backward!(mha::MultiHeadAttention, d_out, cache)
    x, Q_proj, K_proj, V_proj, Q, K, V, attended_values, concatenated, sdpa_caches = cache
    seq_len = size(x, 2)
    
    mha.∇W_o .+= d_out * concatenated'
    d_concatenated = mha.W_o' * d_out
    
    d_attended_values = permutedims(reshape(d_concatenated, mha.head_dim, mha.num_heads, seq_len), (1, 3, 2))
    
    d_Q = zeros(size(Q)); d_K = zeros(size(K)); d_V = zeros(size(V))
    
    for i in 1:mha.num_heads
        sdpa_cache = sdpa_caches[i]
        d_q_head, d_k_head, d_v_head = backward(mha.attention, d_attended_values[:,:,i], sdpa_cache)
        d_Q[:,:,i] = d_q_head; d_K[:,:,i] = d_k_head; d_V[:,:,i] = d_v_head
    end
    
    d_Q_proj = reshape(permutedims(d_Q, (1, 3, 2)), mha.embed_size, seq_len)
    d_K_proj = reshape(permutedims(d_K, (1, 3, 2)), mha.embed_size, seq_len)
    d_V_proj = reshape(permutedims(d_V, (1, 3, 2)), mha.embed_size, seq_len)
    
    mha.∇W_q .+= d_Q_proj * x'
    mha.∇W_k .+= d_K_proj * x'
    mha.∇W_v .+= d_V_proj * x'
    
    d_x = mha.W_q' * d_Q_proj + mha.W_k' * d_K_proj + mha.W_v' * d_V_proj
    return d_x
end

function backward!(block::TransformerBlock, d_out, cache)
    mha_cache, ln1_cache, x_norm1, ff_cache, ln2_cache = cache

    d_ln2_input = backward!(block.ln2, d_out, ln2_cache)
    d_x_norm1 = d_ln2_input
    d_ff_output = d_ln2_input

    d_x_norm1 += backward!(block.ff, d_ff_output, ff_cache)
    
    d_ln1_input = backward!(block.ln1, d_x_norm1, ln1_cache)
    d_x = d_ln1_input
    d_attn_output = d_ln1_input
    
    d_x += backward!(block.mha, d_attn_output, mha_cache)

    return d_x
end

function backward!(model::Transformer, dlogits, cache)
    x_indices, x_emb, x, block_outputs, block_caches, ln_final_output, ln_final_cache = cache
    current_seq_len = length(x_indices)

    model.∇lm_head .+= dlogits * ln_final_output'
    d_ln_final_output = model.lm_head' * dlogits
    
    d_block_output = backward!(model.ln_final, d_ln_final_output, ln_final_cache)
    
    for i in length(model.blocks):-1:1
        d_block_output = backward!(model.blocks[i], d_block_output, block_caches[i])
    end
    
    d_x = d_block_output
    d_x_emb = d_x 
    
    for t in eachindex(x_indices)
        token_idx = x_indices[t]
        model.∇token_embedding[:, token_idx] .+= d_x_emb[:, t]
    end
end

# --- Optimizer ---

function zero_gradients!(ff::FeedForward)
    fill!(ff.∇W1, 0); fill!(ff.∇b1, 0); fill!(ff.∇W2, 0); fill!(ff.∇b2, 0)
end
function update!(ff::FeedForward, optimizer::Adam)
    for p in [:W1, :b1, :W2, :b2]
        param = getfield(ff, p)
        grad = getfield(ff, Symbol("∇", p))
        m = getfield(ff, Symbol("m_", p))
        v = getfield(ff, Symbol("v_", p))

        @. m = optimizer.beta1 * m + (1 - optimizer.beta1) * grad
        @. v = optimizer.beta2 * v + (1 - optimizer.beta2) * (grad .^ 2)
        m_hat = m / (1 - optimizer.beta1^optimizer.t)
        v_hat = v / (1 - optimizer.beta2^optimizer.t)
        @. param -= optimizer.lr * m_hat / (sqrt.(v_hat) + optimizer.epsilon)
    end
end

function zero_gradients!(ln::LayerNorm)
    fill!(ln.∇gamma, 0); fill!(ln.∇beta, 0)
end
function update!(ln::LayerNorm, optimizer::Adam)
    for p in [:gamma, :beta]
        param = getfield(ln, p)
        grad = getfield(ln, Symbol("∇", p))
        m = getfield(ln, Symbol("m_", p))
        v = getfield(ln, Symbol("v_", p))

        @. m = optimizer.beta1 * m + (1 - optimizer.beta1) * grad
        @. v = optimizer.beta2 * v + (1 - optimizer.beta2) * (grad .^ 2)
        m_hat = m / (1 - optimizer.beta1^optimizer.t)
        v_hat = v / (1 - optimizer.beta2^optimizer.t)
        @. param -= optimizer.lr * m_hat / (sqrt.(v_hat) + optimizer.epsilon)
    end
end

function zero_gradients!(mha::MultiHeadAttention)
    fill!(mha.∇W_q, 0); fill!(mha.∇W_k, 0); fill!(mha.∇W_v, 0); fill!(mha.∇W_o, 0)
end

function update!(mha::MultiHeadAttention, optimizer::Adam)
    for p in [:W_q, :W_k, :W_v, :W_o]
        param = getfield(mha, p)
        grad = getfield(mha, Symbol("∇", p))
        m = getfield(mha, Symbol("m_", p))
        v = getfield(mha, Symbol("v_", p))

        @. m = optimizer.beta1 * m + (1 - optimizer.beta1) * grad
        @. v = optimizer.beta2 * v + (1 - optimizer.beta2) * (grad .^ 2)
        m_hat = m / (1 - optimizer.beta1^optimizer.t)
        v_hat = v / (1 - optimizer.beta2^optimizer.t)
        @. param -= optimizer.lr * m_hat / (sqrt.(v_hat) + optimizer.epsilon)
    end
end

function zero_gradients!(block::TransformerBlock)
    zero_gradients!(block.mha); zero_gradients!(block.ln1); zero_gradients!(block.ff); zero_gradients!(block.ln2)
end
function update!(block::TransformerBlock, optimizer::Adam)
    update!(block.mha, optimizer); update!(block.ln1, optimizer); update!(block.ff, optimizer); update!(block.ln2, optimizer)
end

function zero_gradients!(model::Transformer)
    fill!(model.∇token_embedding, 0); fill!(model.∇lm_head, 0)
    for block in model.blocks
        zero_gradients!(block)
    end
    zero_gradients!(model.ln_final)
end

function update!(model::Transformer, optimizer::Adam)
    optimizer.t += 1
    
    for p in [:token_embedding, :lm_head]
        param = getfield(model, p)
        grad = getfield(model, Symbol("∇", p))
        m = getfield(model, Symbol("m_", p))
        v = getfield(model, Symbol("v_", p))

        @. m = optimizer.beta1 * m + (1 - optimizer.beta1) * grad
        @. v = optimizer.beta2 * v + (1 - optimizer.beta2) * (grad .^ 2)
        m_hat = m / (1 - optimizer.beta1^optimizer.t)
        v_hat = v / (1 - optimizer.beta2^optimizer.t)
        @. param -= optimizer.lr * m_hat / (sqrt.(v_hat) + optimizer.epsilon)
    end

    for block in model.blocks
        update!(block, optimizer)
    end
    update!(model.ln_final, optimizer)
end
