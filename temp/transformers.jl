using LinearAlgebra, Random, Statistics

# --- Data Preparation ---
input_text = repeat("AB", 1000)
chars = sort(unique(collect(input_text)))
vocab_size = length(chars)
stoi = Dict(c => i for (i, c) in enumerate(chars))
itos = Dict(i => c for (i, c) in enumerate(chars))
encode(s) = [stoi[c] for c in s]
decode(l) = join([itos[i] for i in l])

# --- Hyperparameters ---
embed_size = 32
seq_len = 8
num_heads = 4
num_layers = 2
ff_hidden_size = 4 * embed_size
learning_rate = 1e-3
max_iters = 1000

# --- Helper Functions ---
function softmax(x; dims=1)
    e_x = exp.(x .- maximum(x, dims=dims))
    return e_x ./ sum(e_x, dims=dims)
end

# --- Model Components ---

struct LayerNorm
    gamma::Vector{Float64}
    beta::Vector{Float64}
    epsilon::Float64

    function LayerNorm(embed_size::Int; epsilon=1e-5)
        gamma = ones(embed_size)
        beta = zeros(embed_size)
        new(gamma, beta, epsilon)
    end
end

function (ln::LayerNorm)(x)
    μ = mean(x, dims=1)
    σ² = var(x, dims=1, corrected=false)
    x_norm = (x .- μ) ./ sqrt.(σ² .+ ln.epsilon)
    return ln.gamma .* x_norm .+ ln.beta
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
    return V * p_attn
end

struct MultiHeadAttention
    embed_size::Int
    num_heads::Int
    head_dim::Int
    
    W_q::Matrix{Float64}
    W_k::Matrix{Float64}
    W_v::Matrix{Float64}
    W_o::Matrix{Float64}
    
    attention::ScaledDotProductAttention

    function MultiHeadAttention(embed_size::Int, num_heads::Int)
        @assert embed_size % num_heads == 0
        head_dim = embed_size ÷ num_heads
        
        limit = sqrt(3.0 / embed_size)
        W_q = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit
        W_k = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit
        W_v = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit
        W_o = rand(Float64, embed_size, embed_size) .* 2 .* limit .- limit
        
        new(embed_size, num_heads, head_dim, W_q, W_k, W_v, W_o, ScaledDotProductAttention())
    end
end

function (mha::MultiHeadAttention)(x; mask=nothing)
    seq_len = size(x, 2)
    
    Q = mha.W_q * x
    K = mha.W_k * x
    V = mha.W_v * x

    Q = permutedims(reshape(Q, mha.head_dim, mha.num_heads, seq_len), (1, 3, 2))
    K = permutedims(reshape(K, mha.head_dim, mha.num_heads, seq_len), (1, 3, 2))
    V = permutedims(reshape(V, mha.head_dim, mha.num_heads, seq_len), (1, 3, 2))
    
    attended_values = similar(V)
    for i in 1:mha.num_heads
        head_q = Q[:, :, i]
        head_k = K[:, :, i]
        head_v = V[:, :, i]
        attended_values[:, :, i] = mha.attention(head_q, head_k, head_v, mask=mask)
    end
    
    concatenated = reshape(permutedims(attended_values, (1, 3, 2)), mha.embed_size, seq_len)
    
    output = mha.W_o * concatenated
    return output
end

struct FeedForward
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}

    function FeedForward(embed_size::Int, hidden_size::Int)
        limit1 = sqrt(6.0 / (embed_size + hidden_size))
        W1 = rand(Float64, hidden_size, embed_size) .* 2 .* limit1 .- limit1
        b1 = zeros(hidden_size)
        
        limit2 = sqrt(6.0 / (hidden_size + embed_size))
        W2 = rand(Float64, embed_size, hidden_size) .* 2 .* limit2 .- limit2
        b2 = zeros(embed_size)
        
        new(W1, b1, W2, b2)
    end
end

function (ff::FeedForward)(x)
    hidden = max.(0, ff.W1 * x .+ ff.b1)
    return ff.W2 * hidden .+ ff.b2
end

function cross_entropy_loss(logits, targets)
    # logits: (vocab_size, seq_len)
    # targets: (seq_len,)
    _, seq_len = size(logits)
    probs = softmax(logits, dims=1)
    indices = CartesianIndex.(targets, 1:seq_len)
    p = probs[indices]
    loss = -mean(log.(p .+ 1e-9))
    return loss
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
    attn_output = block.mha(x, mask=mask)
    x = block.ln1(x + attn_output)
    ff_output = block.ff(x)
    x = block.ln2(x + ff_output)
    return x
end

# --- Positional Encoding ---
function positional_encoding(seq_len::Int, embed_size::Int)
    PE = zeros(embed_size, seq_len)
    pos = reshape(1:seq_len, seq_len, 1)
    div_term = exp.((0:2:embed_size-1) .* -(log(10000.0) / embed_size))'
    PE[1:2:end, :] = sin.(pos * div_term)'
    PE[2:2:end, :] = cos.(pos * div_term)'
    return PE
end

# --- Transformer Model ---
struct Transformer
    token_embedding::Matrix{Float64}
    pos_encoding::Matrix{Float64}
    blocks::Vector{TransformerBlock}
    ln_final::LayerNorm
    lm_head::Matrix{Float64}

    function Transformer(vocab_size::Int, embed_size::Int, seq_len::Int, num_heads::Int, num_layers::Int, ff_hidden_size::Int)
        token_embedding = randn(Float64, embed_size, vocab_size) .* 0.02
        pos_encoding = positional_encoding(seq_len, embed_size)
        blocks = [TransformerBlock(embed_size, num_heads, ff_hidden_size) for _ in 1:num_layers]
        ln_final = LayerNorm(embed_size)
        
        limit_head = sqrt(6.0 / (embed_size + vocab_size))
        lm_head = rand(Float64, vocab_size, embed_size) .* 2 .* limit_head .- limit_head
        
        new(token_embedding, pos_encoding, blocks, ln_final, lm_head)
    end
end

function (model::Transformer)(x_indices)
    current_seq_len = length(x_indices)
    
    x = model.token_embedding[:, x_indices]
    x = x .+ model.pos_encoding[:, 1:current_seq_len]
    
    mask = triu(fill(-Inf, current_seq_len, current_seq_len), 1)

    for block in model.blocks
        x = block(x; mask=mask)
    end
    
    x = model.ln_final(x)
    logits = model.lm_head * x
    
    return logits
end

# --- Training & Generation ---

function get_batch(data, seq_len)
    i = rand(1:(length(data) - seq_len))
    x = encode(data[i:i+seq_len-1])
    y = encode(data[i+1:i+seq_len])
    return x, y
end

function sample_multinomial(probs)
    r = rand()
    c = 0.0
    for (i, p) in enumerate(probs)
        c += p
        if r < c
            return i
        end
    end
    return length(probs) # fallback
end

function generate_text(model::Transformer, start_string::String, max_new_tokens::Int)
    tokens = encode(start_string)
    
    print("Generating text: ", start_string)
    
    for _ in 1:max_new_tokens
        context_tokens = length(tokens) > seq_len ? tokens[end-seq_len+1:end] : tokens
        
        logits = model(context_tokens)
        last_logits = logits[:, end]
        probs = softmax(last_logits)
        
        next_token = sample_multinomial(probs)
        
        push!(tokens, next_token)
        print(itos[next_token])
    end
    println()
end

# --- Main Execution ---
println("Initializing Transformer Model...")
model = Transformer(vocab_size, embed_size, seq_len, num_heads, num_layers, ff_hidden_size)

println("\nModel Architecture:")
println("  Vocab Size: $vocab_size")
println("  Embedding Size: $embed_size")
println("  Sequence Length: $seq_len")
println("  Num Heads: $num_heads")
println("  Num Layers: $num_layers")
println("  FF Hidden Size: $ff_hidden_size")
println("  Learning Rate: $learning_rate")
println("  Max Iters: $max_iters\n")

data = collect(input_text)

println("Starting training loop (without weight updates)...")
for iter in 1:max_iters
    x, y = get_batch(data, seq_len)
    logits = model(x)
    loss = cross_entropy_loss(logits, y)

    # In a real training loop, we would now perform backpropagation
    # to calculate gradients and then use an optimizer to update model weights.
    
    if iter % 100 == 0
        println("iter $iter, loss: $loss")
    end
end
println("Training loop finished.\n")


println("Running one forward pass with random weights...")
x, y = get_batch(data, seq_len)
logits = model(x)
loss = cross_entropy_loss(logits, y)
println("Input tokens: ", x)
println("Target tokens: ", y)
println("Output logits shape: ", size(logits))
println("Initial random loss: ", loss)
println()

generate_text(model, "A", 50)
