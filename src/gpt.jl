using StatsBase
using Random, Logging, LinearAlgebra, Statistics
# using Enzyme # Compile time not worth the speed boost for simple projects
using Zygote

# Tokenizer Functions
build_vocab(text) = sort(unique(collect(text)))

function encode(text, vocab)
    char_to_int = Dict(c => i for (i, c) in enumerate(vocab))
    [char_to_int[c] for c in text]
end

function decode(encoded_text, vocab)
    join([vocab[i] for i in encoded_text])
end

# Positional Encoding functions
function positional_encoding(seq_len::Int, embed_size::Int)
    PE = zeros(embed_size, seq_len)
    pos = reshape(1:seq_len, seq_len, 1)
    div_term = exp.((0:2:embed_size-1) .* -(log(10000.0) / embed_size))'
    PE[1:2:end, :] = sin.(pos * div_term)'
    PE[2:2:end, :] = cos.(pos * div_term)'
    return PE
end

# Weight initialization
glorot(m, n) = (rand(Float32, m, n) .- 0.5f0) .* sqrt(2.0f0 / (m + n))

# Activation
function softmax(M; dims=2)
    ex = exp.(M .- maximum(M; dims))
    ex ./ sum(ex; dims)
end

# LayerNorm (per-token over features)
function layernorm(X, γ, β; ϵ::Float32=1f-5)
    μ  = mean(X; dims=2)
    σ2 = var(X; dims=2, corrected=false)
    X̂  = (X .- μ) ./ sqrt.(σ2 .+ ϵ)
    return X̂ .* γ' .+ β'
end

# Attention
# causal_mask(L) = triu(fill(-Inf, L, L), 1)

# function attention(Q, K, V)
#     d_k = size(K, 2)
#     S = (Q * K') ./ sqrt(eltype(Q)(d_k))
#     L = size(Q, 1)
#     S .+= causal_mask(L)
#     return softmax(S; dims=2) * V
# end

# One pre-norm Transformer block (single-head for clarity)
function transformer_block(X, θ)
    # Self-attention sublayer (pre-norm + residual)
    X₁ = layernorm(X, θ[:ln1_γ], θ[:ln1_β])
    Q  = X₁ * θ[:W_Q]'
    K  = X₁ * θ[:W_K]'
    V = X₁ * θ[:W_V]'

    # Attention 
    d_k = size(K, 2)
    S = (Q * K') ./ sqrt(d_k)
    L = size(Q, 1)
    S = S .+ triu(fill(-Inf, L, L), 1) # causal mask
    Z = softmax(S; dims=2) * V * θ[:W_O]'

    X̃  = X .+ Z # Add residual (skip) connections

    # MLP sublayer (pre-norm + residual)
    X₂ = layernorm(X̃, θ[:ln2_γ], θ[:ln2_β])
    H₁ = max.(X₂ * θ[:W₁]' .+ θ[:b₁]', 0f0)   # L × d_ff
    H₂ = H₁ * θ[:W₂]' .+ θ[:b₂]'              # L × dₑ
    return X̃ .+ H₂                            # L × dₑ
end

# End-to-end single-layer decoder forward (emb + block + head)
function forward(x, θ)
    L = length(x)
    X = θ[:E][x, :] .+ θ[:P][1:L, :]      # now both are (L×dₑ)
    X = transformer_block(X, θ)
    logits = X * θ[:W_out]' .+ θ[:b_out]'
    return logits
end

text = "ABABAABBAAABBB"
vocab = build_vocab(text)
encode(text, vocab)


# --------------------
# Hyper‑parameters
# --------------------
dₑ      = 8      # embedding dimension
d_ff    = 16     # feed‑forward hidden dimension
h       = 1      # number of heads (kept = 1 for clarity)
η       = 1f-2   # learning rate
epochs  = 500

# --------------------
# Parameter initialisation
# --------------------

θ = Dict{Symbol, Any}(
    :E     => glorot(length(vocab), dₑ),      # token embeddings
    :P     => glorot(length(text), dₑ),       # position embeddings
    :W_Q   => glorot(dₑ, dₑ),
    :W_K   => glorot(dₑ, dₑ),
    :W_V   => glorot(dₑ, dₑ),
    :W_O   => glorot(dₑ, dₑ),
    :ln1_γ => ones(Float32, dₑ),
    :ln1_β => zeros(Float32, dₑ),
    :ln2_γ => ones(Float32, dₑ),
    :ln2_β => zeros(Float32, dₑ),
    :W₁    => glorot(d_ff, dₑ),
    :b₁    => zeros(Float32, d_ff),
    :W₂    => glorot(dₑ, d_ff),
    :b₂    => zeros(Float32, dₑ),
    :W_out => glorot(length(vocab), dₑ),
    :b_out => zeros(Float32, length(vocab)),
)

# --------------------
# Core operations
# --------------------

x = encode(text[1:end-1], vocab)
y = encode(text[2:end], vocab)

function loss(θ, x, y)
    ℓ = 0f0
    ŷ = forward(x, θ)
    for i in eachindex(y)
        p = softmax(ŷ[i:i, :])[1, :]
        ℓ -= log(p[y[i]])
    end
    ℓ / length(y)
end


# Optimisation loop

for epoch in 1:epochs
    ℓ, (∇θ,) = Zygote.withgradient(loss, θ, x, y)
    for (k, v) in θ
        θ[k] = v .- η .* ∇θ[k]
    end
    epoch % 50 == 0 && @info "epoch=$epoch loss=$(round(ℓ; digits = 4))"
end

# --------------------
# Sampling
# --------------------
# function generate(seed::Char, n::Int)
#     idx = [vocab_idx[seed]]
#     for _ in 1:n
#         logits = forward(idx, θ)
#         p = softmax(logits[end:end, :])[1, :]
#         push!(idx, sample(1:length(vocab), Weights(p)))
#     end
#     join(vocab[i] for i in idx)
# end

# @info "Sample: $(generate('A', 20))"
