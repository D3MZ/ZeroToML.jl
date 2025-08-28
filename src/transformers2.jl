using StatsBase
using Random, Logging, LinearAlgebra, Statistics, Enzyme

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
glorot(m, n) = (rand(Float64, m, n) .- 0.5) .* √(2.0 / (m + n))

# Activation
function softmax(M)
    ex = exp.(M .- maximum(M; dims = 2))
    ex ./ sum(ex; dims = 2)
end

# LayerNorm (per-token over features)
function layernorm(X, γ, β; ϵ::Float32=1f-5)
    μ  = mean(X; dims=2)
    σ2 = var(X; dims=2, corrected=false)
    X̂  = (X .- μ) ./ sqrt.(σ2 .+ ϵ)
    return X̂ .* γ' .+ β'
end

# Attention
causal_mask(L) = tril(ones(Float32, L, L))

function attention(Q, K, V; mask::Union{Function,Nothing}=nothing)
    S = (Q * K') ./ √(size(K, 2))

    if mask !== nothing
        M = mask(size(S, 1))
        S = ifelse.(M .== 1, S, oftype(S, -Inf))
    end

    return softmax(S) * V
end

# One pre-norm Transformer block (single-head for clarity)
function transformer_block(X, θ; mask::Union{Function,Nothing}=nothing)
    # Self-attention sublayer (pre-norm + residual)
    X₁ = layernorm(X, θ[:ln1_γ], θ[:ln1_β])
    Q  = X₁ * θ[:W_Q]'
    K  = X₁ * θ[:W_K]'
    Vh = X₁ * θ[:W_V]'
    Z  = attention(Q, K, Vh; mask=mask) * θ[:W_O]'
    X̃  = X .+ Z

    # MLP sublayer (pre-norm + residual)
    X₂ = layernorm(X̃, θ[:ln2_γ], θ[:ln2_β])
    H₁ = max.(X₂ * θ[:W₁]' .+ θ[:b₁]', 0f0)   # L × d_ff
    H₂ = H₁ * θ[:W₂]' .+ θ[:b₂]'              # L × dₑ
    return X̃ .+ H₂                            # L × dₑ
end

# End-to-end single-layer decoder forward (emb + block + head)
function decoder_forward(x, Θ)
    L  = length(x)
    X  = Θ[:E][x, :] .+ Θ[:P][1:L, :]
    X  = transformer_block(X, Θ; mask=causal_mask)
    logits = X * Θ[:W_out]' .+ Θ[:b_out]'      # (L × |vocab|)
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
η       = 1e-2   # ADAM learning rate
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

const x_train = encode(text[1:end-1], vocab)
const y_train = encode(text[2:end], vocab)

function loss(θ)
    ŷ = decoder_forward(x_train, θ)
    ℓ = 0f0
    for i in eachindex(y_train)
        p = softmax(ŷ[i:i, :])[1, :]
        ℓ -= log(p[y_train[i]])
    end
    ℓ / length(y_train)
end

# --------------------
# Optimisation loop
# --------------------
for epoch in 1:epochs
    # compute scalar loss
    ℓ = loss(θ)

    # Enzyme reverse-mode gradient for each parameter array
    for (k, v) in θ
        if v isa AbstractArray{<:Real}
            f = p -> begin
                θ[k] = p
                loss(θ)
            end
            ∇v = Enzyme.gradient(Enzyme.Reverse, f, Enzyme.Active(v))[1]
            θ[k] .= v .- η .* ∇v
        end
    end

    epoch % 50 == 0 && @info "epoch=$epoch loss=$(round(ℓ; digits = 4))"
end

# --------------------
# Sampling
# --------------------
function generate(seed::Char, n::Int)
    idx = [vocab_idx[seed]]
    for _ in 1:n
        logits = decoder_forward(idx, θ)
        p = softmax(logits[end:end, :])[1, :]
        push!(idx, sample(1:length(vocab), Weights(p)))
    end
    join(vocab[i] for i in idx)
end

@info "Sample: $(generate('A', 20))"
