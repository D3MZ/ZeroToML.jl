# Generic transformer utilities shared by text decoders and ViT-style models.
# Tensors are stored as (embedding_dim, sequence_length), matching decoder.jl.

"Scaled dot-product attention over sequence matrices (D×L).

`allowed` is an optional L×L Boolean matrix where rows are key positions and
columns are query positions.  `false` entries are masked out before softmax."
function scaled_dot_product_attention(Q, K, V; allowed = nothing)
    T = eltype(Q)
    dₖ = size(K, 1)
    scores = (K' * Q) .* inv(sqrt(T(dₖ)))

    if allowed !== nothing
        scores = ifelse.(allowed, scores, T(-1f9))
    end

    weights = softmax(scores; dims = 1)
    V * weights
end

"Causal key/query mask for autoregressive attention."
causal_mask(L) = [key <= query for key in 1:L, query in 1:L]

@kwdef struct TransformerEncoder
    embed_dim = 8
    hidden_dim = 16
    max_len = 100
    P = positional_encoding(100, 8)

    Wq = glorot(8, 8; gain = inv(sqrt(3f0)))
    Wk = glorot(8, 8; gain = inv(sqrt(3f0)))
    Wv = glorot(8, 8; gain = inv(sqrt(3f0)))
    Wo = glorot(8, 8; gain = inv(sqrt(3f0)))

    ln₁_γ = ones(Float32, 8, 1)
    ln₁_β = zeros(Float32, 8, 1)
    ln₂_γ = ones(Float32, 8, 1)
    ln₂_β = zeros(Float32, 8, 1)

    W₁ = glorot(16, 8; gain = inv(sqrt(3f0)))
    b₁ = zeros(Float32, 16, 1)
    W₂ = glorot(8, 16; gain = inv(sqrt(3f0)))
    b₂ = zeros(Float32, 8, 1)
end

"Bidirectional transformer encoder block for generic sequence embeddings."
function forward(enc::TransformerEncoder, X, sequence = 1:size(X, 2); allowed = nothing)
    L = size(X, 2)
    T = eltype(X)
    H = X .+ enc.P[:, sequence]

    H₁ = layernorm(H, enc.ln₁_γ, enc.ln₁_β)
    Q = enc.Wq * H₁
    K = enc.Wk * H₁
    V = enc.Wv * H₁
    Z = enc.Wo * scaled_dot_product_attention(Q, K, V; allowed = allowed)
    H̃ = H .+ Z

    H₂ = layernorm(H̃, enc.ln₂_γ, enc.ln₂_β)
    F₁ = relu(enc.W₁ * H₂ .+ enc.b₁)
    F₂ = enc.W₂ * F₁ .+ enc.b₂
    H̃ .+ F₂
end
