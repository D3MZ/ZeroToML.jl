glorot(m, n) = (rand(Float32, m, n) .- 0.5f0) .* sqrt(2.0f0 / (m + n))

# --- Tokenizer ---
build_vocab(text) = sort(unique(collect(text)))

function encode(text, vocab)
    char_to_int = Dict(c => i for (i, c) in enumerate(vocab))
    [char_to_int[c] for c in text]
end

function decode(encoded_text, vocab)
    join([vocab[i] for i in encoded_text])
end

# --- Positional Encoding ---
function positional_encoding(seq_len, embed_size)
    PE = zeros(Float32, seq_len, embed_size)
    pos = reshape(1:seq_len, seq_len, 1)
    div_term = exp.((0:2:embed_size-1) .* -(log(10000.0f0) / embed_size))'
    PE[:, 1:2:end] = sin.(pos * div_term)
    PE[:, 2:2:end] = cos.(pos * div_term)
    return Matrix(PE')
end

# --- Activation function ---
function softmax(x; dims=1)
    e_x = exp.(x .- maximum(x, dims=dims))
    return e_x ./ sum(e_x, dims=dims)
end


function parameters(vocab; dₑ=8, d_ff=16, max_seq_len=100)
    vocab_size = length(vocab)
    (
        E = glorot(dₑ, vocab_size),
        P = positional_encoding(max_seq_len, dₑ),
        W_Q = glorot(dₑ, dₑ),
        W_K = glorot(dₑ, dₑ),
        W_V = glorot(dₑ, dₑ),
        W_O = glorot(dₑ, dₑ),
        ln1_γ = ones(Float32, dₑ, 1),
        ln1_β = zeros(Float32, dₑ, 1),
        ln2_γ = ones(Float32, dₑ, 1),
        ln2_β = zeros(Float32, dₑ, 1),
        W₁ = glorot(d_ff, dₑ),
        b₁ = zeros(Float32, d_ff, 1),
        W₂ = glorot(dₑ, d_ff),
        b₂ = zeros(Float32, dₑ, 1),
        W_out = glorot(vocab_size, dₑ),
        b_out = zeros(Float32, vocab_size, 1),
    )
end

function layernorm(X, γ, β; ϵ=1f-5)
    μ  = mean(X; dims=1)
    σ2 = var(X; dims=1, corrected=false)
    X̂  = (X .- μ) ./ sqrt.(σ2 .+ ϵ)
    return X̂ .* γ .+ β
end

function forward(x, θ)
    L = length(x)
    X = θ.E[:, x] .+ θ.P[:, 1:L]

    T  = eltype(X)

    X₁ = layernorm(X, θ.ln1_γ, θ.ln1_β)
    Q  = θ.W_Q * X₁
    K  = θ.W_K * X₁
    V  = θ.W_V * X₁

    d_k   = size(K, 1)
    scale = inv(sqrt(T(d_k)))
    S     = (K' * Q) .* scale

    S = S .+ triu(fill(eltype(S)(-Inf), L, L), 1)

    Z  = θ.W_O * (V * softmax(S; dims=2)')
    X̃  = X .+ Z

    X₂ = layernorm(X̃, θ.ln2_γ, θ.ln2_β)
    H₁ = max.(θ.W₁ * X₂ .+ θ.b₁, T(0))
    H₂ = θ.W₂ * H₁ .+ θ.b₂
    X = X̃ .+ H₂

    logits = θ.W_out * X .+ θ.b_out
    return logits
end

function loss(θ, x, y)
    ŷ = forward(x, θ)
    max_ŷ = maximum(ŷ; dims=1)
    log_probs = ŷ .- max_ŷ .- log.(sum(exp.(ŷ .- max_ŷ); dims=1))
    correct_log_probs = log_probs[CartesianIndex.(y, eachindex(y))]
    -mean(correct_log_probs)
end

function train!(model, x, y, epochs, η)
    for _ in 1:epochs
        (∇,) = gradient(m -> loss(m, x, y), model)
        model = map((p, g) -> p .- η .* g, model, ∇)
    end
    return model
end

function generate(model, vocab, seed; n::Int=20)
    idx = encode(string(seed), vocab)
    for _ in 1:n
        logits = forward(idx, model)
        p = softmax(logits[:, end])
        push!(idx, sample(1:length(vocab), Weights(p)))
    end
    join(vocab[i] for i in idx)
end
