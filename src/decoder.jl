# --- Weight Initialization ---
glorot_init(m, n) = (rand(Float32, m, n) .- 0.5f0) .* sqrt(2.0f0 / (m + n))

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
    eₓ = exp.(x .- maximum(x, dims=dims))
    return eₓ ./ sum(eₓ, dims=dims)
end

# --- Network ---
@kwdef struct Decoder
    E = glorot_init(8, 29)
    P = positional_encoding(100, 8)
    Wₐ = glorot_init(8, 8)
    Wₖ = glorot_init(8, 8)
    Wᵥ = glorot_init(8, 8)
    Wₒ = glorot_init(8, 8)
    ln₁_γ = ones(Float32, 8, 1)
    ln₁_β = zeros(Float32, 8, 1)
    ln₂_γ = ones(Float32, 8, 1)
    ln₂_β = zeros(Float32, 8, 1)
    W₁ = glorot_init(16, 8)
    b₁ = zeros(Float32, 16, 1)
    W₂ = glorot_init(8, 16)
    b₂ = zeros(Float32, 8, 1)
    W_out = glorot_init(29, 8)
    b_out = zeros(Float32, 29, 1)
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

    X₁ = layernorm(X, θ.ln₁_γ, θ.ln₁_β)
    Q  = θ.Wₐ * X₁
    K  = θ.Wₖ * X₁
    V  = θ.Wᵥ * X₁

    dₖ    = size(K, 1)
    scale = inv(sqrt(T(dₖ)))
    S     = (K' * Q) .* scale #Attention score

    S = S .+ triu(fill(eltype(S)(-Inf), L, L), 1) #Casual mask

    Z  = θ.Wₒ * (V * softmax(S; dims=2)')
    X̃  = X .+ Z

    X₂ = layernorm(X̃, θ.ln₂_γ, θ.ln₂_β)
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

function step(model::Decoder, x, y, η)
    (∇,) = gradient(m -> loss(m, x, y), model)
    updated_params = (p => getproperty(model, p) .- η .* getproperty(∇, p) for p in propertynames(model))
    Decoder(;updated_params...)
end

# --- Training, Inference, and Helpers ---
dataloader(x, y, L) = zip(Iterators.partition(x, L), Iterators.partition(y, L))
train(model, x, y, L, η) = foldl(((m,(xb,yb))->ZeroToML.step(m, xb, yb, η)), dataloader(x, y, L); init = model)
train(model, x, y, L, η, epochs) = foldl((m, _) -> train(m, x, y, L, η),1:epochs;init=model)
param_count(model) = sum(length, values(model))

function generate(model, vocab, seed; n::Int=20)
    idx = encode(string(seed), vocab)
    for _ in 1:n
        logits = forward(idx, model)
        p = softmax(logits[:, end])
        push!(idx, sample(1:length(vocab), Weights(p)))
    end
    join(vocab[i] for i in idx)
end
