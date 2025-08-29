using StatsBase, Random, Logging, LinearAlgebra, Statistics, Zygote

build_vocab(text) = sort(unique(collect(text)))

function encode(text, vocab)
    char_to_int = Dict(c => i for (i, c) in enumerate(vocab))
    [char_to_int[c] for c in text]
end

function decode(encoded_text, vocab)
    join([vocab[i] for i in encoded_text])
end

function positional_encoding(seq_len, embed_size)
    PE = zeros(Float32, seq_len, embed_size)
    pos = reshape(1:seq_len, seq_len, 1)
    div_term = exp.((0:2:embed_size-1) .* -(log(10000.0f0) / embed_size))'
    PE[:, 1:2:end] = sin.(pos * div_term)
    PE[:, 2:2:end] = cos.(pos * div_term)
    return PE
end

@kwdef mutable struct Parameters{T}
    E::Matrix{T}
    P::Matrix{T}
    W_Q::Matrix{T}
    W_K::Matrix{T}
    W_V::Matrix{T}
    W_O::Matrix{T}
    ln1_γ::Vector{T}
    ln1_β::Vector{T}
    ln2_γ::Vector{T}
    ln2_β::Vector{T}
    W₁::Matrix{T}
    b₁::Vector{T}
    W₂::Matrix{T}
    b₂::Vector{T}
    W_out::Matrix{T}
    b_out::Vector{T}
end

function Parameters(vocab; dₑ=8, d_ff=16, max_seq_len=100)
    vocab_size = length(vocab)
    Parameters(
        E = glorot(vocab_size, dₑ),
        P = positional_encoding(max_seq_len, dₑ),
        W_Q = glorot(dₑ, dₑ),
        W_K = glorot(dₑ, dₑ),
        W_V = glorot(dₑ, dₑ),
        W_O = glorot(dₑ, dₑ),
        ln1_γ = ones(Float32, dₑ),
        ln1_β = zeros(Float32, dₑ),
        ln2_γ = ones(Float32, dₑ),
        ln2_β = zeros(Float32, dₑ),
        W₁ = glorot(d_ff, dₑ),
        b₁ = zeros(Float32, d_ff),
        W₂ = glorot(dₑ, d_ff),
        b₂ = zeros(Float32, dₑ),
        W_out = glorot(vocab_size, dₑ),
        b_out = zeros(Float32, vocab_size),
    )
end

glorot(m, n) = (rand(Float32, m, n) .- 0.5f0) .* sqrt(2.0f0 / (m + n))

function softmax(M; dims=2)
    ex = exp.(M .- maximum(M; dims))
    ex ./ sum(ex; dims)
end

function layernorm(X, γ, β; ϵ=1f-5)
    μ  = mean(X; dims=2)
    σ2 = var(X; dims=2, corrected=false)
    X̂  = (X .- μ) ./ sqrt.(σ2 .+ ϵ)
    return X̂ .* γ' .+ β'
end

function transformer_block(X, θ::Parameters)
    X₁ = layernorm(X, θ.ln1_γ, θ.ln1_β)
    Q  = X₁ * θ.W_Q'
    K  = X₁ * θ.W_K'
    V = X₁ * θ.W_V'

    d_k = size(K, 2)
    S = (Q * K') ./ sqrt(d_k)
    L = size(Q, 1)
    S = S .+ triu(fill(-Inf, L, L), 1)
    Z = softmax(S; dims=2) * V * θ.W_O'

    X̃  = X .+ Z

    X₂ = layernorm(X̃, θ.ln2_γ, θ.ln2_β)
    H₁ = max.(X₂ * θ.W₁' .+ θ.b₁', 0f0)
    H₂ = H₁ * θ.W₂' .+ θ.b₂'
    return X̃ .+ H₂
end

function forward(x, θ::Parameters)
    L = length(x)
    X = θ.E[x, :] .+ θ.P[1:L, :]
    X = transformer_block(X, θ)
    logits = X * θ.W_out' .+ θ.b_out'
    return logits
end

text = "ABABAABBAAABBB"
vocab = build_vocab(text)
vocab_idx = Dict(c => i for (i, c) in enumerate(vocab))

η       = 1f-2
epochs  = 500

model = Parameters(vocab)

x = encode(text[1:end-1], vocab)
y = encode(text[2:end], vocab)

function loss(θ::Parameters, x, y)
    ŷ = forward(x, θ)
    max_ŷ = maximum(ŷ; dims=2)
    log_probs = ŷ .- max_ŷ .- log.(sum(exp.(ŷ .- max_ŷ); dims=2))
    correct_log_probs = log_probs[CartesianIndex.(eachindex(y), y)]
    -mean(correct_log_probs)
end

function update!(model::Parameters, ∇model, η)
    for name in fieldnames(Parameters)
        grad = getfield(∇model, name)
        grad === nothing && continue
        param = getfield(model, name)
        setfield!(model, name, param .- η .* grad)
    end
end

function train!(model, x, y, epochs, η)
    for epoch in 1:epochs
        ℓ, (∇model,) = Zygote.withgradient(loss, model, x, y)
        update!(model, ∇model, η)
        epoch % 50 == 0 && @info "epoch=$epoch loss=$(round(ℓ; digits = 4))"
    end
    return model
end

function generate(model, seed, n)
    idx = [vocab_idx[seed]]
    for _ in 1:n
        logits = forward(idx, model)
        p = softmax(logits[end:end, :])[1, :]
        push!(idx, sample(1:length(vocab), Weights(p)))
    end
    join(vocab[i] for i in idx)
end

@code_warntype train!(model, x, y, epochs, η)

# @info "Sample: $(generate(model, 'A', 20))"
