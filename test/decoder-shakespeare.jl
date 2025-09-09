#too large for automatic testing.

using ZeroToML
using Test
using Statistics
using BenchmarkTools
using Random

Random.seed!(0xBAADF00D)
text = read("./data/shakespeare.txt", String)[1:1000]
vocab = build_vocab(text)
max_seq_len = 100
model = parameters(vocab; max_seq_len=max_seq_len)

x = encode(text[1:end-1], vocab)
y = encode(text[2:end], vocab)

learning_rate = 9f-1
epochs  = 100

model = train!(model, x, y, epochs, learning_rate)
ℓ = loss(model, x, y)
@info "Post-train loss" loss=ℓ
# @test ℓ < 1e-3

n_generate = 40
seed_len = 10
start_idx = rand(1:(length(text) - (n_generate + seed_len)))
seed_text = text[start_idx:start_idx+seed_len-1]
generated = generate(model, vocab, seed_text; n=n_generate)
actual_text = text[start_idx:start_idx+seed_len+n_generate-1]
@info "Generated" seed=seed_text generated=generated actual=actual_text