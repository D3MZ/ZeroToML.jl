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

learning_rate = 1e-2
# epochs  = 10

for (x, y) in zip(Iterators.partition(x, max_seq_len), Iterators.partition(y, max_seq_len))
    train!(model, x, y, 1, learning_rate)
    @info "Post-train loss" loss=loss(model, x, y)
end

# model = train!(model, x, y, epochs, learning_rate)
# @test ℓ < 1e-3

# n_generate = 40
# seed_len = 10
# start_idx = rand(1:(length(text) - (n_generate + seed_len)))
# seed_text = text[start_idx:start_idx+seed_len-1]
# generated = generate(model, vocab, seed_text; n=n_generate)
# actual_text = text[start_idx:start_idx+seed_len+n_generate-1]
# @info "Generated" seed=seed_text generated=generated actual=actual_text
