#This takes about an hour to train on a M1 Max CPU
using ZeroToML
using Statistics
using Random

Random.seed!(0xBAADF00D)
text = read("examples/shakespeare.txt", String)
vocab = build_vocab(text)
max_seq_len = 1000
model = parameters(vocab; dₑ=128, d_ff=512, max_seq_len=max_seq_len)

x = encode(text[1:end-1], vocab)
y = encode(text[2:end], vocab)

learning_rate = 1f-1
epochs  = 1

@time model = train(model, x, y, max_seq_len, learning_rate, epochs)
@info param_count(model)

n_generate = 40
seed_len = 10
start_idx = rand(1:(length(text) - (n_generate + seed_len)))
seed_text = text[start_idx:start_idx+seed_len-1]
generated = generate(model, vocab, seed_text; n=n_generate)
actual_text = text[start_idx:start_idx+seed_len+n_generate-1]
@info "Generated" seed=seed_text generated=generated actual=actual_text