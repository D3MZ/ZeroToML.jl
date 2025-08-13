# --- Data Preparation ---
input_text = repeat("AB", 1_000)
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
max_iters = 10000


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
        
        logits, _ = model(context_tokens)
        last_logits = logits[:, end]
        probs = softmax(last_logits)
        
        next_token = sample_multinomial(probs)
        
        push!(tokens, next_token)
        print(itos[next_token])
    end
    println()
end

# --- Main Execution ---
@info "Initializing Transformer Model..."
model = Transformer(vocab_size, embed_size, seq_len, num_heads, num_layers, ff_hidden_size)

@info """

Model Architecture:
  Vocab Size: $vocab_size
  Embedding Size: $embed_size
  Sequence Length: $seq_len
  Num Heads: $num_heads
  Num Layers: $num_layers
  FF Hidden Size: $ff_hidden_size
  Learning Rate: $learning_rate
  Max Iters: $max_iters
"""

data = collect(input_text)

@info "Starting training loop..."
for iter in 1:max_iters
    zero_gradients!(model)
    
    x, y = get_batch(data, seq_len)
    
    logits, cache = model(x)
    loss = cross_entropy_loss(logits, y)
    
    dlogits = cross_entropy_loss_backward(logits, y)
    backward!(model, dlogits, cache)
    
    update!(model, learning_rate)
    
    if iter % 100 == 0 || iter == 1
        @info "iter $iter, loss: $loss"
    end
end
@info "Training loop finished."

@info "Generating text after training..."
generate_text(model, "A", 50)
