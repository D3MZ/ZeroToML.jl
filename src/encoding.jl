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
