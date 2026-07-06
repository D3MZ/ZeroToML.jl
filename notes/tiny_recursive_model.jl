# Tiny Recursive Model walkthrough
# Run this file one cell/section at a time in the Julia REPL.
# The goal is not to reproduce the paper's large experiments, but to make the
# mechanics visible on one tiny supervised puzzle.

using ZeroToML
using Plots
using Random
using Statistics
using Zygote

import ZeroToML: embed, network, update

Random.seed!(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. A tiny puzzle
# ─────────────────────────────────────────────────────────────────────────────
# TRM is supervised: given an input question x, it repeatedly improves an
# embedded answer y and a latent reasoning state z.
#
# Here the puzzle is deliberately tiny: copy the 3×3 token grid.
# Token 1 is just another token; the model only sees integers.

x = [2, 3, 4,
     5, 2, 3,
     4, 5, 2]

y = copy(x)

grid(v) = reshape(v, 3, 3)'

heatmap(grid(x); title="input x", aspect_ratio=1, c=:viridis, clims=(1, 5))

# ─────────────────────────────────────────────────────────────────────────────
# 2. The model pieces
# ─────────────────────────────────────────────────────────────────────────────
# x is embedded by E.
# y starts from a learned answer state Y.
# z starts from a learned latent reasoning state Z.
# The same tiny network is used for both updates:
#
#   z ← net(x + y + z)      latent reasoning
#   y ← net(y + z)          answer refinement
#
# A full recursion does n latent updates and 1 answer update.
# Deep recursion does T full recursions, detaching between cycles in training.

model = TRM(vocab=5, context=9, width=12)

X = embed(model, x)
Y = model.Y
Z = model.Z

heatmap(X; title="embedded input X", xlabel="position", ylabel="feature")
heatmap(Y; title="initial answer state Y", xlabel="position", ylabel="feature")
heatmap(Z; title="initial latent state Z", xlabel="position", ylabel="feature")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Look at the untrained model
# ─────────────────────────────────────────────────────────────────────────────

logits, answer, latent = forward(x, model; n=3, T=2)
ŷ = predict(model, x; n=3, T=2)

heatmap(grid(ŷ); title="untrained prediction", aspect_ratio=1, c=:viridis, clims=(1, 5))
loss(model, x, y; n=3, T=2)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Trace every recursive step
# ─────────────────────────────────────────────────────────────────────────────
# This helper records the answer after every latent update and answer update.
# The answer usually only changes after y is refined, but plotting every step
# makes the rhythm of the algorithm clear.

function confidence(logits)
    vec(maximum(softmax(logits); dims=1))
end

function trace(model, x; n=3, T=2)
    X = embed(model, x)
    Y = model.Y
    Z = model.Z
    rows = NamedTuple[]

    push!(rows, (; cycle=0, step=0, kind="init", answer=predict(model, x; n=0, T=1), confidence=confidence(model.O * Y), Y, Z))

    for cycle in 1:T
        for step in 1:n
            Z = network(model, X .+ Y .+ Z)
            push!(rows, (; cycle, step, kind="latent", answer=vec(argmax.(eachcol(model.O * Y))), confidence=confidence(model.O * Y), Y, Z))
        end
        Y = network(model, Y .+ Z)
        push!(rows, (; cycle, step=n + 1, kind="answer", answer=vec(argmax.(eachcol(model.O * Y))), confidence=confidence(model.O * Y), Y, Z))
    end

    rows
end

rows = trace(model, x; n=3, T=2)

plot([mean(row.confidence) for row in rows]; marker=:circle, label=false,
     xlabel="recursive step", ylabel="mean max probability", title="untrained confidence")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train functionally
# ─────────────────────────────────────────────────────────────────────────────
# This mirrors the library's step function, but keeps every loss value so we can
# plot learning. The model is immutable; each update returns a new model.

function train_trace(model, x, y, η, epochs; n=3, T=2)
    losses = Float32[]
    for _ in 1:epochs
        push!(losses, loss(model, x, y; n, T))
        (∇,) = gradient(θ -> loss(θ, x, y; n, T), model)
        model = update(model, ∇, η)
    end
    model, losses
end

trained, losses = train_trace(model, x, y, 0.03f0, 160; n=3, T=2)

plot(losses; label=false, xlabel="epoch", ylabel="loss", title="training loss")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Compare before and after
# ─────────────────────────────────────────────────────────────────────────────

ŷ₀ = predict(model, x; n=3, T=2)
ŷ₁ = predict(trained, x; n=3, T=2)

plot(
    heatmap(grid(x); title="target", aspect_ratio=1, c=:viridis, clims=(1, 5)),
    heatmap(grid(ŷ₀); title="before", aspect_ratio=1, c=:viridis, clims=(1, 5)),
    heatmap(grid(ŷ₁); title="after", aspect_ratio=1, c=:viridis, clims=(1, 5));
    layout=(1, 3), size=(900, 280)
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Watch the trained model improve through recursion
# ─────────────────────────────────────────────────────────────────────────────

trained_rows = trace(trained, x; n=3, T=3)
accuracy(row) = mean(row.answer .== y)

plot(
    [accuracy(row) for row in trained_rows];
    marker=:circle,
    label="token accuracy",
    xlabel="recursive step",
    ylabel="accuracy",
    ylim=(0, 1.05),
    title="answer improves through recursion"
)

plot(
    [mean(row.confidence) for row in trained_rows];
    marker=:circle,
    label="mean confidence",
    xlabel="recursive step",
    ylabel="confidence",
    ylim=(0, 1.05),
    title="confidence through recursion"
)

# Show selected answer grids from the recursive trace.
selected = [1, 2, 4, 8, length(trained_rows)]
plots = [heatmap(grid(trained_rows[i].answer); title="step $(i - 1): $(trained_rows[i].kind)", aspect_ratio=1, c=:viridis, clims=(1, 5)) for i in selected]
plot(plots...; layout=(1, length(plots)), size=(1200, 260))

# ─────────────────────────────────────────────────────────────────────────────
# 8. Look inside y and z
# ─────────────────────────────────────────────────────────────────────────────
# y is the embedded answer state. Applying O turns it into token logits.
# z is latent reasoning state. It is useful to the model, but not directly an
# answer. The paper's main reinterpretation is exactly this:
#
#   y = current embedded solution
#   z = latent scratchpad / reasoning state

final = last(trained_rows)

plot(
    heatmap(final.Y; title="final answer state y", xlabel="position", ylabel="feature"),
    heatmap(final.Z; title="final latent state z", xlabel="position", ylabel="feature"),
    heatmap(model.O * final.Y; title="output logits O*y", xlabel="position", ylabel="token");
    layout=(1, 3), size=(950, 280)
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Test-time compute
# ─────────────────────────────────────────────────────────────────────────────
# TRM can spend more recursion at test time by increasing T or n.
# This plot asks: how much accuracy do we get as T grows?

Ts = 1:6
accuracies = [mean(predict(trained, x; n=3, T=T) .== y) for T in Ts]

plot(Ts, accuracies; marker=:circle, label=false,
     xlabel="T full recursions", ylabel="token accuracy", ylim=(0, 1.05),
     title="more test-time recursion")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Decoder-style discrete vocabulary example
# ─────────────────────────────────────────────────────────────────────────────
# Because TRM predicts discrete tokens, the closest existing test is the decoder
# test rather than the diffusion test. Diffusion learns continuous denoising;
# this learns a token at every position with cross entropy.
#
# Here we copy the decoder test setup: x is a character sequence and y is the
# same text shifted one character forward.

text = "A quick brown fox jumps over the lazy dog. "
vocab = build_vocab(text)
context = 12

x_text = text[begin:context]
y_text = text[begin+1:context+1]

x_tokens = encode(x_text, vocab)
y_tokens = encode(y_text, vocab)

text_model = TRM(vocab=length(vocab), context=context, width=20)
text_before = predict(text_model, x_tokens; n=2, T=2)

text_trained, text_losses = train_trace(text_model, x_tokens, y_tokens, 0.05f0, 200; n=2, T=2)
text_after = predict(text_trained, x_tokens; n=2, T=2)

println("input:  ", x_text)
println("target: ", y_text)
println("before: ", decode(text_before, vocab))
println("after:  ", decode(text_after, vocab))

plot(text_losses; label=false, xlabel="epoch", ylabel="loss", title="TRM decoder-style loss")

text_rows = trace(text_trained, x_tokens; n=2, T=4)
text_accuracy = [mean(row.answer .== y_tokens) for row in text_rows]

plot(text_accuracy; marker=:circle, label=false,
     xlabel="recursive step", ylabel="token accuracy", ylim=(0, 1.05),
     title="next-character accuracy through recursion")

# Each row is one recursive step. Each column is one character position.
# Green means correct next-character prediction at that position.
correct = hcat([(row.answer .== y_tokens) for row in text_rows]...)'
heatmap(correct; c=:greens, colorbar=false,
        xlabel="character position", ylabel="recursive step",
        title="where the shifted-text prediction is correct")
