@kwdef struct TRM
    vocab = 4
    context = 4
    width = 8
    E = glorot(width, vocab)
    Y = glorot(width, context)
    Z = glorot(width, context)
    A₁ = glorot(context, context)
    A₂ = glorot(context, context)
    W₁ = glorot(width, width)
    W₂ = glorot(width, width)
    O = glorot(vocab, width)
    q = glorot(1, width)
    bq = zeros(Float32, 1)
end

rmsnorm(X; ϵ=1f-5) = X ./ sqrt.(mean(abs2, X; dims=1) .+ ϵ)
embed(model::TRM, x) = model.E[:, x]

function network(model::TRM, X)
    H = X .+ relu(rmsnorm(X) * model.A₁) * model.A₂
    H .+ model.W₂ * relu(model.W₁ * rmsnorm(H))
end

function latent_recursion(model::TRM, x, y, z, n=6)
    for _ in 1:n
        z = network(model, x .+ y .+ z)
    end
    network(model, y .+ z), z
end

function deep_recursion(model::TRM, x, y=model.Y, z=model.Z; n=6, T=3)
    for _ in 1:T-1
        y, z = latent_recursion(model, x, y, z, n)
        y = Zygote.dropgrad(y)
        z = Zygote.dropgrad(z)
    end
    latent_recursion(model, x, y, z, n)
end

function forward(x, model::TRM; n=6, T=3)
    y, z = deep_recursion(model, embed(model, x), model.Y, model.Z; n, T)
    model.O * y, y, z
end

predict(model::TRM, x; n=6, T=3) = vec(argmax.(eachcol(first(forward(x, model; n, T)))))
halt(model::TRM, y) = only(1f0 ./ (1f0 .+ exp.(-(model.q * mean(y; dims=2) .+ model.bq))))

function loss(model::TRM, x, y; n=6, T=3)
    logits, ŷ, = forward(x, model; n, T)
    shifted = logits .- maximum(logits; dims=1)
    log_probs = shifted .- log.(sum(exp.(shifted); dims=1))
    token_loss = -mean(log_probs[CartesianIndex.(y, eachindex(y))])
    target = Float32(all(vec(argmax.(eachcol(logits))) .== y))
    q = halt(model, ŷ)
    token_loss - 0.1f0 * (target * log(q + Float32(eps())) + (1f0 - target) * log(1f0 - q + Float32(eps())))
end

function update(model::TRM, ∇, η)
    values = map(propertynames(model)) do field
        θ = getproperty(model, field)
        g = ∇ === nothing ? nothing : getproperty(∇, field)
        (g === nothing || !(θ isa AbstractArray)) ? θ : θ .- η .* g
    end
    TRM(; zip(propertynames(model), values)...)
end

function step(model::TRM, x, y, η; n=6, T=3)
    (∇,) = gradient(θ -> loss(θ, x, y; n, T), model)
    update(model, ∇, η)
end

train(model::TRM, x, y, η, epochs; n=6, T=3) = foldl((m, _) -> step(m, x, y, η; n, T), 1:epochs; init=model)
