# --- Weight Initialization ---
glorot(m, n) = (rand(Float32, m, n) .- 0.5f0) .* sqrt(2.0f0 / (m + n))

randn_like(x) = randn(eltype(x), size(x)...)  # one-liner

relu(x) = max.(x, 0f0)

# simple SGD update
sgd!(param, grad, η) = (param .-= η.*grad)
