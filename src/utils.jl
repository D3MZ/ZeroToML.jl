using Random

"Glorot/Xavier initializer for dense weight matrices"
@fastmath function glorot(m, n; gain=1f0)
    scale = gain * sqrt(6f0 / (m + n))
    rand(Float32, m, n) .* (2f0 * scale) .- scale
end

"Glorot/Xavier initializer for convolutional kernels"
@fastmath function glorot(w, h, c_in, c_out; gain=1f0)
    scale = gain * sqrt(6f0 / (w * h * (c_in + c_out)))
    (rand(Float32, w, h, c_in, c_out) .* 2f0 .- 1f0) .* scale
end

"ReLU activation"
relu(x::AbstractArray) = max.(x, zero(eltype(x)))
"ReLU activation"
relu(x::Number) = max(x, zero(x))

"Numerically stable softmax for vectors"
function softmax(logits::AbstractVector)
    shifted = logits .- maximum(logits)
    weights = exp.(shifted)
    weights ./ (sum(weights) + Float32(eps()))
end

"Numerically stable softmax for matrices, computed column-wise"
function softmax(logits::AbstractMatrix)
    shifted = logits .- maximum(logits; dims=1)
    weights = exp.(shifted)
    weights ./ (sum(weights; dims=1) .+ Float32(eps()))
end

"In-place stochastic gradient descent update"
function sgd!(model, grads, η)
    for field in propertynames(model)
        θ = getproperty(model, field)
        g = getproperty(grads, field)
        (g === nothing || !(θ isa AbstractArray)) && continue
        θ .-= η .* g
    end
    model
end
