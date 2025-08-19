module Activations

using Statistics

function softmax(x; dims=1)
    e_x = exp.(x .- maximum(x, dims=dims))
    return e_x ./ sum(e_x, dims=dims)
end

end # module
