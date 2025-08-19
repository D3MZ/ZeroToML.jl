mutable struct Adam
    lr::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    t::Int
end

function Adam(; lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
    Adam(lr, beta1, beta2, epsilon, 0)
end
