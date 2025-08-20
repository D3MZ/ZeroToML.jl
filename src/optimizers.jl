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

mₜ(mₜ₋₁, gₜ, β₁) = β₁ .* mₜ₋₁ .+ (1 - β₁) .* gₜ
vₜ(vₜ₋₁, gₜ, β₂) = β₂ .* vₜ₋₁ .+ (1 - β₂) .* (gₜ .^ 2)
bias_correct(mₜ, vₜ, β₁, β₂, t) = (mₜ ./ (1 - β₁^t), vₜ ./ (1 - β₂^t))
θ_update(θ, m̂, v̂, η, ϵ) = θ .- η .* m̂ ./ (sqrt.(v̂) .+ ϵ)