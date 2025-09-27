function linear_beta_schedule(T, β_start=0.0001, β_end=0.02)
    return range(β_start, stop=β_end, length=T)
end

function precompute_constants(βs)
    α = 1.0 .- βs
    ᾱ = cumprod(α)
    ᾱ_prev = vcat(1.0, ᾱ[1:end-1])
    
    sqrt_ᾱ = sqrt.(ᾱ)
    sqrt_one_minus_ᾱ = sqrt.(1.0 .- ᾱ)

    posterior_mean_coef1 = βs .* sqrt.(ᾱ_prev) ./ (1.0 .- ᾱ)
    posterior_mean_coef2 = (1.0 .- ᾱ_prev) .* sqrt.(α) ./ (1.0 .- ᾱ)
    
    posterior_variance = βs .* (1.0 .- ᾱ_prev) ./ (1.0 .- ᾱ)
    
    return (
        βs=βs,
        α=α,
        ᾱ=ᾱ,
        ᾱ_prev=ᾱ_prev,
        sqrt_ᾱ=sqrt_ᾱ,
        sqrt_one_minus_ᾱ=sqrt_one_minus_ᾱ,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        posterior_variance=posterior_variance,
    )
end

# q(xₜ|x₀)
function q_sample(x₀, t, constants)
    ϵ = randn(size(x₀))
    xₜ = constants.sqrt_ᾱ[t] * x₀ + constants.sqrt_one_minus_ᾱ[t] * ϵ
    return xₜ, ϵ
end

# μ̃ₜ(xₜ, x₀)
function q_posterior_mean(xₜ, x₀, t, constants)
    μ̃ = constants.posterior_mean_coef1[t] * x₀ + constants.posterior_mean_coef2[t] * xₜ
    return μ̃
end

# β̃ₜ
function q_posterior_variance(t, constants)
    β̃ = constants.posterior_variance[t]
    return β̃
end

# μ_θ(xₜ, t)
function p_mean(model, xₜ, t, constants)
    ϵ_θ = model(xₜ, t)
    μ_θ = (1 / sqrt(constants.α[t])) * (xₜ - constants.βs[t] / constants.sqrt_one_minus_ᾱ[t] * ϵ_θ)
    return μ_θ
end

# p_θ(x_{t-1}|xₜ)
function p_sample(model, xₜ, t, constants)
    μ = p_mean(model, xₜ, t, constants)
    if t == 1
        return μ
    end
    
    σ_t = sqrt(q_posterior_variance(t, constants))
    z = randn(size(xₜ))
    return μ + σ_t * z
end

# L_simple(θ)
function loss(model, x₀, t, constants)
    xₜ, ϵ = q_sample(x₀, t, constants)
    ϵ_θ = model(xₜ, t)
    return sum((ϵ - ϵ_θ).^2)
end
