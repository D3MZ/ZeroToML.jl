using Random, Statistics, Zygote, LinearAlgebra

"Proximal Policy Optimization (PPO) actor-critic implemented with small multilayer perceptrons"
@kwdef struct PPO
    input_dim
    action_dim
    hidden_dim = 32
    actor_W₁ = glorot(hidden_dim, input_dim)
    actor_b₁ = zeros(Float32, hidden_dim)
    actor_W₂ = glorot(action_dim, hidden_dim)
    actor_b₂ = zeros(Float32, action_dim)
    critic_W₁ = glorot(hidden_dim, input_dim)
    critic_b₁ = zeros(Float32, hidden_dim)
    critic_w₂ = vec(glorot(1, hidden_dim))
    critic_b₂ = zeros(Float32, 1)
    clipϵ = 0.2f0
    γ = 0.99f0
    λ = 0.95f0
    η = 3e-4f0
end

"Placeholder reset! to be specialized by concrete environments"
reset!(env) = error("reset! not implemented for $(typeof(env))")

"Placeholder step! to be specialized by concrete environments"
step!(env, action) = error("step! not implemented for $(typeof(env))")

"Dense layer: W × x + b"
@fastmath dense(W, b, x) = W * x .+ b

"Convert arbitrary state into a Float32 feature vector"
features(state::AbstractArray) = vec(Float32.(state))
features(state::Tuple) = features(collect(state))
features(states::AbstractVector{<:AbstractVector}) = hcat(features.(states)...)
features(state::Number) = Float32[state]

"Information entropy of a categorical distribution"
entropy(probs) = -sum(probs .* log.(probs .+ Float32(eps())))

"Information entropy of a batch of categorical distributions"
entropy(probs::AbstractMatrix) = vec(-sum(probs .* log.(probs .+ Float32(eps())); dims=1))

"Actor network logits ϕ(s) for the current policy"
function actor_logits(ppo::PPO, state)
    x = features(state)
    h = relu(dense(ppo.actor_W₁, ppo.actor_b₁, x))
    dense(ppo.actor_W₂, ppo.actor_b₂, h)
end

"Policy π(a|s) represented as a categorical distribution over actions"
policy(ppo::PPO, state) = softmax(actor_logits(ppo, state))

critic_head(ppo::PPO, h::AbstractVector) = dot(ppo.critic_w₂, h) + first(ppo.critic_b₂)
critic_head(ppo::PPO, h::AbstractMatrix) = vec(ppo.critic_w₂' * h .+ first(ppo.critic_b₂))

"Critic network estimating the state value V(s)"
function value(ppo::PPO, state)
    x = features(state)
    h = relu(dense(ppo.critic_W₁, ppo.critic_b₁, x))
    critic_head(ppo, h)
end

"Log probability of sampling an action from a categorical distribution"
logprob(probs, action) = log(probs[action] + Float32(eps()))

"Log probability of sampling actions from a batch of categorical distributions"
logprob(probs::AbstractMatrix, actions::AbstractVector) = log.(probs[CartesianIndex.(actions, eachindex(actions))] .+ Float32(eps()))

"Categorical sampling without auxiliary frameworks"
function sample_action(probs)
    u = rand(Float32)
    total = zero(Float32)
    for idx in eachindex(probs)
        total += probs[idx]
        if u <= total
            return idx
        end
    end
    lastindex(probs)
end

"Collect trajectory data for a fixed number of interaction steps"
function rollout(ppo::PPO, env, steps)
    states = Vector{Vector{Float32}}(undef, steps)
    actions = Vector{Int}(undef, steps)
    rewards = zeros(Float32, steps)
    dones = falses(steps)
    log_probs = zeros(Float32, steps)
    values = zeros(Float32, steps)

    state = features(reset!(env))
    for step in eachindex(actions)
        states[step] = copy(state)
        probs = policy(ppo, state)
        action = sample_action(probs)
        actions[step] = action
        log_probs[step] = logprob(probs, action)
        values[step] = value(ppo, state)

        new_state, reward, done = step!(env, action)
        rewards[step] = Float32(reward)
        dones[step] = done
        state = done ? features(reset!(env)) : features(new_state)
    end

    (; states, actions, rewards, dones, log_probs, values)
end

"Generalized Advantage Estimation (GAE-λ)"
function advantages(ppo::PPO, rewards, values, dones)
    adv = similar(rewards)
    gae = 0f0
    next_value = 0f0
    for t in reverse(eachindex(rewards))
        mask = dones[t] ? 0f0 : 1f0
        δ = rewards[t] + ppo.γ * next_value * mask - values[t]
        gae = δ + ppo.γ * ppo.λ * mask * gae
        adv[t] = gae
        next_value = values[t]
    end
    returns = adv .+ values
    μ = Float32(mean(adv))
    σ = Float32(std(adv))
    ϵ = Float32(eps())
    normalized = (adv .- μ) ./ (σ + ϵ)
    normalized, returns
end

"Clipped PPO objective combining policy, value, and entropy terms"
function loss(ppo::PPO, states, actions, advantages, returns, old_log_probs)
    πs = policy(ppo, states)
    new_logπs = logprob(πs, actions)
    ratios = exp.(new_logπs .- old_log_probs)
    clipped_ratios = clamp.(ratios, 1f0 - ppo.clipϵ, 1f0 + ppo.clipϵ)
    policy_term = -mean(min.(ratios .* advantages, clipped_ratios .* advantages))

    v̂s = value(ppo, states)
    value_term = mean((v̂s .- returns) .^ 2)

    entropies = entropy(πs)
    entropy_term = mean(entropies)

    policy_term + 0.5f0 * value_term - 0.01f0 * entropy_term
end

"Per-iteration improvement using collected trajectories"
function improve!(ppo::PPO, batch, adv, returns, epochs)
    for _ in 1:epochs
        (∇,) = gradient(θ -> loss(θ, batch.states, batch.actions, adv, returns, batch.log_probs), ppo)
        sgd!(ppo, ∇, ppo.η)
    end
    ppo
end

"Train PPO through repeated rollouts and improvement steps"
function train!(ppo::PPO, env, steps, iterations; epochs=4)
    foldl(1:iterations; init=ppo) do agent, iteration
        batch = rollout(agent, env, steps)
        adv, returns = advantages(agent, batch.rewards, batch.values, batch.dones)
        improve!(agent, batch, adv, returns, epochs)
        loss_value = loss(agent, batch.states, batch.actions, adv, returns, batch.log_probs)
        @info "iteration=$(iteration) loss=$(loss_value)"
        agent
    end
end
