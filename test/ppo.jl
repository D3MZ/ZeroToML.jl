using ZeroToML
using Test
using Random
using Statistics

import ZeroToML: reset!, step!

struct BanditEnv
    state::Base.Vector{Float32}
    rewards::Base.Vector{Float32}
end

BanditEnv() = BanditEnv(Float32.([1, 0]), Float32.([1, 0]))

function reset!(env)
    env.state .= Float32.([1, 0])
    env.state
end

function step!(env, action)
    reward = env.rewards[action]
    done = true
    (copy(env.state), reward, done)
end

@testset "PPO" begin
    Random.seed!(7)
    env = BanditEnv()
    s₀ = reset!(env)

    input_dim = length(s₀)
    action_dim = length(env.rewards)

    η = 5f-3
    agent = PPO(; input_dim=input_dim, action_dim=action_dim, hidden_dim=16, η=η)

    π₀ = policy(agent, s₀)

    steps = 64
    iterations = 25
    agent = train!(agent, env, steps, iterations; epochs=4)

    πᴱ = policy(agent, reset!(env))
    @debug "initial=$(π₀) trained=$(πᴱ)"
    @test first(πᴱ) > first(π₀)
    @test sum(πᴱ) ≈ 1f0 atol=1f-3
end
