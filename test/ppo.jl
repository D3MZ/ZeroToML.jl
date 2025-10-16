using ZeroToML
using Test
using Random
using Statistics

struct BanditEnv
    state::Vector{Float32}
    rewards::Vector{Float32}
end

BanditEnv() = BanditEnv(Float32[1f0, 0f0], Float32[1f0, 0f0])

function ZeroToML.reset!(env::BanditEnv)
    env.state .= Float32[1f0, 0f0]
    env.state
end

function ZeroToML.step!(env::BanditEnv, action)
    reward = env.rewards[action]
    done = true
    (copy(env.state), reward, done)
end

@testset "PPO" begin
    Random.seed!(7)
    env = BanditEnv()
    s₀ = ZeroToML.reset!(env)

    input_dim = length(s₀)
    action_dim = length(env.rewards)

    η = 5f-3
    agent = ZeroToML.PPO(; input_dim=input_dim, action_dim=action_dim, hidden_dim=16, η=η)

    π₀ = ZeroToML.policy(agent, s₀)

    steps = 64
    iterations = 25
    agent = ZeroToML.train!(agent, env, steps, iterations; epochs=4)

    πᴱ = ZeroToML.policy(agent, ZeroToML.reset!(env))
    @info "initial=$(π₀) trained=$(πᴱ)"
    @test first(πᴱ) > first(π₀)
    @test sum(πᴱ) ≈ 1f0 atol=1f-3
end
