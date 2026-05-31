using ZeroToML

using Test
using Logging

const TEST_LOG_LEVELS = Dict(
    "debug" => Logging.Debug,
    "info" => Logging.Info,
    "warn" => Logging.Warn,
    "error" => Logging.Error,
)
const TEST_LOG_LEVEL = TEST_LOG_LEVELS[lowercase(get(ENV, "ZEROTOML_TEST_LOG_LEVEL", "warn"))]

with_logger(ConsoleLogger(stderr, TEST_LOG_LEVEL)) do
    @testset "ZeroToML.jl" begin
        @testset "Transformers" begin
            include("decoder.jl")
        end

        @testset "Recursive Reasoning" begin
            include("trm.jl")
        end

        @testset "Diffusion" begin
            include("ddpm.jl")
            include("sde.jl")
            include("flow_matching.jl")
        end

        @testset "Neural Network Primitives" begin
            include("convolution.jl")
        end

        @testset "Reinforcement Learning" begin
            include("ppo.jl")
        end

        @testset "Gaussian Processes" begin
            include("gaussian.jl")
        end

        @testset "Kalman Filter" begin
            include("kalman.jl")
        end
    end
end
