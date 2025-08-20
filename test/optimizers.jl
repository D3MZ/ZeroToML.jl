using ZeroToML
using Test
using Optimisers
import Optimisers: Adam as optimisers_adam

@testset "Optimizers" begin
    @testset "Adam ≈ Optimisers.Adam" begin
        # Model under test
        ff = FeedForward(2, 2)

        # Deterministic init
        ff.W1 .= 1.0; ff.b1 .= 1.0; ff.W2 .= 1.0; ff.b2 .= 1.0
        ff.∇W1 .= 0.1; ff.∇b1 .= 0.1; ff.∇W2 .= 0.1; ff.∇b2 .= 0.1
        fill!(ff.m_W1, 0); fill!(ff.v_W1, 0); fill!(ff.m_b1, 0); fill!(ff.v_b1, 0)
        fill!(ff.m_W2, 0); fill!(ff.v_W2, 0); fill!(ff.m_b2, 0); fill!(ff.v_b2, 0)

        # Mirror parameters for Optimisers.jl as a NamedTuple
        params = (
            W1 = ones(size(ff.W1)),
            b1 = ones(size(ff.b1)),
            W2 = ones(size(ff.W2)),
            b2 = ones(size(ff.b2)),
        )
        grads = (
            W1 = fill(0.1, size(ff.W1)),
            b1 = fill(0.1, size(ff.b1)),
            W2 = fill(0.1, size(ff.W2)),
            b2 = fill(0.1, size(ff.b2)),
        )

        # Optimisers.jl Adam update (first step, bias-corrected)
        opt_ref = optimisers_adam(0.001)
        state = Optimisers.setup(opt_ref, params)
        params_ref, state = Optimisers.update!(state, params, grads)

        # Our Adam update (first step, bias-corrected)
        opt = Adam(lr=0.001)
        opt.t = 1
        update!(ff, opt)

        @testset "parameters match Optimisers.jl" begin
            @test ff.W1 ≈ params_ref.W1
            @test ff.b1 ≈ params_ref.b1
            @test ff.W2 ≈ params_ref.W2
            @test ff.b2 ≈ params_ref.b2
        end
    end
end
