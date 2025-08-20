using ZeroToML
using Test
import Optimisers

@testset "Optimizers" begin
    @testset "Adam ≈ Optimisers.Adam" begin
        ff = FeedForward(2, 2)
        
        # Mirror parameters for Optimisers.jl as a NamedTuple
        params = (
            W1 = copy(ff.W1),
            b1 = copy(ff.b1),
            W2 = copy(ff.W2),
            b2 = copy(ff.b2),
        )
        grads = (
            W1 = copy(ff.∇W1),
            b1 = copy(ff.∇b1),
            W2 = copy(ff.∇W2),
            b2 = copy(ff.∇b2),
        )

        # Adam parameters
        η, β1, β2 = rand(), rand(), rand()

        # Optimisers.jl Adam update (first step, bias-corrected)
        opt_ref = Optimisers.Adam(η, (β1, β2))
        state = Optimisers.setup(opt_ref, params)
        state, params_ref = Optimisers.update!(state, params, grads)

        # Our Adam update (first step, bias-corrected)
        opt = Adam(lr=η, beta1=β1, beta2=β2)
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
