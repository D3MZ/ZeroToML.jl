using ZeroToML
using Test

@testset "Optimizers" begin
    @testset "Adam" begin
        # Using FeedForward as a test case for a component with parameters
        ff = FeedForward(2, 2)
        
        # Initialize parameters and gradients for predictability
        ff.W1 .= 1.0; ff.b1 .= 1.0; ff.W2 .= 1.0; ff.b2 .= 1.0
        ff.∇W1 .= 0.1; ff.∇b1 .= 0.1; ff.∇W2 .= 0.1; ff.∇b2 .= 0.1
        
        # Initialize Adam moment estimates to zero
        fill!(ff.m_W1, 0); fill!(ff.v_W1, 0); fill!(ff.m_b1, 0); fill!(ff.v_b1, 0)
        fill!(ff.m_W2, 0); fill!(ff.v_W2, 0); fill!(ff.m_b2, 0); fill!(ff.v_b2, 0)

        optimizer = Adam(lr=0.001)
        optimizer.t = 1 # Manually increment t as update!(model, ...) does

        update!(ff, optimizer)

        # Manual calculation for W1
        lr=0.001; beta1=0.9; beta2=0.999; t=1; epsilon=1e-8
        grad = fill(0.1, 2, 2)
        m = zeros(2, 2)
        v = zeros(2, 2)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad .^ 2)
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        
        expected_W1 = ones(2, 2) - lr * m_hat ./ (sqrt.(v_hat) .+ epsilon)

        @test ff.W1 ≈ expected_W1
        
        # test b1 as well
        grad_b = fill(0.1, 2)
        m_b = zeros(2)
        v_b = zeros(2)

        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * (grad_b .^ 2)
        m_hat_b = m_b / (1 - beta1^t)
        v_hat_b = v_b / (1 - beta2^t)
        
        expected_b1 = ones(2) - lr * m_hat_b ./ (sqrt.(v_hat_b) .+ epsilon)

        @test ff.b1 ≈ expected_b1
    end
end
