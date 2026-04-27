"Kalman filter for the linear-Gaussian state-space model"
mutable struct KalmanFilter{T}
    Φ          # Transition matrix
    M          # Observation matrix
    Q          # Process noise covariance
    R          # Observation noise covariance
    x          # State estimate
    P          # Error covariance
end

"One step of the Kalman filter: predict then update given observation y"
function step!(kf::KalmanFilter, y)
    S = kf.M * kf.P * kf.M' + kf.R      # innovation covariance
    Δ = kf.Φ * kf.P * kf.M' / S         # optimal weighting (Kalman gain)
    Φ★ = kf.Φ - Δ * kf.M                 # Φ★ = Φ - ΔM
    kf.x = Φ★ * kf.x + Δ * y             # state update
    kf.P = Φ★ * kf.P * Φ★' + kf.Q        # covariance update
end

"Simulate a trajectory from the latent state-space model"
function simulate(Φ, M, Q, x₀, T; R=0.01)
    d = length(x₀)
    o = size(M, 1)
    xs = Matrix{eltype(x₀)}(undef, d, T)
    ys = Matrix{eltype(x₀)}(undef, o, T)
    x = copy(x₀)
    Qₗ = cholesky(Q).L
    Rₗ = sqrt(R)
    for t in 1:T
        x = Φ * x + Qₗ * randn(eltype(x₀), d)
        y = M * x + Rₗ * randn(eltype(x₀), o)
        xs[:, t] = x
        ys[:, t] = y
    end
    xs, ys
end
