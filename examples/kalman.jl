using ZeroToML
using LinearAlgebra
using Plots
using Random

Random.seed!(42)

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: 1D constant position — tracking a fixed target
# ─────────────────────────────────────────────────────────────────────────────
# Model: position stays still, we observe it with noise
Φ₁ = Float32[1 0; 0 1]
M₁ = Float32[1 0]           # observe position, not velocity
Q₁ = 1f-4 * I                # tiny process noise
R₁ = Float32[0.25;;]
kf₁ = KalmanFilter(Φ₁, M₁, Q₁, Float32[0.25;;], Float32[0, 0], Float32[10 0; 0 10])
timesteps = 200
obs = [5f0 + 0.5f0 * randn() for _ in 1:timesteps]
est = Vector{Float32}(undef, timesteps)
for t in 1:timesteps
    step!(kf₁, Float32[obs[t]])
    est[t] = kf₁.x[1]
end

p1 = plot(1:timesteps, obs; label="noisy obs", color=:red, marker=:circle, markersize=3, lw=0)
plot!(1:timesteps, est; label="estimate", color=:blue, lw=2)
hline!([5f0]; color=:black, linestyle=:dash, lw=2, label="truth (5)")
title!("Tracking a fixed target")
xlabel!("time"); ylabel!("position")

# ─────────────────────────────────────────────────────────────────────────────
# Example 2: 2D constant-velocity — position tracking under noise
# ─────────────────────────────────────────────────────────────────────────────
dt = 1f0
Φ₂ = Float32[1 dt; 0 1]
M₂ = Float32[1 0]
Q₂ = Float32[1f-4 0; 0 1f-4]

# Simulate ground truth
T = 50
xs = Matrix{Float32}(undef, 2, T)
ys = Matrix{Float32}(undef, 1, T)
global state = Float32[0, 1]
for t in 1:T
    global state
    state = Φ₂ * state + Float32[0.02f0 * randn(), 0.05f0 * randn()]
    ys[:, t] = M₂ * state .+ 0.5f0 * randn()
    xs[:, t] = state
end

kf₂ = KalmanFilter(Φ₂, M₂, Q₂, Float32[0.25;;], Float32[0, 1], Float32[10 0; 0 10])
est = similar(xs)
for t in 1:T
    step!(kf₂, ys[:, t])
    est[:, t] = kf₂.x
end

p2 = plot(1:T, xs[1, :]; label="truth", lw=2, color=:black, linestyle=:dash)
scatter!(1:T, vec(ys); label="noisy obs", color=:red, markersize=2)
plot!(1:T, est[1, :]; label="estimate", color=:blue, lw=2)
title!("Position tracking")
xlabel!("time"); ylabel!("position")

# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — velocity estimation (hidden state recovered)
# ─────────────────────────────────────────────────────────────────────────────
p3 = plot(1:T, xs[2, :]; label="truth", lw=2, color=:black, linestyle=:dash)
plot!(1:T, est[2, :]; label="estimate", color=:green, lw=2)
title!("Velocity (hidden state)")
xlabel!("time"); ylabel!("velocity")

# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — error covariance shrinks as observations accumulate
# ─────────────────────────────────────────────────────────────────────────────
Φ₄ = Float32[1 0; 0 1]
M₄ = Float32[1 0]
Q₄ = 1f-4 * I
kf₄ = KalmanFilter(Φ₄, M₄, Q₄, Float32[0.25;;], Float32[0, 0], Float32[10 0; 0 10])

covs = Vector{Float32}(undef, 50)
for t in 1:50
    step!(kf₄, Float32[5f0 + 0.5f0 * randn()])
    covs[t] = kf₄.P[1, 1]
end

p4 = plot(1:50, covs; label="error variance P[1,1]", color=:purple, lw=2)
title!("Uncertainty shrinks with data")
xlabel!("number of observations"); ylabel!("P")

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────
plot(p1, p2, p3, p4; layout=(2, 2), size=(1000, 800))
