# Kalman filter step — unpacked for repl-style learning
# Each line is a standalone operation you can inspect.

using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# 1. Set up a simple model
# ─────────────────────────────────────────────────────────────────────────────
Φ = [1 0; 0 1]          # state stays still (position, velocity)
M = [1 0]                # observe position only
Q = 1e-4 * I             # tiny process noise
R = [0.25;;]             # observation noise variance (0.5²)

x = [0.0, 0.0]           # initial state estimate
P = [10 0; 0 10]         # initial uncertainty
y = [5.3]                # a single noisy observation

# ─────────────────────────────────────────────────────────────────────────────
# 2. Step through the filter line by line
# ─────────────────────────────────────────────────────────────────────────────
# Innovation covariance: uncertainty projected into observation space + noise
S = M * P * M' + R

# Kalman gain: how much to trust the observation vs the prediction
Δ = Φ * P * M' / S

# Modified transition: prediction - gain × observation matrix
Φ★ = Φ - Δ * M

# State update: blend prediction and observation
x = Φ★ * x + Δ * y

# Covariance update: shrink uncertainty
P = Φ★ * P * Φ★' + Q

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inspect
# ─────────────────────────────────────────────────────────────────────────────
println("Innovation covariance S = $S")
println("Kalman gain          Δ = $Δ")
println("Modified transition Φ★ = $Φ★")
println("Updated state        x = $x")
println("Updated covariance   P = $P")
println()
println("Step 1: gain Δ[1,1] ≈ $(round(Δ[1], digits=3)) — large because P[1,1]=10 ≫ R=0.25")
println("The estimate jumps from 0 to ≈$(round(x[1], digits=2)).")
println()
println("After more steps, P shrinks and Δ settles to ~0.014.")
println("Then each new observation only nudges the estimate slightly.")
