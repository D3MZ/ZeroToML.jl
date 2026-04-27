# Kalman filter step — fully expanded with every matrix multiply shown.
# If you're still fuzzy on the matrix math, read the // comments
# which show each product concretely for our specific model.

using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# 0. The big idea
# ─────────────────────────────────────────────────────────────────────────────
# The Kalman filter assumes both the state (position, velocity) and the
# observations are random variables with Gaussian (normal) distributions.
#
# Instead of tracking a single number for position, it tracks a *distribution*
# over positions: a mean (best guess) and a variance (how sure we are).
#
# At each step:
#   1. PREDICT: evolve the distribution forward in time (Φ makes it move,
#      Q adds uncertainty because the world is random)
#   2. UPDATE: blend the prediction with a new measurement (weighted by how
#      sure we are vs how noisy the sensor is)
#
# Because Gaussians have nice math properties, both steps are just matrix
# multiplications. The whole filter is ~5 lines of linear algebra.
#
# ─────────────────────────────────────────────────────────────────────────────
# 1. What our model looks like
# ─────────────────────────────────────────────────────────────────────────────
# State:  x = [position, velocity]    (2 numbers we want to track)
# Obs:    y = [position measurement]  (1 number we actually see)

Φ = [1 0; 0 1]          # Transition: position stays, velocity stays
# ┌         ┐
# │ 1    0  │   positionₜ₊₁ = 1·positionₜ + 0·velocityₜ
# │ 0    1  │   velocityₜ₊₁ = 0·positionₜ + 1·velocityₜ
# └         ┘
# → both stay the same (we're tracking a fixed target)

M = [1 0]                # Observation: we see position, not velocity
# ┌         ┐
# │ 1    0  │   y = 1·position + 0·velocity = position
# └         ┘
# → velocity is "hidden" — we never measure it directly

Q = 1e-4 * I             # Process noise: tiny random bumps
# → if the target drifts, it drifts very slowly (variance 0.0001 per step)

R = [0.25;;]             # Observation noise: std=0.5, variance=0.25
# → each measurement is off by about ±0.5 on average

x = [0.0, 0.0]           # Initial guess: "I think it's at position 0, velocity 0"
P = [10 0; 0 10]         # Initial uncertainty: "but I'm very unsure"
# ┌          ┐
# │ 10    0  │   position uncertainty = 10 (std ≈ 3.2)
# │ 0    10  │   velocity uncertainty = 10
# └          ┘
#   0 off-diagonal = position & velocity uncertainty are independent (no correlation)

y = [5.3]                # First measurement: "the sensor says 5.3"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Step-by-step with every matrix product expanded
# ─────────────────────────────────────────────────────────────────────────────

# ── Step A: Innovation covariance S ──
# S tells us: "how uncertain are we about what we expect to see?"
# Big S = we're less sure = we won't trust the observation much
# Small S = we're more sure = we'll trust the observation more
#
# Formula: S = M * P * M' + R
#
# M is 1×2, P is 2×2, M' is 2×1  →  result is 1×1 (a single number)

# First compute the vector P * M'  (projecting uncertainty into observation space)
# ┌          ┐   ┌   ┐    ┌      ┐
# │ 10    0  │ * │ 1 │  = │ 10   │    P*M' is 2×1
# │ 0    10  │   │ 0 │    │  0   │
# └          ┘   └   ┘    └      ┘
#
# Interpretation:
#   row 1 (position): 10*1 + 0*0 = 10  → position uncertainty projected forward
#   row 2 (velocity):  0*1 + 10*0 = 0   → velocity uncertainty has no effect (M ignores velocity)

PM′ = P * M'
println("P * M' = $PM′")
println("  → projects uncertainty into observation dimensions")
println()

# Now multiply by M to get the scalar innovation
# ┌         ┐   ┌      ┐
# │ 1    0  │ * │ 10   │  =  [10]
# └         ┘   │  0   │
#               └      ┘

MPM′ = M * PM′
println("M * P * M' = $MPM′")
println("  = position uncertainty alone (10)")
println()

S = MPM′ + R
println("S = M*P*M' + R = $MPM′ + $R = $S")
println("  = position uncertainty (10) + observation noise (0.25) = 10.25")
println("  → the observation noise barely adds anything because")
println("     our initial uncertainty (10) is way bigger than sensor noise (0.25)")
println()

# ── Step B: Kalman gain Δ ──
# Δ tells us: "what fraction of the new info should I blend in?"
# Δ near 1 → trust the observation almost completely
# Δ near 0 → ignore the observation, trust the prediction
#
# Formula: Δ = Φ * P * M' / S
#
# Φ is 2×2, P*M' is 2×1, we divide by scalar S → result is 2×1

ΦPM′ = Φ * PM′   # Φ = I (identity), so this is just P*M' again
println("Φ * P * M' = $ΦPM′")
println("  (Φ is identity, so it's the same as P*M')")
println()

Δ = ΦPM′ / S[1]   # S is a 1×1 matrix, extract the scalar
println("Δ = (Φ * P * M') / S = [$ΦPM′] / $(S[1])")
println()
println("  Δ[1] = $(ΦPM′[1]) / $(S[1]) = $(ΦPM′[1]/S[1])")
println("  Δ[2] = $(ΦPM′[2]) / $(S[1]) = $(ΦPM′[2]/S[1])")
println()
println("  → Δ[1] = $(round(Δ[1], digits=3)) meaning we trust the observation 97.6%")
println("  → Δ[2] = $(round(Δ[2], digits=3)) meaning we learn nothing about velocity")
println("     (M has a 0 in the velocity column, so measurements give zero velocity info)")
println()

# ── Step C: Modified transition Φ★ ──
# Φ★ = Φ - Δ * M
# This says: "after accounting for the observation, how much do I keep from before?"

ΔM = Δ * M
println("Δ * M = [$Δ] * [1 0] = $ΔM")
println("  → subtracts gain from the position row, leaves velocity alone")

Φ★ = Φ - ΔM
println("Φ★ = Φ - Δ*M = $Φ★")
println("  → position row: [1 0] - [$(round(Δ[1], digits=3)) 0] = [$(round(Φ★[1,1], digits=3)) 0]")
println("  → velocity row: unchanged (since Δ[2]=0)")
println("  → meaning: position keeps $(round(Φ★[1,1]*100, digits=1))% of its old value")
println("     (the other $(round(Δ[1]*100, digits=1))% comes from the observation via Δ)")
println()

# ── Step D: State update ──
# x = Φ★ * x + Δ * y
# Blend: (what we kept from before) + (what we learned from the observation)

Φ★x = Φ★ * x
println("Φ★ * x = $Φ★ * $x = $Φ★x")
println("  → old position $(x[1]) × $(Φ★[1,1]) = $(Φ★x[1]) (almost nothing)")
println("  → old velocity $(x[2]) × $(Φ★[2,2]) = $(Φ★x[2]) (unchanged)")

Δy = Δ .* y[1]   # Δ is 2×1, y[1] is scalar
println("Δ * y = $Δ * $(y[1]) = $Δy")
println("  → new position info: Δ[1] × $(y[1]) = $(round(Δy[1], digits=3))")
println("  → new velocity info: Δ[2] × $(y[1]) = $(round(Δy[2], digits=3)) (zero, as expected)")

x = Φ★x + Δy
println("x = Φ★*x + Δ*y = $Φ★x + $Δy = $x")
println("  → position: $(round(Φ★x[1], digits=3)) + $(round(Δy[1], digits=3)) = $(round(x[1], digits=3))")
println("  → velocity: $(round(Φ★x[2], digits=3)) + $(round(Δy[2], digits=3)) = $(round(x[2], digits=3))")
println("  → the estimate jumps from 0 to ≈$(round(x[1], digits=2)) (close to the observation)")
println()

# ── Step E: Covariance update ──
# P = Φ★ * P * Φ★' + Q
# Shrink uncertainty: Φ★ shrinks P because we learned something
# Add Q: the process noise adds a tiny bit back (target might drift)

Φ★P = Φ★ * P
println("Φ★ * P = $Φ★ * [10 0; 0 10] = $Φ★P")
println("  → position uncertainty was 10, now $(round(Φ★P[1,1], digits=4)) ($(round(Φ★P[1,1]/10*100, digits=1))% of before)")
println("  → velocity uncertainty unchanged at row 2")

Φ★PΦ★′ = Φ★P * Φ★'
println("Φ★ * P * Φ★' = $Φ★P * $Φ★' = $Φ★PΦ★′")
println("  → position uncertainty: $(round(Φ★PΦ★′[1,1], digits=6)) (cratered!)")
println("  → velocity uncertainty: $(round(Φ★PΦ★′[2,2], digits=2)) (barely changed)")

P = Φ★PΦ★′ + Q
println("P = Φ★*P*Φ★' + Q = $Φ★PΦ★′ + $(Q) = $P")
println("  → position uncertainty went from 10 → $(round(P[1,1], digits=5))")
println("  → velocity uncertainty went from 10 → $(round(P[2,2], digits=2))")
println("  → we learned a LOT about position, almost nothing about velocity")
println()

# ─────────────────────────────────────────────────────────────────────────────
# 3. What happens on subsequent steps?
# ─────────────────────────────────────────────────────────────────────────────
println("=== What changes on step 2? ===")
println()
println("P[1,1] is now ~$(round(P[1,1], digits=5)) instead of 10.")
println("So S = P[1,1] + R ≈ $(round(P[1,1], digits=5)) + 0.25 = $(round(P[1,1]+0.25, digits=5))")
println("And Δ[1] = P[1,1] / (P[1,1] + R) ≈ $(round(P[1,1]/(P[1,1]+0.25), digits=4)) (much smaller!)")
println()
println("The gain drops because uncertainty shrunk. The filter now")
println("listens less to each new observation and trusts its estimate more.")
println()
println("After ~70 steps it reaches steady state:")
println("  P[1,1] ≈ 0.0036")
println("  Δ[1]  ≈ 0.014")
println("  → each new observation is only blended in at 1.4% weight")
println("  → the estimate is an exponential average over ~71 observations")
println("  → it bounces around the truth with std ≈ 0.042")
println("    (it never converges to a point because Q > 0 keeps the filter alert)")
