# SDE Denoised Correlation Optimization

**Goal**: Improve `correlate(x₀, denoised)` from `test/sde.jl`.

**Primary metric**: `denoised_correlation` (higher is better).
**Baseline**: 0.913.
**Constraint**: Training must use `Second(10)` — no cheating by increasing training time.
**Check**: `julia --project test/sde.jl` must pass (both tests).

**What can change**:
- `src/sde.jl`: Model architecture, learning rate, β schedule, sampler, loss weighting, optimizer
- `test/sde.jl`: Only the `@info` line and params fed to `train!`/samplers (keep `Second(10)`, keep tests honest)

**What NOT to do**:
- Don't encode the box projection into the model output
- Don't hard-code benchmark outputs
- Don't special-case evaluation samples
- Don't increase training time beyond `Second(10)`
- Don't change the test assertions themselves
