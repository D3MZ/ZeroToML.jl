using Plots

walk(n_steps; start=0.0f0, step_std=1.0f0) = cumsum([start; randn(n_steps) .* step_std])
residual(y) = walk(length(y)) .- y

y = walk(100)
[residual(y) for _ in 1:10]