# Autoresearch: diffusion raw box loss

## Objective
Move the diffusion raw denoising mean box loss toward zero without using `reproduce` as the model output. The workload trains one `DDPM` per noise process on the 16×16 / 3×3 box dataset, denoises held-out noisy boxes, then measures how close the raw denoised image is to its nearest valid 3×3 box projection.

## Metrics
- **Primary**: `mean_raw_box_loss` (unitless, lower is better) — mean MSE between raw denoised samples and their reproduced nearest valid boxes across noise processes and evaluation samples.
- **Secondary**: `max_training_s` (s, lower is better) — largest post-warmup training time for any single noise model; must stay below 1.0s.
- **Secondary**: `mean_training_s` (s, lower is better) — mean post-warmup training time across Gaussian, StudentT, and Cauchy.
- **Secondary**: `noise_loss` (unitless, lower is better) — mean trained noise prediction loss across processes.

## How to Run
`./autoresearch.sh` — outputs `METRIC name=value` lines.

## Files in Scope
- `src/diffusion.jl` — DDPM model, forward pass, noise process, training, reverse helpers.
- `test/diffusion.jl` — diffusion visualization/test and raw box loss reporting.
- `autoresearch.sh` — benchmark harness and metric extraction.
- `autoresearch.md` — session notes and tried ideas.
- `autoresearch.ideas.md` — deferred promising ideas.

## Off Limits
- Do not change the benchmark to use `reproduce` as the final denoising result.
- Do not hard-code benchmark outputs or special-case the evaluation samples.
- Do not remove or weaken the runtime gate.
- Do not modify unrelated tests or unrelated model code unless necessary for shared primitives.

## Constraints
- Be careful not to overfit to the benchmark samples.
- Do not cheat by encoding the box projection into the model output or metric path.
- Keep `max_training_s < 1.0` for every kept experiment.
- Prefer simple changes and preserve readability.
- Run `./autoresearch.sh` for every experiment and log all runs.

## What's Been Tried
- Baseline starts from current `DDPM` with 4 conv layers, one training pass over all 196 boxes per process, `T = 100`, `denoise_steps = 100`, and raw box loss measured before the `reproduce` projection.
