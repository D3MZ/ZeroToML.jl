# Score-Based Generative Modeling through SDEs

Paper: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

The main idea is similar to DDPM:

1. Start with clean data `x₀`.
2. Add noise until it looks like Gaussian noise.
3. Learn how to reverse that process.
4. Generate by starting from noise and stepping backward.

The difference is that DDPM uses a discrete Markov chain, while this paper writes the noising process as a continuous-time stochastic differential equation.

## 1. Forward SDE

The paper describes a forward noising process:

```math
dx = f(x,t)dt + g(t)dw
```

where:

- `x` is the current sample.
- `t` is continuous time from `0` to `1`.
- `f(x,t)` is the deterministic drift.
- `g(t)` controls how much random noise is added.
- `w` is Brownian motion.

In `src/sde.jl`, the implemented version is the variance-preserving SDE:

```math
dx = -\frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dw
```

This is implemented as:

```julia
β(sde::VPSDE, t) = sde.βmin + t * (sde.βmax - sde.βmin)
drift(sde::VPSDE, x, t) = -0.5f0 * β(sde, t) .* x
diffusion(sde::VPSDE, t) = sqrt(β(sde, t))
```

## 2. Noise schedule

The VP SDE uses a continuous noise schedule:

```math
\beta(t) = \beta_{min} + t(\beta_{max} - \beta_{min})
```

Small `t` means little noise. Large `t` means lots of noise.

In the test, we use a mild schedule:

```julia
sde = VPSDE(βmin=0.1f0, βmax=2f0)
```

The paper often uses stronger schedules for image generation. The repo test uses a smaller toy setup so training stays fast.

## 3. Closed-form perturbation

Even though the SDE is continuous, the VP SDE has a closed-form marginal distribution:

```math
p_{0t}(x_t | x_0) = \mathcal{N}(x_t; m(t)x_0, \sigma(t)^2 I)
```

where:

```math
m(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)ds\right)
```

and:

```math
\sigma(t) = \sqrt{1 - \exp\left(-\int_0^t \beta(s)ds\right)}
```

In code:

```julia
∫β(sde::VPSDE, t) = sde.βmin * t + 0.5f0 * (sde.βmax - sde.βmin) * t^2
marginal_mean(sde::VPSDE, x, t) = exp(-0.5f0 * ∫β(sde, t)) .* x
marginal_std(sde::VPSDE, t) = sqrt(1f0 - exp(-∫β(sde, t)))
```

So we can sample noisy data directly:

```math
x_t = m(t)x_0 + \sigma(t)\epsilon
```

where:

```math
\epsilon \sim \mathcal{N}(0, I)
```

In code:

```julia
perturbed_sample(sde::VPSDE, x₀, t, ε) = marginal_mean(sde, x₀, t) .+ marginal_std(sde, t) .* ε
```

## 4. What the model learns

The model does not predict noise like DDPM. It predicts a score:

```math
s_\theta(x_t,t) \approx \nabla_{x_t}\log p_t(x_t)
```

The score tells us which direction makes a noisy sample more likely under the data distribution at time `t`.

For the Gaussian perturbation kernel, the conditional score is known:

```math
\nabla_{x_t}\log p_{0t}(x_t | x_0) = -\frac{\epsilon}{\sigma(t)}
```

So training can use denoising score matching.

The repo uses the equivalent scaled loss:

```math
\mathbb{E}\left[\|\sigma(t)s_\theta(x_t,t) + \epsilon\|^2\right]
```

In code:

```julia
loss(m::ScoreSDE, sde::VPSDE, x, t, ε) =
    mean((marginal_std(sde, t) .* forward(m, perturbed_sample(sde, x, t, ε), t) .+ ε).^2)
```

This says:

1. Pick clean data `x₀`.
2. Pick random time `t`.
3. Pick Gaussian noise `ε`.
4. Create noisy data `xₜ`.
5. Train the model so `sθ(xₜ,t)` points back toward the clean data.

## 5. Reverse-time SDE

The key theorem in the paper is that the reverse process is also an SDE:

```math
dx = [f(x,t) - g(t)^2\nabla_x\log p_t(x)]dt + g(t)d\bar{w}
```

The only unknown part is the score:

```math
\nabla_x\log p_t(x)
```

So after training, replace it with the neural network:

```math
s_\theta(x,t)
```

For the VP SDE:

```math
dx = \left[-\frac{1}{2}\beta(t)x - \beta(t)s_\theta(x,t)\right]dt + \sqrt{\beta(t)}d\bar{w}
```

In code, the reverse sampler uses Euler-Maruyama steps from `t=1` down to `t=0`:

```julia
score = forward(m, x, t)
dx = drift(sde, x, t) .- β(sde, t) .* score
x = x .- dx .* Δt .+ diffusion(sde, t) * sqrt(Δt) .* randn(Float32, size(x))
```

The subtraction appears because the loop moves backward in time.

## 6. Denoising a sample

For tests, full reverse sampling is noisy and expensive. The test also uses the Tweedie-style denoised mean:

```math
\hat{x}_0 = \frac{x_t + \sigma(t)^2s_\theta(x_t,t)}{m(t)}
```

In code:

```julia
denoised_mean(m::ScoreSDE, sde::VPSDE, x, t) =
    (x .+ marginal_std(sde, t)^2 .* forward(m, x, t)) ./ exp(-0.5f0 * ∫β(sde, t))
```

This is useful for the toy box test:

1. Make a clean square.
2. Add SDE noise.
3. Train the score network.
4. Denoise the noisy square.
5. Check that the recovered image still contains a square.

## 7. How this differs from DDPM

DDPM:

- discrete timesteps
- predicts `ε`
- uses `βₜ`, `αₜ`, and `ᾱₜ`
- reverse process is a discrete Gaussian transition

Score SDE:

- continuous time
- predicts the score `∇ log pₜ(x)`
- uses drift `f(x,t)` and diffusion `g(t)`
- reverse process is another SDE

The VP SDE connects closely to DDPM. DDPM can be viewed as a discretization of a variance-preserving noising process.

## 8. What is implemented

Implemented in `src/sde.jl`:

- `VPSDE`
- `ScoreSDE`
- forward drift and diffusion functions
- closed-form perturbation kernel
- denoising score-matching loss
- training loop
- reverse SDE sampler
- denoised mean helper

Tested in `test/sde.jl` with the same style as the DDPM toy box test.

## 9. What is not implemented yet

The paper also includes ideas not yet implemented here:

- VE SDE
- sub-VP SDE
- predictor-corrector sampling
- probability flow ODE
- exact likelihood computation
- inverse problems like inpainting and colorization
- larger U-Net style architectures

Those should probably be separate follow-up files or incremental additions to `src/sde.jl`.
