# Diffusion

## General Background

| Symbol              | Meaning                                 | Notes                                                      |
|:--------------------|:----------------------------------------|:-----------------------------------------------------------|
| $x$                 | observed data (image, text, etc.)       | element of space $\mathcal{X}$                             |
| $z$                 | latent variable (hidden cause)          | element of space $\mathcal{Z}$                             |
| $p(z)$              | prior distribution on latent variable   | typically standard Gaussian $\mathcal{N}(0,I)$; here, $\mathcal{N}(\mu,\Sigma)$ denotes a (possibly multivariate) normal distribution with mean vector $\mu$ and covariance matrix $\Sigma$ |
| $I$                 | identity matrix                        | square matrix with ones on the diagonal and zeros elsewhere; used as covariance in standard normal $\mathcal{N}(0,I)$ |
| $\mathcal{N}(x; \mu, \Sigma)$ | Gaussian (normal) probability density function evaluated at $x$ | $x$ is the vector, $\mu$ is the mean, $\Sigma$ is the Covariance (or Variance if scalar) |
| $\mathcal{N}(\mu,\Sigma)$ | Gaussian (normal) distribution (as a distribution object) |
| $\mathcal{N}(0,I)$ |  standard normal |
| $p_\theta(x \mid z)$| likelihood of $x$ given $z$             | parameterized by neural network weights $\theta$           |
| $p_\theta(x, z)$    | joint distribution                      | factorizes as $p_\theta(x \mid z)p(z)$                     |
| $p_\theta(x)$       | marginal likelihood                     | obtained by integrating out $z$                            |
| $x \sim p(x)$       | sampled from distribution               | In statistics and probability theory, the tilde means "is distributed as" (e.g. X ~ B(n, p) for a binomial distribution)|

| Formula                                                                                                  | Explanation                                                                                                                                                                                                     |
|:---------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$                     | A Gaussian (normal) distribution with density where $\mu$ is the mean and $\sigma^2$ is the variance.                                                                   |
| $p(x_0, x_1) \\= p(x_1, x_0) \\= p(x_1)\,p(x_0 \mid x_1) \\= p(x_0)\,p(x_1 \mid x_0)$                                        | The joint probability of the pair $(x_0, x_1)$ occurring at the same time.                                       |
| $p(x_0 \mid x_1) = \frac{p(x_0, x_1)}{p(x_1)}$                                                           | The Conditional probability of $x_0$ given $x_1$.                                                                                                                                                                          |
| $p(x_0)=\sum_{x_1}p(x_0,x_1)$   | The probability of $x_0$ regardless of $x_1$ in the discrete case (summing over $x_1$). |
| $p(x_0)=\int p(x_0,x_1) dx_1$   | The probability of $x_0$ regardless of $x_1$ in the continuous case (integrating over $x_1$). |

## [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

The model only predicts the parameters of the Gaussian reverse transition (Mean $\mu_\theta(x_t,t)$ and Variance $\Sigma_\theta(x_t,t)$), not the entire distribution explicitly.

| Symbol / Formula                                                                                                  | Explanation |
|:---------------------------------------------------------------------------------------------------------|:-------------|
| $x_0$ | observed data vector | can represent arbitrary vector (e.g. if $x_0$ is a $32\times 32\times 3$ image, it is that shape) |
| $x_{1:T}$ | latent variables | hidden / unobserved variables with same dimensionality as $x_0$ |
| $\varepsilon_t \sim \mathcal{N}(0, I)$ | random noise | 
| $p(x_0) = \int p(x_0, x_1) \, dx_1$ | The probability of $x_0$ regardless of $x_1$ in the continuous case (integrating over $x_1$). |
| $p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)$ | Definition of the reverse process as a Markov chain with learned Gaussian transitions. |
| $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$ | Gaussian conditional transition in the reverse process with learned mean and covariance. |
| $q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$ | Forward (diffusion) process: fixed Markov chain gradually adding Gaussian noise. |
| $\beta_t = \beta_{\min} + (\beta_{\max}-\beta_{\min})\frac{t-1}{T-1}$ | Evenly increases noise per step |
| $\beta_t = \beta_{\min} + (\beta_{\max}-\beta_{\min})\left(\frac{t-1}{T-1}\right)^2$ | Adds very little noise early, more near the end. |
| $\bar{\alpha}_t = \frac{\cos^2(\frac{\pi}{2}(t/T+0.008))}{\cos^2(0.008\pi/2)}\\$ $\beta_t = 1-\bar{\alpha}_t/\bar{\alpha}_{t-1}$ | Cosine variance schedule; here $\bar{\alpha}_t$ is defined directly as a cosine-squared function of $t$ (not as a cumulative product), and $\beta_t$ is derived via $\beta_t = 1-\bar{\alpha}_t/\bar{\alpha}_{t-1}$; empirically better because it keeps $\bar{\alpha}_t$ high in early steps, making denoising easier. |
| $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ | Single-step Gaussian noise addition with variance schedule $\beta_t$. |
| $\alpha_t := 1-\beta_t$ | per-step signal retention factor | |
| $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$ | cumulative product of per-step $\alpha_s$ | used for closed-form $q(x_t \mid x_0)$ |
| $q(x_t\mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$ | Closed-form distribution of $x_t$ given $x_0$ using cumulative noise schedule. |

Both $q(x_{0:T})$ and $p_\theta(x_{0:T})$ are distributions over the same path $x_0,\dots,x_T$, but factorized in opposite directions:

| Term | Factorization | Direction |
|------|---------------|-----------|
| $q(x_{0:T})$ | $q(x_0)\prod_{t=1}^{T} q(x_t \mid x_{t-1})$ | Forward (data → noise) |
| $p_\theta(x_{0:T})$ | $p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$ | Reverse (noise → data) |

### Forward

$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t,
\quad
\varepsilon_t \sim \mathcal{N}(0,I)$

1. Start from data $x_0$.
2.	For each timestep $t$, iteratively apply noise:
    - Decrease signal $x_t$ by $\sqrt{1-\beta_t}$.
    - Increase noise $\varepsilon_t$ by $\sqrt{\beta_t}$.
3.	After T steps, $x_T$ is nearly pure Gaussian noise $\mathcal{N}(0,I)$ based on $\beta_t$ scheduling.

### Reverse 
1. You start from noise $x_T \sim \mathcal{N}(0,I)$.
2. At each step $t=T,\dots,1:$ Sample $x_{t-1}$ from a Gaussian with mean and variance predicted by the model:
$x_{t-1} \sim \mathcal{N}(\mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$.
3. After T steps, you have $x_0$, which should look like real data.

For any joint distribution $q(x_{0:T})$, we could use the Chain Rule to get the next step by computing all the historical steps:
```math
\begin{aligned}
q(x_{0:T}) &= q(x_0)\, q(x_1 \mid x_0)\, q(x_2 \mid x_1) \,\dots q(x_T \mid x_{T-1}) \\
          &= q(x_0) \prod_{t=1}^{T} q(x_t \mid x_{t-1})
\end{aligned}
```

But allowing for the Markov assumption, where the step is only dependent on previous step, $q(x_t \mid x_{0:t-1}) = q(x_t \mid x_{t-1})$, this can be defined as:

$q(x_t \mid x_{t-1}) := \mathcal{N}\bigl(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\bigr)$
