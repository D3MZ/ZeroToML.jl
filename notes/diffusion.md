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
| $\mathbb{E}_{p}[f(X)]$ | expectation of $f(X)$ under distribution $p$ | discrete: $\sum_x f(x)\,p(x)$; continuous: $\int f(x)\,p(x)\,dx$ |

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

Markov kernel:
```math
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t,
\quad
\varepsilon_t \sim \mathcal{N}(0,I)
```

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

### Training

### [Variational Calculus](https://en.wikipedia.org/wiki/Calculus_of_variations)

| Concept    | Notation                                     | Example                                                |
|------------|----------------------------------------------|--------------------------------------------------------|
| Function   | $f:\mathbb{R}\to\mathbb{R}$                  | $f(x)=x^2 \Rightarrow f(3)=9$                          |
| Functional | $J:\{y:\mathbb{R}\to\mathbb{R}\}\to\mathbb{R}$ | $J[y]=\int_0^1 y(x)^2\,dx$ takes the entire function $y(x)$ and outputs a number |

Variational Calculus finds the function $y(x)$ that minimizes or maximizes a functional:

$$
J[y] = \int_{x_0}^{x_1} F(x, y, y')\, dx,
\qquad y' = \frac{dy}{dx}
$$

The solution is given by the Euler–Lagrange equation:

$$
\frac{\partial F}{\partial y} - \frac{d}{dx}\Bigl(\frac{\partial F}{\partial y'}\Bigr) = 0
$$

#### Examples
* Straight line on flat plane: If \(F = \sqrt{1+y'^2}\) (arc length functional), solving Euler–Lagrange gives \(y''=0 \Rightarrow y=ax+b\), a straight line — the curve of shortest length.
* Great circle on a sphere: If constrained to a sphere, Euler–Lagrange with the constraint yields **geodesics** (great circles). Multiple geodesics can connect two points, especially antipodal ones.

#### [Variational Bayesian inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)

##### Problem
Given
- 
- Unobserved variables ${\displaystyle \mathbf {U} =\{U_{1}\dots U_{n}\}}$ given some data ${\displaystyle \mathbf {X} }$ is approximated by a variational distribution, ${\displaystyle Q(\mathbf {U} ):}$
- 



#### Kullback–Leibler divergence
Denoted as ${\displaystyle D_{\text{KL}}(P\parallel Q)}$ measures of how much a model probability distribution $Q$ is different from a true probability distribution $P$. It is defined as

```math
D_{\mathrm{KL}}(P\parallel Q) = \displaystyle \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} = - \displaystyle \sum_{x \in \mathcal{X}} P(x) \log \frac{Q(x)}{P(x)}
```

It is the expectation, $\mathbb{E}$, of the logarithmic difference between the probabilities P and Q, where the expectation is taken using the probabilities P.

[Gibbs' inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality)
- ${\displaystyle D_{\text{KL}}(P\parallel Q)} > 0$
- ${\displaystyle D_{\text{KL}}(P\parallel Q)} = 0$ when $P = Q$

Relative entropy (KL divergence) $D_{\mathrm{KL}}(P \parallel Q)$ is convex in the pair of probability measures $(P, Q)$. Specifically, if $(P_1,Q_1)$ and $(P_2,Q_2)$ are two pairs of probability measures, then
```math
D_{\mathrm{KL}}\!\bigl(\lambda P_1 + (1-\lambda)P_2 \,\|\, \lambda Q_1 + (1-\lambda)Q_2 \bigr)
\;\le\;
\lambda\,D_{\mathrm{KL}}(P_1 \,\|\, Q_1) + (1-\lambda)\,D_{\mathrm{KL}}(P_2 \,\|\, Q_2)
```

#### Jensen’s Inequality 
Evidence lower bound (ELBO) a.k.a variational lower bound or negative variational free energy.

```math
\underbrace{f(\mathbb{E}[X])}{\text{true but intractable objective}}
\;\le\;
\underbrace{\mathbb{E}[f(X)]}{\text{tractable upper bound}}.
```
We cannot evaluate $f(\mathbb{E}[X])$ directly (because the expectation inside is an intractable integral over latent variables).
But we know:
- $\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])$ for all $X$.
- If we minimize $\mathbb{E}[f(X)]$, we push it downwards toward its smallest achievable value.
- Because it is always above $f(\mathbb{E}[X])$, minimizing the upper bound also reduces $f(\mathbb{E}[X])$.

Formally, if
$f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$,
then
```math
\arg\min_{\theta}\,\mathbb{E}[f(X)]
\;\approx\;
\arg\min_{\theta}\, f(\mathbb{E}[X])
```
,
because both are minimized at the same parameter values when $q = p_\theta$ (the bound is tight there).


#### Loss function 
```math
\mathbb{E}\!\bigl[-\log p_{\theta}(\mathbf{x}_0)\bigr]
\;\le\;
\mathbb{E}_{q}\!\!\Biggl[-\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)}\Biggr]
=
\mathbb{E}_{q}\!\!\Biggl[-\log p(\mathbf{x}_T)
-\sum_{t \ge 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}\mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}\Biggr]
=: \mathcal{L}
```

The Loss, $\mathcal{L}$, is not providing much guarantees as it's stating the expected loss from the forward diffusion process, $\mathbb{E}_{q}$, is going to be worse or equal to the real data, $\mathbb{E}\!\bigl[-\log p_{\theta}(\mathbf{x}_0)\bigr]$.
This is because
1. From the chain rule you get $p_{\theta}(\mathbf{x}_0) = \frac{p_{\theta}(\mathbf{x}_{0:T})}{p_{\theta}(\mathbf{x}_{1:T}\mid \mathbf{x}_0)}$
2. The model's goal is for $p_{\theta}(\mathbf{x}_0) \stackrel{\text{ideal}}{=} q(\mathbf{x_0})$

Therefore, it'll probably be better to say
```math
\mathbb{E}\!\bigl[-\log p_{\theta}(\mathbf{x}_0)\bigr]
\stackrel{\text{ideal}}{=}
\mathbb{E}_{q}\!\!\Biggl[-\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)}\Biggr]
```


With that said, the probability density function for the real data, $x_0$, and rest of the path generated from diffusion $p_{\theta}(\mathbf{x}_{0:T})$, would be 

$\mathbb{E}_{q}\!\!\Biggl[-\log \frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)}\Biggr]$



$q(x_{0:T}) = q(x_0)\, q(x_{1:T}\mid x_0)$


The Expected $\mathbb{E}\!\bigl[-\log p_{\theta}(\mathbf{x}_0)\bigr]$



| Formula | Case | Examples |
|--------|------|----------|
| $\mathbb{E}_{q}[-\log p_{\theta}(x_0)] = \sum_{x_0} q(x_0)[-\log p_{\theta}(x_0)]$ | Discrete | Tokens or words in NLP ($x_0 \in \{1,\dots,V\}$), or discrete categories (e.g. class labels) |
| $\mathbb{E}_{q}[-\log p_{\theta}(x_0)] = \int q(x_0)[-\log p_{\theta}(x_0)] dx_0$ | Continuous | • Pixel intensities (images normalized to [0,1] or [-1,1])<br>• Audio waveforms (real-valued amplitudes)<br>• Time series values (real numbers)<br>• Any continuous latent variable |


TODO:
Update x to boldface x to imply a vector.