# Diffusion

## Background

| Symbol              | Meaning                                 | Notes                                                      |
|:--------------------|:----------------------------------------|:-----------------------------------------------------------|
| $x$                 | observed data (image, text, etc.)       | element of space $\mathcal{X}$                             |
| $z$                 | latent variable (hidden cause)          | element of space $\mathcal{Z}$                             |
| $p(z)$              | prior distribution on latent variable   | typically standard Gaussian $\mathcal{N}(0,I)$; here, $\mathcal{N}(\mu,\Sigma)$ denotes a (possibly multivariate) normal distribution with mean vector $\mu$ and covariance matrix $\Sigma$ |
| $I$                 | identity matrix                        | square matrix with ones on the diagonal and zeros elsewhere; used as covariance in standard normal $\mathcal{N}(0,I)$ |
| $\mathcal{N}(\mu,\Sigma)$ | Gaussian (normal) distribution; $\mathcal{N}$ is short for "Normal"  | mean $\mu$, covariance $\Sigma$; special case $\mathcal{N}(0,I)$ = standard normal |
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

| Symbol / Formula                                                                                                  | Explanation |
|:---------------------------------------------------------------------------------------------------------|:-------------|
| $x_0$ | observed data vector | can represent arbitrary vector (e.g. if $x_0$ is a $32\times 32\times 3$ image, it is that shape) |
| $x_{1:T}$ | latent variables | hidden / unobserved variables with same dimensionality as $x_0$ |
| $p(x_0) = \int p(x_0, x_1) \, dx_1$ | The probability of $x_0$ regardless of $x_1$ in the continuous case (integrating over $x_1$). |
| $p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)$ | Definition of the reverse process as a Markov chain with learned Gaussian transitions. |
| $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$ | Gaussian conditional transition in the reverse process with learned mean and covariance. |
| $q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$ | Forward (diffusion) process: fixed Markov chain gradually adding Gaussian noise. |
| $\beta_t$ | variance schedule | controls noise level added at diffusion step t (small positive scalar, often linearly or cosine scheduled) |
| $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ | Single-step Gaussian noise addition with variance schedule $\beta_t$. |
| $\bar{\alpha}_t$ | cumulative product of $(1-\beta_t)$ | $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$, used for closed-form $q(x_t \mid x_0)$ |
| $q(x_t\mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$ | Closed-form distribution of $x_t$ given $x_0$ using cumulative noise schedule. |
