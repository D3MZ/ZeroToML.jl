# Diffusion

## Background 
Suppose $P = \{p_{1}, \ldots, p_{n}\}$ and $Q = \{q_{1}, \ldots, q_{n}\}$ are discrete probability distributions. 

The [Gibbs' inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality), derived from [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality#Probabilistic_form), states the information entropy of $P$ is less than or equal to its cross entropy with any other distribution $Q$. 
```math
-\sum_{i=1}^{n} p_{i} \log p_{i} \;\leq\; -\sum_{i=1}^{n} p_{i} \log q_{i}
```
This is an [evidence lower bound (ELBO)](https://en.wikipedia.org/wiki/Evidence_lower_bound) that by itself is obvious and not much of a guarantee. However, when rearranged into the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence), it's easier to spot that as the gap between P and Q narrows, the distributions must be getting similar:

```math
D_{\mathrm{KL}}(P \parallel Q) \;\equiv\; \sum_{i=1}^{n} p_{i} \log \frac{p_{i}}{q_{i}} \;\geq\; 0
```
This [divergence](https://en.wikipedia.org/wiki/Divergence_(statistics)) that has a few useful properties:
1. The divergence is a real number $\ge 0$. 
2. P $=$ Q only when there's there's no divergence anywhere: $p_i - q_i = 0$.
3. As aformentioned, the ELBO implies as the gap P(X) and Q(X) gets smaller, then Q -> P. This is used as a loss function, because reducing the expected [surprisal](https://en.wikipedia.org/wiki/Information_content) implies the model is gaining information.
4. $D_{\mathrm{KL}}(P \parallel Q) = \infty$ if $p_i > 0,\ q_i = 0$ is anywhere. We'll earmark this for now, this is problematic as a loss functiond.
5. Additive for independent distributions: 
${\displaystyle D_{\text{KL}}(P\parallel Q)=D_{\text{KL}}(P_{1}\parallel Q_{1})+D_{\text{KL}}(P_{2}\parallel Q_{2}).}$
6. [Convex](https://en.wikipedia.org/wiki/Convex_function) in the pair of probability measures: ${\displaystyle (P,Q)}$, i.e. if ${\displaystyle (P_{1},Q_{1})}$ and $ {\displaystyle (P_{2},Q_{2})}$ are two pairs of probability measures then ${\displaystyle D_{\text{KL}}(\lambda P_{1}+(1-\lambda )P_{2}\parallel \lambda Q_{1}+(1-\lambda )Q_{2})\leq \lambda D_{\text{KL}}(P_{1}\parallel Q_{1})+(1-\lambda )D_{\text{KL}}(P_{2}\parallel Q_{2}){\text{ for }}0\leq \lambda \leq 1.}$

### [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) 

Instead of building a model, Q(X), to satisfy P(X) directly. The insight is that we can generate data by sampling, and estimate distributions 



The insight is that you could use noise to prevent "overfitting" the model to the data. 

 could be many possible models, Q, that could satisfy 

 




instead of creating a model Q directly from P, which could lead to memorizing 

 KL divergence still holds when add noise.


KL divergence is just a measure of discrepancy between two distributions, but in diffusion models we can have Q model P without directly fitting on the data.

add the Markov assumption to factorize the path distribution:

```math
D_{\mathrm{KL}}\bigl(q(x_{0:T}) \parallel p_\theta(x_{0:T})\bigr)
= \mathbb{E}_{q}\!\Biggl[\log \frac{q(x_{0:T})}{p_\theta(x_{0:T})}\Biggr]
```

Using the Markov factorization:

```math
q(x_{0:T}) = q(x_0)\prod_{t=1}^{T} q(x_t \mid x_{t-1}), \quad
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)
```

we expand:

```math
\log \frac{q(x_{0:T})}{p_\theta(x_{0:T})}
= \log q(x_0) - \log p(x_T) + \sum_{t=1}^{T} \log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}
```

Taking expectation under $q$ gives a sum of per-step KL terms:

```math
D_{\mathrm{KL}}\bigl(q(x_{0:T}) \parallel p_\theta(x_{0:T})\bigr)
= \underbrace{\mathbb{E}_q[\log q(x_0) - \log p(x_T)]}_{\text{constant wrt $\theta$}}
+ \sum_{t=1}^{T} \mathbb{E}_q\!\Bigl[\log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}\Bigr]
```

Training therefore matches each reverse conditional $p_\theta(x_{t-1}\mid x_t)$ to the true forward conditional $q(x_t\mid x_{t-1})$, step by step.

### Reverse Posterior and Learned Reverse Transition (Explicit)

**Definitions.** Let $\beta_t\in(0,1)$ be the noise schedule, $\alpha_t := 1-\beta_t$, and $\bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s$. Let $I$ denote the identity matrix and $\mathbf{x}_t\in\mathbb{R}^d$.

**Forward (noising).**
```math
q(\mathbf{x}_t\mid \mathbf{x}_{t-1}) 
= \mathcal{N}\!\Bigl(\mathbf{x}_t; \sqrt{\alpha_t}\,\mathbf{x}_{t-1},\; \beta_t I\Bigr)
```
Marginal under the data $\mathbf{x}_0$:
```math
q(\mathbf{x}_t\mid \mathbf{x}_0) 
= \mathcal{N}\!\Bigl(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t) I\Bigr)
```

**True reverse (posterior of the forward chain).** This is the distribution the model must approximate at training time:
```math
q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0) 
= \mathcal{N}\!\Bigl(\mathbf{x}_{t-1};\; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0),\; \tilde{\beta}_t I\Bigr)
```
with
```math
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0) 
= \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\mathbf{x}_0
\; +\; \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t,
\qquad
\tilde{\beta}_t 
= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t.
```

**Learned reverse transition.** We parameterize a Markov chain
```math
p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t) 
= \mathcal{N}\!\Bigl(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\; \sigma_t^2 I\Bigr),
```
with common choices $\sigma_t^2 = \tilde{\beta}_t$ (fixed) or learned.

**$\varepsilon$-parameterization.** The network predicts the forward noise $\varepsilon_\theta(\mathbf{x}_t,t)$, giving
```math
\hat{\mathbf{x}}_0(\mathbf{x}_t,t;\theta) 
= \frac{1}{\sqrt{\bar{\alpha}_t}}\Bigl(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta(\mathbf{x}_t,t)\Bigr),
```
```math
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t) 
= \frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(\mathbf{x}_t,t)\Bigr).
```
With these choices, the per-step KL simplifies (up to constants and scalar weights $w_t$) to the familiar **noise-prediction MSE**:
```math
\mathrm{KL}\bigl(q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)\,\|\, p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)\bigr)
\propto \; \mathbb{E}_{\mathbf{x}_0,\varepsilon, t}\bigl[\, w_t\,\|\varepsilon - \varepsilon_\theta(\mathbf{x}_t,t)\|_2^2\,\bigr].
```

**Forward vs. Reverse (at a glance).**

| Direction | Joint factorization | One-step transition | Posterior needed for training |
|---|---|---|---|
| Forward $q$ | $q(\mathbf{x}_{0:T})=q(\mathbf{x}_0)\prod_{t=1}^{T} q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ | $\mathcal{N}(\sqrt{\alpha_t}\,\mathbf{x}_{t-1}, \beta_t I)$ | $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t I)$ |
| Reverse $p_\theta$ | $p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ | $\mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t,t), \sigma_t^2 I)$ | matched to the forward posterior via per-step KL |