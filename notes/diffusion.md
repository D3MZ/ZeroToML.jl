# Diffusion
### [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) 

Since $p_\theta(x_{0})$ is intractable, diffusion models efficiently address this by using:
- Gaussian transitions with a variance schedule $\beta_t$.
- A Markov forward process $q(x_t \mid x_{t-1})$.

This is clever in two ways:
1. Instead of modeling the data distribution $p(x_0)$ directly, diffusion models fix a simple forward noising process $q(x_{1:T}\mid x_0)=\prod_{t=1}^T q(x_t\mid x_{t-1})$ (Gaussian, Markov) and train a network so that each reverse step $p_\theta(x_{t-1}\mid x_t)$ approximates the true posterior $q(x_{t-1}\mid x_t,x_0)$. Once trained, you can start from Gaussian $x_T\sim\mathcal N(0,I)$ and apply the learned reverse steps to generate $x_0$.
2. Because the reverse chain is Markov, sampling requires only the current $x_t$ to draw $x_{t-1}$; there is no need to keep the entire latent path.


### The Forward Diffusion Process

The forward process, $q(x_{1:T}|x_0)$, or diffusion process, is a fixed Markov chain that gradually adds Gaussian noise to the data $x_0$ over $T$ timesteps.

#### Forward Transition
The fixed transitions of the forward process are Gaussian, governed by a variance schedule $\beta_1, \ldots, \beta_T$:
$$\mathbf{q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})}$$

#### Sampling at Arbitrary Timestep $t$
A crucial property of the forward process is that it allows sampling $x_t$ at any arbitrary timestep $t$ in closed form, conditioned only on the initial data point $x_0$. This relies on the definitions $\alpha_t := 1-\beta_t$ and $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$:
$$\mathbf{q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I})}$$

#### Forward Process Posterior
The model formulation also requires the posterior distribution of the previous state $x_{t-1}$ given the current state $x_t$ and the original data $x_0$. This is tractable and is defined as a Gaussian distribution $q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t\mathbf{I})$. The mean $\tilde{\mu}_t(x_t, x_0)$ and variance $\tilde{\beta}_t$ are defined as:

$$\mathbf{\tilde{\mu}_t(x_t, x_0) := \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t}$$
$$\mathbf{\tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t}$$

***

### The Reverse Sampling Process

The reverse process, $p_\theta(x_{0:T})$, is a Markov chain with learned transitions, $p_\theta(x_{t-1}|x_t)$, defined as conditional Gaussians. This process starts from the standard normal prior $p(x_T) = \mathcal{N}(x_T; 0, \mathbf{I})$ and learns to reverse the diffusion process to produce a sample $x_0$.

#### Reverse Transition
The general learned reverse transition is given by:
$$\mathbf{p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))}$$

#### Reparameterization for Learning (Noise Prediction)
The best results are achieved when the mean $\mu_\theta(x_t, t)$ is parameterized to predict the noise $\epsilon$ that corrupted $x_0$ to form $x_t$. Using a fixed isotropic variance $\Sigma_\theta(x_t, t) = \sigma_t^2 \mathbf{I}$, the mean is defined based on a function approximator $\epsilon_\theta(x_t, t)$ intended to predict the noise $\epsilon$:

$$\mathbf{\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)}$$

#### Sampling Step
The actual sampling step to compute $x_{t-1}$ from $x_t$ involves the learned mean and an added noise term $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ (for $t > 1$):
$$\mathbf{x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t \mathbf{z}}$$

***

### Training Objective

Training is performed by optimizing a variational bound on the negative log likelihood. However, the best sample quality results were obtained by optimizing a simplified, weighted variant of this bound, $L_{\text{simple}}$. This simplified objective resembles denoising score matching.

$$\mathbf{L_{\text{simple}}(\theta) := \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \right\|^2 \right]}$$

This objective minimizes the squared error between the true noise $\epsilon$ and the predicted noise $\epsilon_\theta$ applied to the noisy data point $x_t$ (where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$).

## Background
### Definitions
- A latent variable model (LVM) is a probabilistic model that introduces an unobserved random variable z and defines a joint density $p_\theta(x,z)=p_\theta(z)\,p_\theta(x\mid z)$, so that the observable density is $p_\theta(x)=\int p_\theta(x,z)\,dz$ .
    - An MLP is usually deterministic. A standard feed-forward MLP implements a deterministic map
$f_\theta:\mathbb{R}^{d_{\mathrm{in}}}\to \mathbb{R}^{d_{\mathrm{out}}}, y=f_\theta(x)$ with no latent random variable. There is no $\int p_\theta(x,z)\,dz$ or posterior inference.







Suppose $P = \{p_{1}, \ldots, p_{n}\}$ and $Q = \{q_{1}, \ldots, q_{n}\}$ are discrete probability distributions. 

The [Gibbs' inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality), derived from [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality#Probabilistic_form), states the [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)#) H(P) is less than or equal to cross entropy H(P, Q). 
```math
H(P) \;\leq\; H(P, Q)
```
```math
-\sum_{i=1}^{n} p_{i} \log p_{i} \;\leq\; -\sum_{i=1}^{n} p_{i} \log q_{i}
```
This [Evidence Lower BOund (ELBO)](https://en.wikipedia.org/wiki/Evidence_lower_bound) is not much of a guarantee. Rearranging such that $H(P, Q) - H(P) \;\geq\; 0$ produces the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence). This [divergence](https://en.wikipedia.org/wiki/Divergence_(statistics)) does show however that as the gap between P and Q narrows, the distributions must be getting similar because As Q moves closer to P, each ratio $p_i/q_i \to 1$, so the logarithm $\log(p_i/q_i) \to 0$. Therefore the sum shrinks to 0: 

```math
D_{\mathrm{KL}}(P \parallel Q) \;\equiv\; \sum_{i=1}^{n} p_{i} \log \frac{p_{i}}{q_{i}} \;\geq\; 0
```
This has a few useful properties:
1. The divergence is a real number $\ge 0$. 
2. P $=$ Q only when there's there's no divergence anywhere: $p_i - q_i = 0$.
3. As aformentioned, the ELBO implies as the gap P(X) and Q(X) gets smaller, then Q -> P. This is used as a loss function, because reducing the expected [surprisal](https://en.wikipedia.org/wiki/Information_content) implies the model is gaining information.
4. $D_{\mathrm{KL}}(P \parallel Q) = \infty$ if $p_i > 0,\ q_i = 0$ is anywhere. We'll earmark this for now, this is somewhat problematic as a loss function.
5. Additive for independent distributions: 
${\displaystyle D_{\text{KL}}(P\parallel Q)=D_{\text{KL}}(P_{1}\parallel Q_{1})+D_{\text{KL}}(P_{2}\parallel Q_{2}).}$
6. [Convex](https://en.wikipedia.org/wiki/Convex_function) in the pair of probability measures: ${\displaystyle (P,Q)}$, i.e. if ${\displaystyle (P_{1},Q_{1})}$ and $ {\displaystyle (P_{2},Q_{2})}$ are two pairs of probability measures then ${\displaystyle D_{\text{KL}}(\lambda P_{1}+(1-\lambda )P_{2}\parallel \lambda Q_{1}+(1-\lambda )Q_{2})\leq \lambda D_{\text{KL}}(P_{1}\parallel Q_{1})+(1-\lambda )D_{\text{KL}}(P_{2}\parallel Q_{2}){\text{ for }}0\leq \lambda \leq 1.}$