# Diffusion

## Background

### Probability
| Concept | Description |
|---|---|
| Joint | $p(x_0, x_1)$ is the joint probability of the pair $(x_0, x_1)$ occuring at the same time (not ordered, such that $p(x_0, x_1) = p(x_1, x_0)$). This can be factorized via Chain Rule: $p(x_0, x_1) = p(x_1)\,p(x_0 \mid x_1) = p(x_0)\,p(x_1 \mid x_0)$. |
| Conditional | $p(x_0 \mid x_1)$ is the probability of $x_0$ given $x_1$, defined as $p(x_0 \mid x_1) = \frac{p(x_0, x_1)}{p(x_1)}$. |
| Marginal | $p(x_0)$ is the probability of $x_0$ regardless of $x_1$, found by summing or integrating over $x_1$: $p(x_0) = \sum_{x_1} p(x_0, x_1)$ or $p(x_0) = \int p(x_0, x_1)\,dx_1$. |
| Gaussian | A Gaussian (normal) distribution has density $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ where $\mu$ is the mean and $\sigma^2$ is the variance. |

#### Discrete Example
Given 
* $p(rain) = 0.3$ (probability of rain is 30%)
* $p(sun) = 1 - p(rain) = 0.7$ (probability of sun is 70%)
* $p(heavy | rain) = 0.8$ (probability of heavy traffic given rain)
* $p(heavy | sun) = 0.3$ (probability of heavy traffic given sun)

Then we can calculate the joint probability of heavy traffic and rain occuring at the same through:
$p(rain, heavy) = p(rain) * p(heavy | rain) = 0.3 * 0.8 = 0.24$.

And we can calculate the marginal probability of it being heavy traffic regardless of weather by summing all the states:

#### Continuous Example

Suppose $x$ and $z$ are jointly Gaussian random variables. Here, $x \sim \mathcal{N}(\mu, \sigma^2)$ means that $x$ follows a normal distribution with mean $\mu$ and variance $\sigma^2$. The variable $z$ is standard normal, i.e., $z \sim \mathcal{N}(0, 1)$. The conditional distribution $p(x \mid z)$ can shift the mean of $x$ based on $z$, for example, $x \mid z \sim \mathcal{N}(\mu + \alpha z, \sigma^2)$, where $\alpha$ controls how $z$ influences $x$. This is why we integrate over $z$ to obtain the marginal distribution of $x$: 

$$
p(x) = \int p(x \mid z)\, p(z)\, dz
$$


Suppose $x \sim \mathcal{N}(\mu, \sigma^2)$ and $z \sim \mathcal{N}(0, 1)$, and the joint density is given by $p(x, z) = p(x \mid z)\, p(z)$, where $p(x \mid z)$ could be, for example, a normal distribution whose mean depends on $z$.

To obtain the marginal distribution $p(x)$, we integrate out $z$:

$$
p(x) = \int p(x \mid z)\, p(z)\, dz
$$

This is how marginalization works for continuous variables: you integrate over the variable you do not care about (here, $z$) to get the marginal distribution for the variable of interest (here, $x$).




$$
\begin{aligned}
p(heavy) 
&= p(rain) \times p(heavy \mid rain) + p(sun) \times p(heavy \mid sun) \\
&= 0.3 \times 0.8 + 0.7 \times 0.3 \\
&= 0.45 \\
\end{aligned}
$$





### Marginal probability










## Papers
[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

A diffusion probabilistic model is a parameterized Markov chain trained using
variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.


Diffusion models are latent variable models of the form $p_\theta(x_0) := p_\theta(x_{0:T}) dx_{1:T}$. Latent variables are hidden random variables that capture underlying structure in the data. They are not directly observed but are inferred during training or sampling. 

Let
* $x \in \mathcal{X}$: observed data (image, text, etc.),
* $z \in \mathcal{Z}$: latent variable (hidden cause),
* $p(z)$: prior distribution on latent variable,
* $p_\theta(x \mid z)$: likelihood of x given z parameterized by $\theta$.

Then the joint distribution is

```math
p_\theta(x, z) = p_\theta(x \mid z) \, p(z)
```

and the marginal is

```math
p_\theta(x) = \int p_\theta(x \mid z) \, p(z) \, dz
```


In probability, marginalization means summing or integrating over variables you do not care about, leaving a probability distribution over the variables you do care about.





* $\prod$ = “product over a range of indices” (just like $\sum$ means sum).


