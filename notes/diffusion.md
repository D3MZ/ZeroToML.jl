# Diffusion
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


