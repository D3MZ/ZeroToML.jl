# Gaussian Processes

Gaussian process regression models a latent function $f$ with a Gaussian process prior and conditions on observed data to produce a posterior over functions. The process is fully specified by its mean function $m(x)$ and kernel $\kappa(x, x')$.

## Prior

We place a zero-mean GP prior on the latent function:
$$
f \sim \mathcal{GP}\left(0,\ \kappa(x, x')\right).
$$

In this project we use the squared-exponential (radial basis function) kernel:
$$
\kappa(x, x') = \sigma^2 \exp\left(-\frac{1}{2} \left\| \frac{x - x'}{\ell} \right\|^2 \right),
$$
where $\ell$ controls correlation length and $\sigma^2$ is the signal variance.

Observations are assumed to be noisy evaluations of \( f \):
$$
y_i = f(x_i) + \epsilon_i,\quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2).
$$

## Conditioning on Data

Collect the inputs $X = [x_1, \ldots, x_n]^\top$ and outputs $y = [y_1, \ldots, y_n]^\top$. Construct the covariance matrix:
$$
K = \kappa(X, X) + \sigma_n^2 I.
$$

We factorize $K$ with the Cholesky decomposition:
$$
LL^\top = K,
$$
and solve for
$$
\alpha = K^{-1} y = (LL^\top)^{-1} y
$$
via triangular solves:
$$
L u = y,\quad L^\top \alpha = u.
$$

These quantities are cached for prediction.

## Posterior Prediction

For test inputs $X_*$, compute cross-covariances and prior covariances:
$$
K_* = \kappa(X_*, X),\qquad K_{**} = \kappa(X_*, X_*).
$$

The posterior mean and covariance are:
$$
\mu_* = K_* \alpha,
$$
$$
\Sigma_* = K_{**} - K_* K^{-1} K_*^\top = K_{**} - (L^{-1} K_*^\top)^\top (L^{-1} K_*^\top).
$$

The diagonal of $\Sigma_*$ yields the predictive variances, and its square roots give predictive standard deviations.
