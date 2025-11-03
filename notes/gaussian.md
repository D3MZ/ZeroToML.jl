You can have all sorts of wacky functions that'll perfectly interpolate data because nothing constrains the behavior between the known points. You may draw a straight line between two points, but nothing is stoping you from connecting those two points by first going to infinity and then back.

A Gaussian Process (GP) places a probability distribution over functions, which effectively bounds the infinite space of possibilities. 

Given the general form of a probability density function is:
$${\displaystyle f(x;\mu ,\sigma ^{2})={\frac {1}{\sqrt {2\pi \sigma ^{2}}}}e^{-{\frac {(x-\mu )^{2}}{2\sigma ^{2}}}}}=\mathcal{N}(x\mid\mu,\sigma^2)$$

A Gaussian Process represents the mean $\mu$ as a function $m(x)$, and the variance $\sigma ^{2}$ as a covariance (kernel) function $k(x, x')$ - which just needs to be symmetric and a PSD in output.


The GP prior $f \sim \mathcal{GP}(m, k)$ encodes assumptions—via the mean function $m(x)$ and covariance (kernel) function $k(x, x’)$—about smoothness, correlation, and scale. Once conditioned on observed data $D = \{(x_i, y_i)\}_{i=1}^n$, the GP produces a posterior distribution that specifies how likely each function is between points, yielding both a mean prediction and uncertainty that reflect what is plausible given the data and the prior.


is an opinionated way of deciding what the ideal function looks like inbetween the datapoints that we know about.


When $y = f(x)$ is too hard to specify directly, we can extend our toolbox by adding Gaussian (normal) noise $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$ that's independent from x. For a linear model we can write

$$f(x) = x^{\top} w, \quad y = f(x) + \varepsilon,$$

where $x$ is the input vector, $w$ is the vector of weights (parameters) of the linear model, $f(x)$ is the function value, and $y$ is the observed target value. 

This reframing yields a likelihood without violating the model: $\mathbb{E}[y\mid x]=f(x)$ because the noise has zero mean.

$$p(\mathbf{y} \mid X, \mathbf{w}) \;=\; \prod_{i=1}^n p(y_i \mid x_i, \mathbf{w}).$$

Because the additive noise is Gaussian with zero mean and variance $\sigma_n^2$, each conditional term is Gaussian; **independence across $i$** is used only to factorize the joint (product form), not to make each term Gaussian:

$$y_i \mid x_i,\mathbf{w} \sim \mathcal{N}(x_i^{\top}\mathbf{w},\,\sigma_n^2).$$

Since for a scalar (univariate) Gaussian random variable \(y\):

$$
\mathcal{N}(y;\mu,\sigma^2)
= \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

Then

$$p(y_i \mid x_i, \mathbf{w}) \;=\; \frac{1}{\sqrt{2\pi\sigma_n^2}} \exp\!\left( -\frac{(y_i - x_i^{\top} \mathbf{w})^2}{2\sigma_n^2} \right).$$

So
$$p(\mathbf{y} \mid X, \mathbf{w})
= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_n^2}}
\exp\!\left( - \frac{(y_i - x_i^{\top} \mathbf{w})^2}{2\sigma_n^2} \right)$$

Simplify the constants

$$\prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma_n^2}}
= \left( \frac{1}{\sqrt{2\pi\sigma_n^2}} \right)^n
= \frac{1}{(2\pi\sigma_n^2)^{n/2}}$$

Simplify the exponential part
$$
\prod_{i=1}^n \exp\!\left( - \frac{(y_i - x_i^{\top} \mathbf{w})^2}{2\sigma_n^2} \right)
= \exp\!\left( - \frac{1}{2\sigma_n^2} \sum_{i=1}^n (y_i - x_i^{\top} \mathbf{w})^2 \right)
$$
$$
= \exp\!\left( - \frac{1}{2\sigma_n^2} \lVert \mathbf{y} - X \mathbf{w} \rVert_2^2 \right)
$$

So now we have
$$
p(\mathbf{y} \mid X, \mathbf{w})
= \frac{1}{(2\pi\sigma_n^2)^{n/2}}
\exp\!\left(-\frac{\|\mathbf{y}-X\mathbf{w}\|_2^2}{2\sigma_n^2}\right)
= \mathcal{N}\!\big(\mathbf{y};\,X\mathbf{w},\,\sigma_n^2 I_n\big).
$$


---

**Gaussian probability density function (pdf)**

For a scalar (univariate) Gaussian random variable \(y\):

$$
\mathcal{N}(y;\mu,\sigma^2)
= \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

For a vector (multivariate) Gaussian random variable \(\mathbf{y}\in\mathbb{R}^n\):

$$
\mathcal{N}(\mathbf{y};\boldsymbol{\mu},\Sigma)
= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}
\exp\!\left(-\tfrac{1}{2}(\mathbf{y}-\boldsymbol{\mu})^{\top}\Sigma^{-1}(\mathbf{y}-\boldsymbol{\mu})\right)
$$

**Special case — isotropic (independent, equal variance):**
if \(\Sigma = \sigma^2 I_n\),
$$
\mathcal{N}(\mathbf{y};\boldsymbol{\mu},\sigma^2 I_n)
= \frac{1}{(2\pi\sigma^2)^{n/2}}
\exp\!\left(-\frac{\|\mathbf{y}-\boldsymbol{\mu}\|_2^2}{2\sigma^2}\right)
$$

| Symbol | Definition |
|---------|-------------|
| \(\mathbf{y}\) | Observation vector \([y_1,\dots,y_n]^{\top}\) |
| \(\boldsymbol{\mu}\) | Mean vector (e.g., \(Xw\) in linear models) |
| \(\Sigma\) | Covariance matrix |
| \(|\Sigma|\) | Determinant of covariance matrix |
| \(I_n\) | \(n\times n\) identity matrix |
| \(\sigma^2\) | Scalar variance (isotropic case) |

---

Mapping the terms, we can also show the product of independent Gaussians is also a Gaussian:

For independent terms
$$p(y_i \mid x_i, w) = \mathcal{N}(y_i; x_i^{\top} w, \sigma_n^2), \quad i=1,\dots,n,$$
their product is
$$\prod_{i=1}^n p(y_i \mid x_i, w)
  = \prod_{i=1}^n \mathcal{N}(y_i; x_i^{\top} w, \sigma_n^2).$$
Map to the multivariate normal parameters:
- stack observations: $\mathbf{y} = [y_1,\dots,y_n]^{\top}$
- stack means: $\boldsymbol{\mu} = [x_1^{\top} w,\dots,x_n^{\top} w]^{\top} = X w$
- homoscedastic, independent noise ⇒ diagonal covariance: $\Sigma = \sigma_n^2 I_n$ (equal variance, zero cross-covariance)

Then by the multivariate Gaussian formula
$$\prod_{i=1}^n \mathcal{N}(y_i; x_i^{\top} w, \sigma_n^2)
  = \mathcal{N}(\mathbf{y};\, X w,\, \sigma_n^2 I_n).$$
This works because the joint density of independent Gaussians is a Gaussian whose mean vector is the stacked univariate means and whose covariance is diagonal with the univariate variances.



### Noise Models

| Noise Model | Equation | Effect |
|-------------|----------|--------|
| **Additive** | $$ y = f(x) + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0,\sigma^2) $$ | Constant-variance, independent noise (homoscedastic) |
| **Multiplicative** | $$ y = f(x)\bigl(1 + \varepsilon\bigr) $$ | Variance scales with signal magnitude |
| **Heteroscedastic** | $$ y = f(x) + \varepsilon(x),\quad \varepsilon(x) \sim \mathcal{N}\bigl(0,\sigma^2(x)\bigr) $$ | Noise variance depends on input \(x\) |
| **Correlated (GP-style)** | $$ \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \Sigma) $$ | Observations have nonzero covariance; needed for GP/regression with correlated errors |


### Covariance Function and GP Properties

#### 1. Covariance Function ⇒ Function Properties

Given  
$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$  

#### Definition of the Gaussian Process

A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. The notation
$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$
means that for any finite set of points $x_1, \ldots, x_n$, the random vector $\mathbf{f} = [f(x_1), \ldots, f(x_n)]^\top$ is distributed as a multivariate normal:
$$
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, \mathbf{K}),
$$
where:
- $m(x)$ is the mean function: $m(x) = \mathbb{E}[f(x)]$
- $k(x, x')$ is the covariance (kernel) function: $k(x, x') = \operatorname{Cov}(f(x), f(x'))$
- $\mathbf{m} = [m(x_1), \ldots, m(x_n)]^\top$
- $\mathbf{K}$ is the $n \times n$ covariance matrix with entries $K_{ij} = k(x_i, x_j)$

#### Joint Gaussian Distribution Definition

For a vector $\mathbf{f} \in \mathbb{R}^n$, the multivariate normal (Gaussian) distribution is defined as:
$$
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, \mathbf{K})
$$
with density
$$
p(\mathbf{f}) = \frac{1}{(2\pi)^{n/2}|\mathbf{K}|^{1/2}} 
\exp\!\left(-\frac{1}{2}(\mathbf{f}-\mathbf{m})^\top \mathbf{K}^{-1}(\mathbf{f}-\mathbf{m})\right)
$$

| Symbol            | Definition                                                                                  |
|-------------------|--------------------------------------------------------------------------------------------|
| $\mathbf{f}$      | $n$-dimensional vector: $\mathbf{f} = [f(x_1), \ldots, f(x_n)]^\top$                       |
| $\mathbf{m}$      | Mean vector: $\mathbf{m} = [m(x_1), \ldots, m(x_n)]^\top$                                  |
| $\mathbf{K}$      | Covariance matrix: $K_{ij} = k(x_i, x_j)$                                                  |
| $n$               | Number of points ($x_1, \ldots, x_n$)                                                      |
| $|\mathbf{K}|$    | Determinant of the covariance matrix $\mathbf{K}$                                          |
| $p(\mathbf{f})$   | Probability density function for $\mathbf{f}$ under $\mathcal{N}(\mathbf{m}, \mathbf{K})$  |


| Property                 | Determined by $k(x,x')$           | Example Kernel               |
|--------------------------|----------------------------------|-----------------------------|
| Smoothness / differentiability | How fast covariance decays with distance | Squared exponential → infinitely smooth |
| Periodicity              | Whether covariance repeats        | Periodic kernel             |
| Stationarity             | Depends only on $x - x'$          | RBF, Matérn                 |
| Non-stationarity         | Depends on $x$ and $x'$ separately | Linear, neural network kernel |
| Amplitude / variance     | Diagonal term $k(x,x)$             | Scaling factor $\sigma^2$   |
| Correlation length / structure | Shape of decay                  | Lengthscale $\ell$          |

#### 2. Why Many Kernels Are Possible

The only mathematical constraint is that $k$ must be positive semi-definite (PSD):  
$$
\forall \{x_i\}, \quad K_{ij} = k(x_i, x_j) \Rightarrow \mathbf{K} \succeq 0
$$  
This allows infinitely many valid covariance structures — any PSD kernel corresponds to a valid GP prior.

You can also combine kernels:  
$$
k_{\text{sum}} = k_1 + k_2, \quad k_{\text{prod}} = k_1 \times k_2
$$  
which yields new induced properties (e.g., periodic + smooth).

#### 2.1 Positive Semi-Definiteness Explained

The notation  
$$
\mathbf{K} \succeq 0
$$  
means that the kernel (covariance) matrix $\mathbf{K}$ is **positive semi-definite (PSD)**.

**Step-by-step explanation:**

1. **Definition of $\mathbf{K}$:**  
   $$
   \mathbf{K} = \begin{bmatrix}
   k(x_1, x_1) & k(x_1, x_2) & \cdots & k(x_1, x_n) \\
   k(x_2, x_1) & k(x_2, x_2) & \cdots & k(x_2, x_n) \\
   \vdots & \vdots & \ddots & \vdots \\
   k(x_n, x_1) & k(x_n, x_2) & \cdots & k(x_n, x_n)
   \end{bmatrix}
   $$

2. **PSD condition:**  
   For any vector $v \in \mathbb{R}^n$,  
   $$
   v^\top \mathbf{K} v \geq 0.
   $$

3. **Intuition:**  
   Positive semi-definiteness ensures that all variances computed from $\mathbf{K}$ are non-negative. This is necessary because variances cannot be negative.

4. **Why it matters for Gaussian Processes:**  
   Since $f(x) \sim \mathcal{N}(0, \mathbf{K})$ is a multivariate Gaussian distribution with covariance $\mathbf{K}$, the PSD property guarantees that this distribution is valid (i.e., $\mathbf{K}$ is a valid covariance matrix).

| Symbol       | Meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| $\mathbf{K}$      | Covariance matrix with entries $K_{ij} = k(x_i, x_j)$                  |
| $\succeq 0$       | Indicates positive semi-definiteness (PSD)                             |
| $v^\top \mathbf{K} v \geq 0$ | For all $v \in \mathbb{R}^n$, quadratic form is non-negative      |
| PSD implication    | Ensures $\mathbf{K}$ is a valid covariance matrix for a Gaussian      |

#### 3. Intuition

- The kernel tells how similar $f(x)$ and $f(x')$ are expected to be.
- Choosing $k = \text{smooth}$ ⇒ functions vary slowly.
- Choosing $k = \text{periodic}$ ⇒ functions oscillate.
- Choosing $k = \text{linear}$ ⇒ functions are straight lines.

Thus, different covariance functions induce different function spaces that the GP “believes” are plausible before seeing data.

### Covariance

| Concept | Formula | Sample Formula | Meaning |
|----------|----------|----------------|----------|
| Variance | $\operatorname{Var}(X) = \mathbb{E}[(X - \mu_X)^2]$ | $\widehat{\operatorname{Var}}(X) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$ | Measures how a single variable spreads around its mean |
| Covariance | $\operatorname{Cov}(X,Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ | $\widehat{\operatorname{Cov}}(X,Y) = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$ | Measures how two variables move together |
| Correlation | $\operatorname{Corr}(X,Y) = \frac{\operatorname{Cov}(X,Y)}{\sigma_X \sigma_Y}$ | $\widehat{\operatorname{Corr}}(X,Y) = \frac{\widehat{\operatorname{Cov}}(X,Y)}{s_X s_Y}$ | Standardized measure of linear relationship between two variables, always in $[-1, 1]$ |

### Expectation 

- $\mathbb{E}[\cdot]$ = expectation: a property of the *true* underlying probability distribution  
- $\mu$ = the (unknown) true mean, often equal to $\mathbb{E}[X]$  
- $\bar{x}$ = sample average from observed data (an estimator of $\mu$)

Relationship (law of large numbers):

$$
\bar{x} \xrightarrow[n \to \infty]{} \mathbb{E}[X]
$$

### Multivariate Covariance Matrix

For $k$ random variables collected into a vector:

$$
\mathbf{X} =
\begin{bmatrix}
X_1 \\ X_2 \\ \vdots \\ X_k
\end{bmatrix}, \quad
\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}] =
\begin{bmatrix}
\mu_1 \\ \mu_2 \\ \vdots \\ \mu_k
\end{bmatrix}.
$$

The covariance matrix is:

$$
\Sigma = \mathbb{E}\!\left[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top\right].
$$

This expands to:

$$
\Sigma =
\begin{bmatrix}
\operatorname{Var}(X_1) & \operatorname{Cov}(X_1,X_2) & \cdots & \operatorname{Cov}(X_1,X_k) \\
\operatorname{Cov}(X_2,X_1) & \operatorname{Var}(X_2) & \cdots & \operatorname{Cov}(X_2,X_k) \\
\vdots & \vdots & \ddots & \vdots \\
\operatorname{Cov}(X_k,X_1) & \operatorname{Cov}(X_k,X_2) & \cdots & \operatorname{Var}(X_k)
\end{bmatrix}.
$$

**Properties:**
- Symmetric: $\Sigma^\top = \Sigma$  
- Positive semi-definite: $v^\top \Sigma v \ge 0 \; \forall v$

**Sample covariance matrix** (for data $A \in \mathbb{R}^{n \times k}$):

$$
\widehat{\Sigma} = \frac{1}{n-1}(A - \mathbf{1}\bar{A})^\top (A - \mathbf{1}\bar{A})
$$

Each element:

$$
\widehat{\Sigma}_{ij} = \frac{1}{n-1}\sum_{t=1}^n (a_{ti} - \bar{a}_i)(a_{tj} - \bar{a}_j)
$$

**Summary Table:**

| Object | Dimension | Definition | Interpretation |
|---------|------------|-------------|----------------|
| $\operatorname{Var}(X_i)$ | scalar | $\mathbb{E}[(X_i - \mu_i)^2]$ | Spread of variable $i$ |
| $\operatorname{Cov}(X_i, X_j)$ | scalar | $\mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$ | Joint variation of $i$ and $j$ |
| $\Sigma$ | $k \times k$ | $\mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]$ | Joint variability of all $k$ variables |

**Geometric interpretation:**
- Defines the elliptical shape of a multivariate distribution.
- For Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$, contours satisfy:
  $$
  (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) = c.
  $$
- Eigenvectors → principal directions of spread.
- Eigenvalues → magnitude of spread along those directions.


### Why divide by $n - k$? (Degrees of Freedom)

When we estimate parameters from data, each estimated parameter imposes a constraint that reduces the number of *free* observations contributing to variability.

In general, the unbiased estimator of variance or covariance divides by:

$$
n - k
$$

where:
- $n$ = number of observations  
- $k$ = number of independent parameters or constraints estimated from data  

---

#### Example 1: Sample Mean (k = 1)

Using the sample mean $\bar{x}$ introduces one constraint:

$$
\sum_{i=1}^n (x_i - \bar{x}) = 0
$$

That removes 1 degree of freedom, giving denominator $n - 1$.

$$
\widehat{\sigma}^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2
$$

---

#### Example 2: Linear Regression (k = p + 1)

For a regression model:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i
$$

There are $p + 1$ estimated parameters (intercept + slopes).  
Residuals:

$$
r_i = y_i - \hat{y}_i
$$

The unbiased estimate of the residual variance uses denominator $n - (p + 1)$:

$$
\widehat{\sigma}^2 = \frac{1}{n - (p+1)} \sum_{i=1}^n r_i^2
$$

---

#### Example 3: General Matrix Form

Let $Y \in \mathbb{R}^{n \times 1}$, $X \in \mathbb{R}^{n \times k}$, and fitted parameters $\hat{\beta} = (X^\top X)^{-1}X^\top Y$.

Then:

$$
\widehat{\sigma}^2 = \frac{1}{n - k}(Y - X\hat{\beta})^\top (Y - X\hat{\beta})
$$

Here, $k$ parameters have been estimated from $n$ data points, leaving $n - k$ effective degrees of freedom.

---

#### Geometric Interpretation

Each estimated parameter removes one dimension from the space of residuals.  
The variance is averaged over the remaining $n - k$ directions:

$$
\text{df} = n - k
$$

For the sample mean, $k = 1$;  
for regression, $k = p + 1$;  
for general models, $k$ equals the number of fitted parameters.

---

**Summary Table**

| Scenario | Parameters Estimated ($k$) | Denominator | Interpretation |
|-----------|---------------------------|--------------|----------------|
| Sample mean | 1 | $n-1$ | Mean estimated → 1 df lost |
| Simple regression | 2 | $n-2$ | Intercept + slope |
| Multiple regression | $p+1$ | $n-(p+1)$ | One df per parameter |
| General model | $k$ | $n-k$ | Subtract fitted parameters |

This generalizes the $n-1$ rule: divide by the number of remaining degrees of freedom after fitting parameters.


#### Vector Convention in Linear Algebra and Machine Learning

In linear algebra and machine learning, vectors are assumed to be **column vectors** by default. This means:

- A column vector $\mathbf{x} \in \mathbb{R}^d$ is represented as a $d \times 1$ matrix.
- Its transpose, $\mathbf{x}^\top \in \mathbb{R}^{1 \times d}$, is a row vector.
- This convention allows matrix multiplication such as $\mathbf{y} = A\mathbf{x}$, where $A$ is a matrix, and the inner product $\mathbf{x}^\top \mathbf{w}$ to yield correctly dimensioned results.
- Some fields, such as statistics or econometrics, sometimes use row vectors as the default, but they adjust shapes accordingly to maintain consistent operations.

| Symbol             | Shape     | Description                            |
|--------------------|-----------|------------------------------------|
| $\mathbf{x}$       | $d \times 1$ | Column vector (features)            |
| $\mathbf{x}^\top$  | $1 \times d$ | Row vector (transpose of $\mathbf{x}$) |
| $\mathbf{w}$       | $d \times 1$ | Column vector (weights)             |
| $\mathbf{x}^\top \mathbf{w}$ | $1 \times 1$ | Scalar (dot product / weighted sum) |

Thus, by default, vectors are treated as columns, and the transpose symbol ensures the correct orientation for inner products and linear transformations.


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




Below's a table of some kernel functions:

| Kernel | $$k(x,x')$$ | Hyperparameters | Stationary | Smoothness / Properties | Typical use |
|---|---|---|---|---|---|
| Constant | $$ \sigma^2 $$ | $\sigma^2 > 0$ | Yes | Flat covariance; adds global offset variance | Baseline variance term |
| White noise | $$ \sigma^2\,\delta_{x,x'} $$ | $\sigma^2 > 0$ | Yes | Independent noise; $\delta$ is Kronecker delta | Observation noise, nugget |
| RBF / Squared‑Exponential | $$ \sigma^2\exp\!\big(-\tfrac{r^2}{2\ell^2}\big) $$ | $\sigma^2,\ \ell > 0$ | Yes | Infinitely mean‑square differentiable (very smooth) | Smooth latent functions |
| RBF (ARD) | $$ \sigma^2\exp\!\big(-\tfrac{1}{2}\sum_j \tfrac{(x_j-x'_j)^2}{\ell_j^2}\big) $$ | $\sigma^2,\ \ell_j > 0$ | Yes | Dimension‑wise lengthscales (relevance) | Automatic relevance determination |
| Matérn $1/2$ (Exponential/OU) | $$ \sigma^2 \exp\!\big(-\tfrac{r}{\ell}\big) $$ | $\sigma^2,\ \ell > 0$ | Yes | Continuous, not differentiable | Rough signals, OU processes |
| Matérn $3/2$ | $$ \sigma^2 \big(1+\tfrac{\sqrt{3}\,r}{\ell}\big)\exp\!\big(-\tfrac{\sqrt{3}\,r}{\ell}\big) $$ | $\sigma^2,\ \ell > 0$ | Yes | Once m.s. differentiable | Moderate smoothness |
| Matérn $5/2$ | $$ \sigma^2 \big(1+\tfrac{\sqrt{5}\,r}{\ell}+\tfrac{5r^2}{3\ell^2}\big)\exp\!\big(-\tfrac{\sqrt{5}\,r}{\ell}\big) $$ | $\sigma^2,\ \ell > 0$ | Yes | Twice m.s. differentiable | Smoother than 3/2, rougher than RBF |
| Rational Quadratic | $$ \sigma^2\big(1+\tfrac{r^2}{2\alpha\ell^2}\big)^{-\alpha} $$ | $\sigma^2,\ \ell > 0,\ \alpha > 0$ | Yes | Scale mixture of RBFs (multi‑scale) | Varying lengthscales |
| Periodic | $$ \sigma^2\exp\!\Big(-\tfrac{2\sin^2(\pi r/p)}{\ell^2}\Big) $$ | $\sigma^2,\ \ell > 0,\ p > 0$ | Yes | Exactly periodic with period $p$ | Seasonal/periodic signals |
| Linear (dot‑product) | $$ \sigma_b^2+\sigma_v^2\,(x-c)^\top(x'-c) $$ | $\sigma_b^2,\ \sigma_v^2,\ c$ | No | Global linear trend; non‑stationary | Trends, regression effects |
| Polynomial (degree $d$) | $$ (x^\top x' + c)^d $$ | $d\in\mathbb{N},\ c\ge 0$ | No | Non‑stationary, grows with $\lVert x\rVert$ | Nonlinear trends of bounded degree |
| Matérn $(\nu)$ | $$ \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}\!\big(\tfrac{\sqrt{2\nu}\,r}{\ell}\big)^\nu K_\nu\!\big(\tfrac{\sqrt{2\nu}\,r}{\ell}\big) $$ | $\sigma^2,\ \ell > 0,\ \nu > 0$ | Yes | $\nu$ controls differentiability; $\nu\!\to\!\infty\Rightarrow$ RBF | Physical systems, roughness control |
| Locally Periodic | $$ \big[\sigma^2\exp\!\Big(-\tfrac{2\sin^2(\pi r/p)}{\ell_p^2}\Big)\big]\times \big[\sigma^2\exp\!\big(-\tfrac{r^2}{2\ell_s^2}\big)\big] $$ | $\sigma^2,\ \ell_p,\ \ell_s,\ p > 0$ | Yes | Periodic but decays with distance | Quasi‑periodic signals |
| Neural Network (arc‑cosine, $q=1$) | $$ \tfrac{\sigma^2}{\pi}\,\lVert x\rVert\,\lVert x'\rVert\,\sin\theta + \tfrac{\sigma^2}{\pi}(\pi-\theta),\ \cos\theta=\tfrac{x^\top x'}{\lVert x\rVert\lVert x'\rVert} $$ | $\sigma^2$ (plus variants) | No | NN covariance; non‑stationary | Deep GP, feature‑like priors |
| Spectral Mixture | $$ \sum_{q=1}^{Q} w_q \exp\!\big(-2\pi^2 (x-x')^\top \Sigma_q (x-x')\big)\cos\!\big(2\pi (x-x')^\top \mu_q\big) $$ | $w_q > 0,\ \mu_q,\ \Sigma_q \succeq 0$ | Yes | Universal stationary kernel; rich spectra | Complex, multi‑scale patterns |

**Notation:** $r=\lVert x-x'\rVert_2$. For the neural-network row, $\theta$ is the angle between $x$ and $x'$ with $\cos\theta=\tfrac{x^\top x'}{\lVert x\rVert\lVert x'\rVert}$.
