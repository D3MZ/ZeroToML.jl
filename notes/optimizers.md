# Optimizers
Optimizers control how model parameters $\theta$ are updated during training so that the loss function $J(\theta)$ is minimized (or maximized, depending on the task).

At each step t, an optimizer takes:
- Current parameters $\theta_t$,
- Gradient of the loss with respect to parameters $g_t = \nabla_\theta J(\theta_t)$,
- Past state (like accumulated momentum or adaptive statistics)

| Optimizer | Update Rule (schematic) | Key Idea | Reference |
|-----------|-------------------------|----------|-----------|
| Momentum  | $$\begin{aligned} v &\leftarrow \gamma v + \eta \nabla_\theta J(\theta) \\ \theta &\leftarrow \theta - v \end{aligned}$$ | Accelerates SGD by accumulating a velocity vector in directions of persistent reduction | [Polyak, 1964](https://doi.org/10.1016/0041-5553(64)90137-5) |
| SGD       | $$\theta \leftarrow \theta - \eta \nabla_\theta J(\theta)$$ | Basic gradient descent with fixed learning rate | [Bottou, 2010](https://leon.bottou.org/publications/pdf/sgd-tricks-2012.pdf) |
| AdaGrad   | $$\begin{aligned} r &\leftarrow r + (\nabla_\theta J(\theta))^2 \\ \theta &\leftarrow \theta - \frac{\eta}{\sqrt{r + \epsilon}} \nabla_\theta J(\theta) \end{aligned}$$ | Adapts learning rates per parameter based on historical gradient magnitude | [Duchi et al., 2011](https://jmlr.org/papers/v12/duchi11a.html) |
| RMSProp   | $$\begin{aligned} r &\leftarrow \gamma r + (1-\gamma)(\nabla_\theta J(\theta))^2 \\ \theta &\leftarrow \theta - \frac{\eta}{\sqrt{r + \epsilon}} \nabla_\theta J(\theta) \end{aligned}$$ | Uses exponentially weighted average of squared gradients to adapt learning rates | [Tieleman & Hinton, 2012](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) |
| Adam      | $$\begin{aligned} m &\leftarrow \beta_1 m + (1-\beta_1) \nabla_\theta J(\theta) \\ v &\leftarrow \beta_2 v + (1-\beta_2) (\nabla_\theta J(\theta))^2 \\ \hat{m} &= \frac{m}{1-\beta_1^t} \\ \hat{v} &= \frac{v}{1-\beta_2^t} \\ \theta &\leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \hat{m} \end{aligned}$$ | Combines momentum and RMSProp with bias correction for adaptive learning rates | [Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980) |

| **Symbols** | **Definition** | **Origin/Context** |
|-----------------|---------------|-------------------|
| $\theta$ | model parameters | Standard notation in optimization for parameters. |
| $J(\theta)$ | loss (objective) function | "J" from control/optimization literature, stands for cost or performance index. |
| $g_t = \nabla_\theta J(\theta_t)$ | gradient of the loss at step $t$ | Gradient notation from calculus/optimization. |
| $\eta$ | learning rate | Greek letter eta, standard symbol for learning rate in optimization. |
| $v$ | momentum accumulator (velocity) | From physics analogy, velocity in momentum methods. |
| $\gamma$ | momentum decay (between 0 and 1) | Common symbol for decay factor in control/optimization. |
| $r$ | accumulated squared gradients (AdaGrad/RMSProp) | Used in AdaGrad/RMSProp for accumulated squared gradients. |
| $\beta_1, \beta_2$ | exponential decay rates for first and second moments (Adam) | Conventional Greek letters for exponential decay rates in Adam. |
| $m$ | first moment (momentum) estimate in Adam | First moment estimate, "m" stands for mean (expected value). |
| $\hat{m}$ | bias-corrected first moment | Bias-corrected mean, notation with hat indicates an estimator. |
| $\hat{v}$ | bias-corrected second moment | Bias-corrected variance, hat indicates estimator. |
| $\epsilon$ | small constant to avoid division by zero | Standard use in numerical analysis for small constant to avoid division by zero. |
| $t$ | time step | Standard notation for discrete time step in iterative algorithms. |