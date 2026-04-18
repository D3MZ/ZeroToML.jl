Regression most simply uses the Ordinary Least Squares (OLS) method, where it aims to minimizing the square differences between points and a curve: 
$$
\begin{aligned}
\min_{\beta}\ \sum_{i=1}^{n}\big(y_i - X_i\beta\big)^2 \\
\Rightarrow\ \hat{\beta} = (X^{\mathsf T}X)^{-1}X^{\mathsf T}y
\end{aligned}
$$


This is matrix form, so $X$ can also contain a column of 1s to create intercepts:
$$X =
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1p} \\
1 & x_{21} & x_{22} & \dots & x_{2p} \\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_{n1} & x_{n2} & \dots & x_{np}
\end{bmatrix}$$

$$\beta =
\begin{bmatrix}
\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p
\end{bmatrix}$$

with $\beta_0$ the intercept.

${\displaystyle \mathbf {X} ^{\mathsf {T}}\mathbf {X} }$ if singular is invertible. This means it doesn't have a full set of linearly independent columns — in other words, the matrix loses information during calculations, making standard solutions impossible. This matters because invertibility is required for many statistical procedures to work reliably.

Ridge regression (L2) solves the problem of a  moment matrix  is alleviated by adding positive elements to the diagonals, thereby decreasing its condition number.

adds a slope penalty scaled by $\lambda$.
$$\min_{\beta}\ \sum_{i=1}^n (y_i - X_i\beta)^2\ +\ \lambda \|\beta\|_2^2$$

This works when columns aren't guaranteed to be independent, 
${\displaystyle {\hat {\boldsymbol {\beta }}}_{R}=\left(\mathbf {X} ^{\mathsf {T}}\mathbf {X} +\lambda \mathbf {I} \right)^{-1}\mathbf {X} ^{\mathsf {T}}\mathbf {y} }$