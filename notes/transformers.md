## Table of Contents

- [Input Embedding & Positional Encoding](#input-embedding--positional-encoding)
- [Multi-Head Self-Attention](#multi-head-self-attention)
  - [Linear projections](#linear-projections)
  - [Scaled dot-product attention](#scaled-dot-product-attention)
  - [Concatenate heads & project](#concatenate-heads--project)
- [Add & LayerNorm](#add--layernorm)
- [Position-wise Feed-Forward Network](#position-wise-feed-forward-network)
- [Another Add & LayerNorm](#another-add--layernorm)

## Input Embedding & Positional Encoding

Given:
- Tokens $t_1, t_2, \dots, t_n$
- Embedding matrix $E \in \mathbb{R}^{V \times d_{\mathrm{model}}}$

Embedding lookup:

$$
X \in \mathbb{R}^{n \times d_{\mathrm{model}}}, \quad X_i = E_{t_i}
$$

Addition (positional encoding $P$):

$$
Z^{(0)} = X + P
$$


## Multi-Head Self-Attention

### Linear projections

For each head $h$:

$$
Q_h = Z W_h^Q,\quad K_h = Z W_h^K,\quad V_h = Z W_h^V
$$

where $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$.

Operations: matrix multiplication.


### Scaled dot-product attention

Score matrix:

$$
S_h = \frac{Q_h K_h^\top}{\sqrt{d_k}}
$$

Operations: matrix multiplication, scalar division.

Masking (optional):

$$
S_h \leftarrow S_h + \log M
$$

Operation: addition.

Softmax normalization:

$$
A_h = \mathrm{softmax}(S_h)
$$

Operations: element-wise exponentiation, summation, division.

Weighted sum of values:

$$
O_h = A_h V_h
$$

Operation: matrix multiplication.


### Concatenate heads & project

$$
O = \mathrm{Concat}(O_1, \dots, O_H) W^O
$$

Operations: concatenation, matrix multiplication.


## Add & LayerNorm

$$
Z' = \mathrm{LayerNorm}(Z + O)
$$

Operations: element-wise addition, subtraction of mean, division by standard deviation, scaling, and shifting.


## Position-wise Feed-Forward Network

Two affine maps with activation (usually ReLU or GELU):

$$
\mathrm{FFN}(x) = \max(0,\, x W_1 + b_1) W_2 + b_2
$$

Operations: matrix multiplications, bias additions, and an activation function.


## Another Add & LayerNorm

$$
Z_{\mathrm{out}} = \mathrm{LayerNorm}(Z' + \mathrm{FFN}(Z'))
$$

Operations: element-wise addition, subtraction of mean, division by standard deviation, scaling, and shifting.
