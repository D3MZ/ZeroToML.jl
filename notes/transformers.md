## Table of Contents

1. [Tokenization](#1-tokenization)  
2. [Input Embedding & Positional Encoding](#2-input-embedding--positional-encoding)  
3. [Linear Projections for Multi-Head Attention](#3-linear-projections-for-multi-head-attention)  
4. [Scaled Dot-Product Attention](#4-scaled-dot-product-attention)  
5. [Concatenate Heads & Project](#5-concatenate-heads--project)  
6. [Add & LayerNorm (Post-Attention)](#6-add--layernorm-post-attention)  
7. [Position-wise Feed-Forward Network](#7-position-wise-feed-forward-network)  
8. [Add & LayerNorm (Post-FFN)](#8-add--layernorm-post-ffn)  
9. [Final Projection & Softmax](#9-final-projection--softmax)  


## 1. Tokenization

The input text is converted into a sequence of token indices using a tokenizer.

- Input text: a string  
- Output tokens: $t = (t_1, t_2, \dots, t_n)$ where each $t_i \in \{0, \dots, V-1\}$  
- Dimension: $t \in \mathbb{Z}^n$ (sequence length $n$, vocabulary size $V$)  


## 2. Input Embedding & Positional Encoding

Tokens are mapped to dense vectors via an embedding matrix, then combined with positional encodings.

- Embedding matrix: $E \in \mathbb{R}^{V \times d_{\mathrm{model}}}$  
  The embedding matrix serves as a lookup table that maps each discrete token index to a continuous vector representation. This matrix is learned during training, allowing the model to capture semantic information about tokens in a dense format.

- Token embeddings: $X \in \mathbb{R}^{n \times d_{\mathrm{model}}}, \quad X_i = E_{t_i}$  
  Token embedding lookup involves selecting the rows of the embedding matrix corresponding to the token indices $t_i$ in the input sequence. Each token $t_i$ is thus represented by the embedding vector $E_{t_i}$, resulting in a sequence of embeddings $X$.

- Positional encoding matrix: $P \in \mathbb{R}^{n \times d_{\mathrm{model}}}$  
  Positional encodings inject information about the order of tokens in the sequence, which is not captured by the embeddings themselves. They provide the model with a sense of token position within the sequence.

### Types of Positional Encodings

1. Fixed sinusoidal encodings (Vaswani et al., 2017)

Each position $i$ and dimension $2k$ is encoded as:
$$
P_{i, 2k} = \sin\left(\frac{i}{10000^{2k / d_{\text{model}}}}\right), \quad
P_{i, 2k+1} = \cos\left(\frac{i}{10000^{2k / d_{\text{model}}}}\right)
$$
These values oscillate between -1 and 1, so “later” positions do not have bigger numbers — they’re just at different phases of sine and cosine waves. The frequency changes with the dimension index, so some dimensions change slowly across positions, others rapidly.

2. Learned absolute position embeddings

Each position index has a learnable vector. There’s no guarantee later positions have higher numbers — the training process can set any pattern of values. The “magnitude” of the vector might be larger or smaller depending on learned parameters.

3. Relative position encodings

These don’t store absolute numbers for positions at all — instead they encode offsets between tokens (e.g., “this token is 3 steps to the right of that token”). Values can be positive, negative, or zero depending on direction and offset.


- Combined input: $Z^{(0)} = X + P$  
  The token embeddings and positional encodings are combined via element-wise addition. Because addition is element-wise, every dimension in the embedding is shifted in a way that encodes its position. The loss of separability between X and P is often not harmful, because the network doesn’t need to recover the original X and P separately — it only needs their combined information to predict the next token.

  Some architectures avoid this potential loss by concatenating X and P and then projecting back to $d_{\text{model}}$, or by using relative position encodings in the attention computation instead of adding P directly.

- Dimension: $Z^{(0)} \in \mathbb{R}^{n \times d_{\mathrm{model}}}$  
  Since the addition is element-wise, the combined input $Z^{(0)}$ retains the same shape as both the token embeddings and positional encodings, preserving the sequence length $n$ and model dimension $d_{\mathrm{model}}$.


## 3. Linear Projections for Multi-Head Attention

For each attention head $h = 1, \dots, H$, compute queries, keys, and values by linear projection of $Z$.

- Input: $Z \in \mathbb{R}^{n \times d_{\mathrm{model}}}$  
- Projection matrices for head $h$:  
  - $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$  
- Projected matrices:  
$$
Q_h = Z W_h^Q, \quad K_h = Z W_h^K, \quad V_h = Z W_h^V
$$  
- Dimensions:  
  - $Q_h, K_h, V_h \in \mathbb{R}^{n \times d_k}$  


## 4. Scaled Dot-Product Attention

Compute attention scores, apply masking (optional), normalize with softmax, and compute weighted sum of values.

- Score matrix:  
$$
S_h = \frac{Q_h K_h^\top}{\sqrt{d_k}} \quad \in \mathbb{R}^{n \times n}
$$  
- Optional masking:  
$$
S_h \leftarrow S_h + \log M
$$  
where $M \in \mathbb{R}^{n \times n}$ is a mask matrix with zeros or $-\infty$ values.  
- Attention weights:  
$$
A_h = \mathrm{softmax}(S_h) \quad \in \mathbb{R}^{n \times n}
$$  
- Output of head:  
$$
O_h = A_h V_h \quad \in \mathbb{R}^{n \times d_k}
$$  


## 5. Concatenate Heads & Project

Concatenate outputs from all heads and project back to model dimension.

- Concatenation:  
$$
O = \mathrm{Concat}(O_1, \dots, O_H) \quad \in \mathbb{R}^{n \times (H \cdot d_k)}
$$  
- Output projection matrix:  
$$
W^O \in \mathbb{R}^{(H \cdot d_k) \times d_{\mathrm{model}}}
$$  
- Projected output:  
$$
O' = O W^O \quad \in \mathbb{R}^{n \times d_{\mathrm{model}}}
$$  


## 6. Add & LayerNorm (Post-Attention)

Add residual connection and apply layer normalization.

- Input: $Z \in \mathbb{R}^{n \times d_{\mathrm{model}}}$ (from previous step)  
- Output:  
$$
Z' = \mathrm{LayerNorm}(Z + O') \quad \in \mathbb{R}^{n \times d_{\mathrm{model}}}
$$  


## 7. Position-wise Feed-Forward Network

Apply two affine transformations with an activation function in between, independently at each position.

- Parameters:  
  - $W_1 \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}$, $b_1 \in \mathbb{R}^{d_{\mathrm{ff}}}$  
  - $W_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}$, $b_2 \in \mathbb{R}^{d_{\mathrm{model}}}$  
- Feed-forward operation at each position $i$:  
$$
\mathrm{FFN}(Z'_i) = \max(0, Z'__i W_1 + b_1) W_2 + b_2
$$  
- Output:  
$$
\mathrm{FFN}(Z') \in \mathbb{R}^{n \times d_{\mathrm{model}}}
$$  


## 8. Add & LayerNorm (Post-FFN)

Add residual connection from $Z'$ and apply layer normalization.

$$
Z_{\mathrm{out}} = \mathrm{LayerNorm}(Z' + \mathrm{FFN}(Z')) \quad \in \mathbb{R}^{n \times d_{\mathrm{model}}}
$$  


## 9. Final Projection & Softmax

Map the final hidden states to vocabulary logits and convert to probabilities.

- Output projection matrix:  
$$
W^{\mathrm{out}} \in \mathbb{R}^{d_{\mathrm{model}} \times V}
$$  
- Logits for each token position:  
$$
\mathrm{logits} = Z_{\mathrm{out}} W^{\mathrm{out}} \quad \in \mathbb{R}^{n \times V}
$$  
- Probability distribution over vocabulary:  
$$
\mathrm{probs} = \mathrm{softmax}(\mathrm{logits}) \quad \in \mathbb{R}^{n \times V}
$$  
where softmax is applied row-wise to produce valid probability distributions for each token position.
