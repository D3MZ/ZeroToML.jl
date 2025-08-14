## Summary

1. [Tokenization](#1-tokenization) — Converts input text into a sequence of integer token IDs.  
2. [Input Embedding](#2-input-embedding) — Maps tokens to dense vectors.  
3. [Positional Encoding](#3-positional-encoding) — Adds position information to embeddings.  
4. [Linear Projections for Multi-Head Attention](#4-linear-projections-for-multi-head-attention) — Projects embeddings into query, key, and value matrices for each head.  
5. [Scaled Dot-Product Attention](#5-scaled-dot-product-attention) — Computes attention weights and weighted sums of values.  
6. [Concatenate Heads & Project](#6-concatenate-heads--project) — Merges outputs from all attention heads and projects back to model dimension.  
7. [Add & LayerNorm (Post-Attention)](#7-add--layernorm-post-attention) — Adds residual connection and normalizes post-attention output.  
8. [Position-wise Feed-Forward Network](#8-position-wise-feed-forward-network) — Applies a two-layer feed-forward network to each position independently.  
9. [Add & LayerNorm (Post-FFN)](#9-add--layernorm-post-ffn) — Adds residual connection and normalizes post-FFN output.  
10. [Final Projection & Softmax](#10-final-projection--softmax) — Maps hidden states to vocabulary logits and converts to probabilities.  

| Symbol | Definition |
|--------|------------|
| $n$ | String (sequence) length |
| $V$ | Vocabulary size |
| $d_{\mathrm{model}}$ | Model (embedding) dimension |
| $d_k$ | Attention head dimension |
| $d_{\mathrm{ff}}$ | Feed-forward hidden dimension |
| $t_i$ | Token index at position $i$, integer in $\{0, \dots, V-1\}$ |
| $E$ | Embedding matrix, $\mathbb{R}^{V \times d_{\mathrm{model}}}$ |
| $X$ | Token embedding matrix for input sequence, $\mathbb{R}^{n \times d_{\mathrm{model}}}$ |
| $P$ | Positional encoding matrix, $\mathbb{R}^{n \times d_{\mathrm{model}}}$ |
| $Z^{(0)}$ | Combined input (token embeddings + positional encodings), $\mathbb{R}^{n \times d_{\mathrm{model}}}$ |
| $W^Q, W^K, W^V$ | Query, Key, and Value projection matrices, $\mathbb{R}^{d_{\mathrm{model}} \times d_k}$ |
| $Q, K, V$ | Projected queries, keys, and values, $\mathbb{R}^{n \times d_k}$ |
| $S$ | Attention score matrix, $\mathbb{R}^{n \times n}$ |
| $A$ | Attention weight matrix after softmax, $\mathbb{R}^{n \times n}$ |
| $O_h$ | Output from attention head $h$, $\mathbb{R}^{n \times d_k}$ |
| $O$ | Concatenated output from all heads, $\mathbb{R}^{n \times (H \cdot d_k)}$ |
| $W^O$ | Output projection matrix after concatenating heads, $\mathbb{R}^{(H \cdot d_k) \times d_{\mathrm{model}}}$ |
| $W_1, W_2$ | Feed-forward layer weights |
| $b_1, b_2$ | Feed-forward layer biases |
| $W^{\mathrm{out}}$ | Final projection matrix to vocabulary logits, $\mathbb{R}^{d_{\mathrm{model}} \times V}$ |


## 1. Tokenization

A tokenizer chunks a datastream, maps those segments to integer IDs, and saves into a lookup table.

- Text:
  - Character-level: each character gets an ID. ([Zhang et al., 2015](https://arxiv.org/abs/1508.06615))
  - Word-level: each word gets an ID. ([Brown et al., 1996](https://aclanthology.org/P96-1041))
  - Subword (BPE, WordPiece, SentencePiece): breaks words into smaller units, each with an ID. ([Sennrich et al., 2016](https://aclanthology.org/P16-1162))
  - Byte-level: each byte of UTF-8 text gets an ID (used in GPT-2/3/4 tokenizers). ([OpenAI GPT-2 BPE](https://openai.com/research/bpe))

- Image:
  - ViT patches: split image into fixed-size patches and embed each ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))
  - VQGAN tokens: quantize image features into discrete tokens ([Esser et al., 2021](https://arxiv.org/abs/2012.09841))

- Video: TimeSformer: split into spatio-temporal patches ([Bertasius et al., 2021](https://arxiv.org/abs/2102.05095))

- Audio: VQ-VAE: discretize spectrogram or waveform into codebook indices ([van den Oord et al., 2017](https://arxiv.org/abs/1711.00937))

- Time series: TS2Vec: discretize or embed continuous readings ([Yue et al., 2021](https://arxiv.org/abs/2106.10466))

- 3D geometry: Point-BERT: discretize point cloud into discrete tokens ([Yu et al., 2022](https://arxiv.org/abs/2111.14819))

## 2. Input Embedding

Each token is then mapped to a dense vector forming a trainable embedding matrix ($E \in \mathbb{R}^{V \times d_{\mathrm{model}}}$).

The embedding matrix $E$ is learned during training to represent token semantics, allowing the model to capture meaningful relationships between tokens.

## 3. Positional Encoding

The embeddings are combined with positional encodings ($P \in \mathbb{R}^{n \times d_{\mathrm{model}}}$). 

Types of Positional Encodings:

- Fixed sinusoidal encodings ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)): Each position $i$ and dimension $2k$ is encoded as:
$$
P_{i, 2k} = \sin\left(\frac{i}{10000^{2k / d_{\text{model}}}}\right), \quad
P_{i, 2k+1} = \cos\left(\frac{i}{10000^{2k / d_{\text{model}}}}\right)
$$
- Learned absolute position embeddings ([Devlin et al., 2019](https://arxiv.org/abs/1810.04805)): Each position $i$ has a learned vector $p_i \in \mathbb{R}^{d_{\text{model}}}$ updated through training, allowing the model to adapt position representations to the task.
- Relative position encodings ([Shaw et al., 2018](https://arxiv.org/abs/1803.02155)): Positions are encoded relative to each other within the attention mechanism, letting the model focus on pairwise distances rather than absolute positions.

The token embeddings and positional encodings are combined via element-wise addition: $Z^{(0)} = X + P$. This has two benefits:
1. The linearity between $X$ and $Z$, and $P$ and $Z$ preserves separate contributions of $X$ and $P$ in the dot products from the "Attention" operation. 
2. Since the addition is element-wise, the combined input $Z^{(0)}$ retains the same shape as both the token embeddings and positional encodings, preserving the sequence length $n$ and model dimension $d_{\mathrm{model}}$.

## 4. Linear Projections for Single Head Attention

The combined token embeddings and positional encodings $Z^{(0)} = X + P$ are projected into three different vector spaces: queries $Q$, keys $K$, and values $V$.

Given
- $X \in \mathbb{R}^{n \times d_{\mathrm{model}}}$ — token embeddings
- $P \in \mathbb{R}^{n \times d_{\mathrm{model}}}$ — positional encodings
- $Z^{(0)} = X + P$ — combined input

Let
- $d_{\mathrm{model}}$ the model dimension
- $d_k$ the attention head dimension
- $n$ be the sequence length
- $i$ — query position index (row in $Q$)
- $j$ — key position index (column in $K$)


For one head with projection matrices $W^Q, W^K, W^V \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$:
$$
\begin{aligned}
Q &= Z^{(0)} W^Q \\
  &= X W^Q + P W^Q \\
K &= Z^{(0)} W^K \\
  &= X W^K + P W^K
\end{aligned}
$$

Attention scores between positions $i,j$:
```math
S_{ij} = \frac{1}{\sqrt{d_k}} \left[ (x_i W^Q + p_i W^Q)(x_j W^K + p_j W^K)^\top \right]
```
This expands to four terms:
1. content–content: $x_i W^Q (x_j W^K)^\top$
2. content–position: $x_i W^Q (p_j W^K)^\top$
3. position–content: $p_i W^Q (x_j W^K)^\top$
4. position–position: $p_i W^Q (p_j W^K)^\top$

The cross-terms inject **order information** directly into the dot-product attention scores, allowing the model to attend not just by content similarity but also by relative or absolute positions.


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


## 5. Scaled Dot-Product Attention

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


## 6. Concatenate Heads & Project

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


## 7. Add & LayerNorm (Post-Attention)

Add residual connection and apply layer normalization.

- Input: $Z \in \mathbb{R}^{n \times d_{\mathrm{model}}}$ (from previous step)  
- Output:  
$$
Z' = \mathrm{LayerNorm}(Z + O') \quad \in \mathbb{R}^{n \times d_{\mathrm{model}}}
$$  


## 8. Position-wise Feed-Forward Network

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


## 9. Add & LayerNorm (Post-FFN)

Add residual connection from $Z'$ and apply layer normalization.

$$
Z_{\mathrm{out}} = \mathrm{LayerNorm}(Z' + \mathrm{FFN}(Z')) \quad \in \mathbb{R}^{n \times d_{\mathrm{model}}}
$$  


## 10. Final Projection & Softmax

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
