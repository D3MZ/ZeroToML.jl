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

| Medium   | Method                                                          | Reference                                                        |
|------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------|
| Text       | Character-level: each character gets an ID                                 | [Zhang et al., 2015](https://arxiv.org/abs/1508.06615)          |
| Text       | Word-level: each word gets an ID                                           | [Brown et al., 1996](https://aclanthology.org/P96-1041)          |
| Text       | Subword: BPE on characters (merge adjacent symbol pairs), WordPiece (likelihood-based), Unigram LM (SentencePiece); produces variable-length subword units with IDs | [Sennrich et al., 2016](https://aclanthology.org/P16-1162/); [Schuster & Nakajima, 2012](https://research.google.com/pubs/archive/37842.pdf); [Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226) |
| Text       | Byte-level BPE: UTF-8 byte tokenization followed by iterative BPE merges on most frequent adjacent byte pairs until target vocabulary size is reached | [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf); [Hugging Face: Tokenizer summary](https://huggingface.co/docs/transformers/en/tokenizer_summary) |
| Image      | ViT patches: split image into fixed-size patches and embed each            | [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)     |
| Image      | VQGAN tokens: quantize image features into discrete tokens                  | [Esser et al., 2021](https://arxiv.org/abs/2012.09841)           |
| Video      | TimeSformer: split into spatio-temporal patches                            | [Bertasius et al., 2021](https://arxiv.org/abs/2102.05095)       |
| Audio      | VQ-VAE: discretize spectrogram or waveform into codebook indices           | [van den Oord et al., 2017](https://arxiv.org/abs/1711.00937)    |
| Time series| TS2Vec: discretize or embed continuous readings                            | [Yue et al., 2021](https://arxiv.org/abs/2106.10466)             |
| 3D geometry| Point-BERT: discretize point cloud into discrete tokens                    | [Yu et al., 2022](https://arxiv.org/abs/2111.14819)              |

## 2. Input Embedding

Each token is then mapped to a dense vector forming a trainable embedding matrix ($E \in \mathbb{R}^{V \times d_{\mathrm{model}}}$).

The embedding matrix $E$ is learned during training to represent token semantics, allowing the model to capture meaningful relationships between tokens.

## 3. Positional Encoding

The embeddings are combined with positional encodings ($P \in \mathbb{R}^{n \times d_{\mathrm{model}}}$). 

Let position $i\in\{0,\dots,n-1\}$, index $k\in\{0,\dots,\tfrac{d_{\mathrm{model}}}{2}-1\}$, positional matrix $P\in\mathbb{R}^{n\times d_{\mathrm{model}}}$, and row vector $p_i\in\mathbb{R}^{d_{\mathrm{model}}}$ (the $i$‑th row of $P$).

| Type                          | Method                                                                                                                                                     | Reference                                                      |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Fixed sinusoidal encodings     | $P_{i, 2k} = \sin(i / 10000^{2k / d_{\text{model}}}), \; P_{i, 2k+1} = \cos(i / 10000^{2k / d_{\text{model}}})$ | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)      |
| Learned absolute position embeddings | learned vector $p_i \in \mathbb{R}^{d_{\text{model}}}$ updated through training, allowing the model to adapt position representations to the task | [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)       |
| Relative position encodings    | Positions are encoded relative to each other within the attention mechanism, letting the model focus on pairwise distances rather than absolute positions       | [Shaw et al., 2018](https://arxiv.org/abs/1803.02155)         |

The token embeddings and positional encodings are combined via element-wise addition: $Z^{(0)} = X + P$. This has two benefits:
1. The linearity between $X$ and $Z$, and $P$ and $Z$ preserves separate contributions of $X$ and $P$ in the dot products from the "Attention" operation. 
2. Since the addition is element-wise, the combined input $Z^{(0)}$ retains the same shape as both the token embeddings and positional encodings, preserving the sequence length $n$ and model dimension $d_{\mathrm{model}}$.

## 4. Attention

The combined token embeddings and positional encodings $Z^{(0)} = X + P$ are factorized into three different learnable vector spaces: $Q$, $K$, and $V$.

$$
\begin{aligned}
Q &= Z^{(0)} W^Q = X W^Q + P W^Q \\
K &= Z^{(0)} W^K = X W^K + P W^K \\
V &= Z^{(0)} W^V = X W^V + P W^V
\end{aligned}
\quad\text{where}\quad Q, K, V \in \mathbb{R}^{n \times d_k}
$$

Given $n$ being the sequence length, and $d_k$ as the feature length for the model.

Attention scores $S_{ij} \in \mathbb{R}^{n \times n}$ are then computed from either
1. Additive $q_i, k_i$, then learning the score through training: $S_{ij} = v^\top \tanh(W_q q_i + W_k k_j + b)$ with a hidden layer: for each query–key pair $(q_i, k_j)$. ([Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473))
2. The dot product of $Q,K$ ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))

### Dot product method
$$
\begin{aligned}
S_{ij}
&= Q_{i}K_{j}^\top \\
&= (x_i W^Q + p_i W^Q)\,(x_j W^K + p_j W^K)^\top \\
&= x_i W^Q (W^K)^\top x_j^\top
+ x_i W^Q (W^K)^\top p_j^\top
+ p_i W^Q (W^K)^\top x_j^\top
+ p_i W^Q (W^K)^\top p_j^\top
\end{aligned}
$$

This separates the position from the content from Z and :
  1. $Q$ content with $K$ content: $x_i W^Q (x_j W^K)^\top$   
  2. $Q$ content with $K$ position: $x_i W^Q (p_j W^K)^\top$   
  3. $Q$ position $K$ content: $p_i W^Q (x_j W^K)^\top$
  4. $Q$ position $K$ position: $p_i W^Q (p_j W^K)^\top$

To prevent softmax saturation the are also scores are scaled $\frac{1}{\sqrt{d_k}}$, then multiplied with $V$.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

### MultiHead Attention

Each head $h$ computes its own queries, keys, and values by linear projection of the same input $Z$.
$$
Q_h = Z W_h^Q, \quad K_h = Z W_h^K, \quad V_h = Z W_h^V
$$  

It's an ensemble techique where the average score of parallel heads converging to an optimal solution faster than a single head.
The outputs are concatenated and projected back to model dimension.

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
