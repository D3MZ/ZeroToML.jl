# ZeroToML

[![Build Status](https://github.com/D3MZ/ZeroToML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/D3MZ/ZeroToML.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/D3MZ/ZeroToML.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/D3MZ/ZeroToML.jl)

Yet another AI from scratch repo.

- The files are appropriately named.
    - [notes](/notes) on my notes. They're still very rough.
    - [test](/test) for it in action on limited resources.
    - [examples](/examples) long training on real datasets.
- May contain AI slop.

## Roadmap
### Research
- [ ] Transformers
  - [ ] Encoder
  - [x] Decoder
  - [ ] Encoder + Decoder
  - [ ] Impact on Different tokenizers (i.e. BPE tokenizer vs Character level)
  - [ ] Impact on number of heads
- [ ] Diffusion
  - [ ] [DDPM — Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
  - [ ] [DDIM — Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
  - [ ] [SDE — Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
  - [ ] [Flow Matching](https://arxiv.org/abs/2210.02747)
  - [ ] [Rectified Flow — Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [ ] RL
  - [ ] [PPO - Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  - [ ] [GRPO - Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
  - [ ] [SAC - Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
  - [ ] Framework: [STOMP/OAK - Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://arxiv.org/pdf/2202.03466) 
- [ ] Neural Network primitives
  - [ ] [CNN - A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285)
    - [x] Discrete convolutions
    - [x] Stride
    - [ ] Padding
    - [ ] Pooling
    - [ ] Transposed
    - [ ] Dilated convolutions
    - [ ] Works with Gradients

### Architecture 
- [x] Abstract common functions? Might reduce readability :(
  - [x] Use Multiple Dispatch and Structs/Types
  - [ ] Filename, Model name is Paper name
- [ ] Stateless? Memory allocations concerns if truely end-to-end stateless. We get history for free though. Maybe Stateless functions only?

### Features
- [ ] Remove Flux / NNlib dependancies in core code. 
- [ ] Remove Zygote / AutoDiff? We lose flexibility in changing the model.
- [ ] Test from-scratch versions against established libraries for correctness.
- [ ] Einstien notation-like that better exposes the math instead of hiding behind API abstractions. Maybe start with Tullio?