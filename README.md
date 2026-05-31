# ZeroToML

[![Build Status](https://github.com/D3MZ/ZeroToML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/D3MZ/ZeroToML.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/D3MZ/ZeroToML.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/D3MZ/ZeroToML.jl)

Yet another AI from scratch repo.

- The files are appropriately named.
    - [notes](/notes) on my notes. They're still very rough.
    - [test](/test) for it in action on limited resources.
    - [examples](/examples) long training on real datasets. This doesn't train well due to the small networks.
- May contain AI slop.

## Roadmap
### Research
- [ ] [Gaussian Processes](https://gaussianprocess.org/gpml/chapters/RW.pdf)
- [ ] Decision Trees
  - [ ] [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- [ ] Transformers
  - [ ] Encoder
  - [x] Decoder
  - [ ] Encoder + Decoder
  - [ ] Impact on Different tokenizers (i.e. BPE tokenizer vs Character level)
  - [ ] Impact on number of heads
- [ ] Diffusion
  - [x] [DDPM — Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
    - [ ] Separate Gaussian DDPM from heavy-tailed diffusion experiments
    - [ ] Replace variance-based scaling for Cauchy with scale-parameter schedules
    - [ ] Add distribution-specific forward sampling for Gaussian, StudentT, Cauchy, and other noise families
    - [ ] Train distribution-specific targets instead of always predicting Gaussian-style ε
    - [ ] Add score-matching targets for non-Gaussian noise processes
    - [ ] Implement distribution-aware reverse steps instead of reusing the Gaussian posterior formula
    - [ ] Explore approximate posterior samplers for StudentT and Cauchy transitions
    - [ ] Compare ε-prediction, score-prediction, and x₀-prediction objectives
    - [ ] Validate heavy-tailed diffusion with broader held-out samples, not only toy box reconstruction
  - [ ] [DDIM — Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
  - [x] [SDE — Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
  - [ ] [Flow Matching](https://arxiv.org/abs/2210.02747)
  - [ ] [Rectified Flow — Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [ ] JEPA
  - [ ] [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/pdf/2301.08243) ([Code](https://github.com/facebookresearch/ijepa))
  - [ ] [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://openreview.net/forum?id=WFYbBOEOtv) ([Code](https://github.com/facebookresearch/jepa))
  - [ ] [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/pdf/2506.09985) ([Code](https://github.com/facebookresearch/vjepa2))
  - [ ] [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/pdf/2603.14482) ([Code](https://github.com/facebookresearch/vjepa2))
  - [ ] [D-JEPA: Denoising with a Joint-Embedding Predictive Architecture](https://arxiv.org/pdf/2410.03755)
  - [ ] [VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/pdf/2512.10942)
  - [ ] [EB-JEPA examples](https://github.com/facebookresearch/eb_jepa)
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
- [ ] [Errors and residuals](https://en.wikipedia.org/wiki/Errors_and_residuals)

### Architecture 
- [ ] train should take Dates.period instead of epochs like sde? This simplifies autoresearch.
- [x] Abstract common functions? Might reduce readability :(
  - [x] Use Multiple Dispatch and Structs/Types
  - [ ] Filename, Model name is Paper name
- [ ] Stateless? Memory allocations concerns if truely end-to-end stateless. We get history for free though. Maybe Stateless functions only?

### Features
- [ ] Remove Flux / NNlib dependancies in core code. 
- [ ] Remove Zygote / AutoDiff? We lose flexibility in changing the model.
- [ ] Test from-scratch versions against established libraries for correctness.
- [ ] Einstien notation-like that better exposes the math instead of hiding behind API abstractions. Maybe start with Tullio?

### TODO
- [ ] Improve training in examples/