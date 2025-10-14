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

### Architecture 
- [ ] Abstract common functions? Might reduce readability :(
  - [ ] Use Multiple Dispatch and Structs/Types
  - [ ] Paper name = Filename, Model name
- [ ] Stateless? Memory allocations concerns if truely end-to-end stateless. We get history for free though. Maybe Stateless functions only?

### Features
- [ ] Remove Flux / NNlib dependancies in core code. May be slower, but gain understanding.
- [ ] Remove Zygote / AutoDiff? We lose flexibility in changing the model, but gain understanding.
- [ ] Test from-scratch versions against established libraries for correctness.
- [ ] Einstien notation-like that better exposes the math instead of hiding behind API abstractions. Maybe start with Tullio?