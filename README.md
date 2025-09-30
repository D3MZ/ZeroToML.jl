# ZeroToML

[![Build Status](https://github.com/D3MZ/ZeroToML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/D3MZ/ZeroToML.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/D3MZ/ZeroToML.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/D3MZ/ZeroToML.jl)

Yet another AI from scratch repo. Framework free.

- The files are appropriately named.
    - [notes](/notes) on my notes.
    - [test](/test) for it in action on limited resources.
    - [examples](/examples) long training on real datasets.
- May contain AI slop.

### Work in Progress
- [Optimizers](notes/optimizers.md)
- [Transformer](notes/transformers.md)
- [Diffusion](notes/diffusion.md)

### Roadmap
- [ ] Transformers
  - [ ] Encoder
  - [x] Decoder
  - [ ] Encoder + Decoder
  - [ ] Impact on Different tokenizers (i.e. BPE tokenizer vs Character level)
  - [ ] Impact on number of heads
- [ ] Diffusion
- [ ] RL
- [x] Refactor to stateless
- [ ] Abstract common functions? Might reduce readability :(
- [ ] Test from-scratch versions against established libraries for correctness.
- [ ] Create einstien notation-like that better exposes the math instead of hiding behind API abstractions.