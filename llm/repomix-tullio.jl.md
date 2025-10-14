This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.buildkite/
  pipeline.yml
.github/
  workflows/
    ci.yml
    CompatHelper.yml
    TagBot.yml
benchmarks/
  01/
    broadcast01.jl
    complex01.jl
    cosine01.jl
    distances01.jl
  02/
    matmul.jl
    permute.jl
    Project.toml
ext/
  TullioChainRulesCoreExt.jl
  TullioCUDAExt.jl
  TullioFillArraysExt.jl
  TullioTrackerExt.jl
src/
  grad/
    avxdual.jl
    reverse.jl
  precompile/
    precompile_Base.jl
    precompile_Core.jl
    precompile_Tullio.jl
  einsum.jl
  eval.jl
  forward.jl
  macro.jl
  precompile.jl
  shifts.jl
  symbolic.jl
  tensor.jl
  threads.jl
  tools.jl
  Tullio.jl
test/
  cuda.jl
  einsum.jl
  gradients.jl
  group-1.jl
  group-2.jl
  group-3.jl
  parsing.jl
  runtests.jl
  tensorgrad.jl
  utils.jl
.gitignore
LICENSE
Project.toml
README.md
```

# Files

## File: .buildkite/pipeline.yml
````yaml
env:
  JULIA_NUM_THREADS: "6"
  # SECRET_CODECOV_TOKEN: "..."

steps:
  - label: "Julia 1.11"
    plugins:
      - JuliaCI/julia#v0.5:
          version: "1.11"
      - JuliaCI/julia-test#v0.3: ~
      # - JuliaCI/julia-coverage#v0.3:
      #     codecov: true
    agents:
      queue: "juliagpu"
      cuda: "*"
    if: build.message !~ /\[skip tests\]/
    timeout_in_minutes: 60

  - label: "Julia 1.10"
    plugins:
      - JuliaCI/julia#v0.5:
          version: "1.10"
      - JuliaCI/julia-test#v0.3: ~
      # - JuliaCI/julia-coverage#v0.3:
      #     codecov: true
    agents:
      queue: "juliagpu"
      cuda: "*"
    if: build.message !~ /\[skip tests\]/
    timeout_in_minutes: 60
````

## File: .github/workflows/ci.yml
````yaml
name: CI
on:
  pull_request:
    branches:
      - master
  push:
    branches:
      - master
    tags: '*'
jobs:
  test:
    name: v${{ matrix.version }} -t${{ matrix.threads }} / group-${{ matrix.group }} / ${{ github.event_name }} / ${{ matrix.os }}+${{ matrix.arch }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        arch:
          - x64
        group:
          - '1'
          - '2'
          - '3'
        os:
          - ubuntu-latest
        threads:
          - '1'
          - '6' # t>2 might be ignored on Julia <= 1.5
        version:
          - '1.10'
          - '1' # automatically expands to the latest stable 1.x release of Julia
          # - 'nightly'
    steps:
      - uses: actions/checkout@v5
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}
      - uses: actions/cache@v4
        env:
          cache-name: cache-artifacts
        with:
          path: ~/.julia/artifacts
          key: ${{ runner.os }}-test-${{ env.cache-name }}-${{ hashFiles('**/Project.toml') }}
          restore-keys: |
            ${{ runner.os }}-test-${{ env.cache-name }}-
            ${{ runner.os }}-test-
            ${{ runner.os }}-
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
        with:
          coverage: 'false'
        env:
          JULIA_NUM_THREADS: ${{ matrix.threads }}
          TULLIO_TEST_GROUP: ${{ matrix.group }}
#       - uses: julia-actions/julia-processcoverage@v1
#       - uses: codecov/codecov-action@v1
#         with:
#           file: lcov.info
````

## File: .github/workflows/CompatHelper.yml
````yaml
name: CompatHelper
on:
  schedule:
    - cron: 0 0 * * *
  workflow_dispatch:
jobs:
  CompatHelper:
    runs-on: ubuntu-latest
    steps:
      - name: Pkg.add("CompatHelper")
        run: julia -e 'using Pkg; Pkg.add("CompatHelper")'
      - name: CompatHelper.main()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          COMPATHELPER_PRIV: ${{ secrets.DOCUMENTER_KEY }}
        run: julia -e 'using CompatHelper; CompatHelper.main()'
````

## File: .github/workflows/TagBot.yml
````yaml
name: TagBot
on:
  issue_comment:
    types:
      - created
  workflow_dispatch:
jobs:
  TagBot:
    if: github.event_name == 'workflow_dispatch' || github.actor == 'JuliaTagBot'
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
````

## File: benchmarks/01/broadcast01.jl
````
# Some quick and dirty broadcasting AD benchmarks, 25 April 2020

julia> using Zygote, Tracker, ReverseDiff, Tullio

julia> x1 = randn(10); f1(x) = 1+tanh(x); # A simple function which isn't hard-coded

julia> Tracker.gradient(x -> sum(f1.(x)), x1)[1]'
Tracked 1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> Zygote.gradient(x -> sum(f1.(x)), x1)[1]'
1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> ReverseDiff.gradient(x -> sum(f1.(x)), (x1,))[1]'
1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> Tracker.gradient(x -> (@tullio s = 1+tanh(x[i])), x1)[1]'
Tracked 1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

julia> Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i])), x1)[1]'
1×10 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.333674  0.174402  0.786548  0.222385  …  0.143003  0.361015  0.910887  0.940078

# Time that on a bigger array:
# right now Zygote is very slow, and ReverseDiff worse.

julia> x2 = randn(1000,1000);

julia> @btime sum(f1.($x2));
  25.181 ms (2 allocations: 7.63 MiB)

julia> @btime sum(f1, $x2);
  24.414 ms (0 allocations: 0 bytes)

julia> @btime sum(Broadcast.broadcasted(f1, $x2)) # on Julia 1.5
  29.678 ms (0 allocations: 0 bytes)

julia> @btime @tullio s = f1($x2[i,j])
  13.195 ms (41 allocations: 2.66 KiB)

julia> @btime Tracker.gradient(x -> sum(f1.(x)), $x2);
  58.780 ms (44 allocations: 53.41 MiB)

julia> @btime Zygote.gradient(x -> sum(f1.(x)), $x2);
  119.821 ms (3000046 allocations: 122.07 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(f1.(x)), ($x2,));
  941.047 ms (15000027 allocations: 567.06 MiB)

julia> @btime Tracker.gradient(x -> (@tullio s = 1+tanh(x[i,j])), $x2);
  33.391 ms (88 allocations: 30.52 MiB)

julia> @btime Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i,j])), $x2);
  29.928 ms (84 allocations: 7.64 MiB)

julia> Tullio.@printgrad 1+tanh(x) x
δx = 1 - tanh(x) ^ 2

# Compare to tanh, which is a special case of Zygote's:

julia> @btime Zygote.gradient(x -> sum(tanh.(x)), $x2);
  25.633 ms (20 allocations: 15.26 MiB)

# Fancier ways to use @tullio:

julia> using LoopVectorization

julia> @btime @tullio s = f1($x2[i,j])
  3.194 ms (39 allocations: 2.59 KiB)

julia> @btime Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i,j])), $x2);
  10.663 ms (81 allocations: 7.64 MiB)

julia> @btime Zygote.gradient(x -> (@tullio s := 1+tanh(x[i,j])), $x2); # skip making s[i]
  9.406 ms (79 allocations: 7.63 MiB)

julia> using ForwardDiff

julia> @btime Zygote.gradient(x -> sum(@tullio s[i] := 1+tanh(x[i,j]) grad=Dual), $x2);
  10.512 ms (81 allocations: 7.64 MiB)

# Another problem:

julia> @btime sum($x2 .+ $x2' ./ 2);
  2.968 ms (2 allocations: 7.63 MiB)

julia> @btime Tracker.gradient(x -> sum(x .+ x' ./ 2), $x2);
  25.699 ms (210 allocations: 68.67 MiB)

julia> @btime Zygote.gradient(x -> sum(x .+ x' ./ 2), $x2);
  9.006 ms (15 allocations: 38.15 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(x .+ x' ./ 2), $x2);
  1.219 s (19000027 allocations: 719.65 MiB)

julia> @btime Zygote.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2), $x2);
  1.756 ms (164 allocations: 7.64 MiB)

# And without @avx magic...

julia> @btime Zygote.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false), $x2);
  2.368 ms (165 allocations: 7.64 MiB)

julia> @btime Zygote.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false threads=false), $x2);
  5.146 ms (25 allocations: 7.63 MiB)

julia> @btime Tracker.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false), $x2);
  7.169 ms (172 allocations: 30.53 MiB)

julia> @btime ReverseDiff.gradient(x -> (@tullio s := x[i,j] + x[j,i]/2  avx=false), $x2);
  7.955 ms (164 allocations: 30.53 MiB)

# ReverseDiff should soon have forward-mode broadcasting from this package:

julia> using DistributionsAD

julia> @btime ReverseDiff.gradient(x -> sum(f1.(x)), ($x2,));
  35.920 ms (34 allocations: 53.41 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(x .+ x' ./ 2), $x2);
  1.365 s (19000087 allocations: 727.28 MiB)

julia> @btime ReverseDiff.gradient(x -> sum(x .+ x'), $x2); # without the ./2 bit!
  48.920 ms (1000030 allocations: 99.18 MiB)

# Yota only supports broadcasting of primitives, i.e. things with an explicit scalar rule

julia> using Yota

julia> @diffrule f1(x)  x  dy*(1-tanh(x)^2)

julia> @btime Yota.grad(x -> sum(f1.(x)), $x2);
  77.449 ms (86 allocations: 45.78 MiB)

julia> @btime Yota.grad(x -> sum(x .+ x' ./ 2), $x2);
  8.423 ms (89 allocations: 38.15 MiB)
````

## File: benchmarks/01/complex01.jl
````
# This is a batched matrix multiplication test, with the batch index coming first,
# from this thread:
# https://discourse.julialang.org/t/non-matching-indices-error-using-tensoroperations/35136
# Some improvements to OMEinsum since then make it much quicker.

julia> using Einsum, OMEinsum, TensorCast, Tullio, LoopVectorization

julia> f_ome(a,b) = @ein c[k, n] := a[k, n, c] * conj(b)[c,k]; # permutedims + batched_mul

julia> f_cast(a,b) = @reduce c[k, n] := sum(l) a[k, n, l] * conj(b[l, k]); # broadcasting

julia> f_ein(a,b) = @einsum c_[k, n] := a[k, n, c] * conj(b[c,k]); # naiive loops

julia> f_viel(a,b) = @vielsum c_[k, n] := a[k, n, c] * conj(b[c,k]); # plus threads

julia> f_tul(a,b) = @tullio c[k, n] := a[k, n, c] * conj(b[c,k]); # less naiive loops?

julia> a = randn(ComplexF64, 300, 400, 500); b = randn(ComplexF64, 500, 300);

julia> f_ome(a,b) ≈ f_cast(a,b) ≈ f_ein(a,b) ≈ f_tul(a,b)
true

julia> @btime f_ome($a, $b);
  342.619 ms (104 allocations: 921.49 MiB)

julia> @btime f_cast($a, $b);
  557.428 ms (26 allocations: 919.65 MiB)

julia> @btime f_ein($a, $b);
  264.429 ms (2 allocations: 1.83 MiB)

julia> @btime f_viel($a, $b);
  147.739 ms (25 allocations: 1.83 MiB)

julia> @btime f_tul($a, $b);
  191.859 ms (836 allocations: 1.86 MiB)

# But LoopVectorization isn't yet in play, as the arrays have complex elements.
# On real numbers, it makes a big difference!

julia> ar = real(a); br = real(b);

julia> @btime f_ome($ar, $br);
  190.672 ms (102 allocations: 459.60 MiB)

julia> @btime f_ein($ar, $br);
  122.616 ms (2 allocations: 937.58 KiB)

julia> @btime f_viel($ar, $br);
  82.922 ms (25 allocations: 940.33 KiB)

julia> @btime f_tul($ar, $br);
  39.105 ms (836 allocations: 967.17 KiB)

# Can we get there with StructArrays?
# (Perhaps this could be made automatic, with yet more macrology...)

julia> using StructArrays, LoopVectorization

julia> @time sa = StructArray(a); sb = StructArray(b);
  4.160518 seconds (120.00 M allocations: 3.576 GiB, 29.61% gc time)

julia> function f_tul(a::StructArray, b::StructArray)
           a_re, a_im = a.re, a.im
           b_re, b_im = b.re, b.im
           @tullio c_re[k, n] := a_re[k, n, c] * b_re[c,k] + a_im[k, n, c] * b_im[c,k]
           @tullio c_im[k, n] := a_im[k, n, c] * b_re[c,k] - a_re[k, n, c] * b_im[c,k]
           StructArray{eltype(a)}((c_re, c_im))
       end
f_tul (generic function with 2 methods)

julia> f_ein(sa, sb) ≈ f_tul(sa, sb)
true

julia> @btime f_tul($sa, $sb);
  136.801 ms (1670 allocations: 1.89 MiB)

# That's worse than twice the real calculation, but I guess each line is harder.
# Converting to StructArrays is really slow though!

julia> @btime StructArray($a);
  3.068 s (119999498 allocations: 3.58 GiB)

julia> @btime real($a);
  248.114 ms (2 allocations: 457.76 MiB)

# Compare Einsum on StructArrays: not great, however you do it.

julia> @btime f_viel($sa, $sb);
  479.932 ms (25 allocations: 1.83 MiB)

julia> typeof(f_viel(sa, sb))
Array{Complex{Float64},2}

julia> similar(sa, 1,3)
1×3 StructArray(::Array{Float64,2}, ::Array{Float64,2}) with eltype Complex{Float64}:
 2.29092e-314+2.27751e-314im  2.29092e-314+3.7518e-314im  3.7518e-314+3.75184e-314im

julia> similar(sa, ComplexF64, 1,3)
1×3 Array{Complex{Float64},2}:
 2.66668e-314+2.66668e-314im  2.66668e-314+2.29246e-314im  2.29246e-314+2.27751e-314im

julia> f_viel!(c_,a,b) = @vielsum c_[k, n] = a[k, n, c] * conj(b[c,k]);

julia> sc = similar(f_tul(sa, sb)); typeof(sc)
StructArray{Complex{Float64},2,NamedTuple{(:re, :im),Tuple{Array{Float64,2},Array{Float64,2}}},Int64}

julia> @btime f_viel!($sc, $sa, $sb);
  478.993 ms (240023 allocations: 5.50 MiB)

# Try KernelAbstractions? This should handle threading instead of my code.

julia> using KernelAbstractions, CuArrays # CuArrays just to trigger it!

julia> ENV["JULIA_DEBUG"] = Main;

julia> f_tul2(a,b) = @tullio c[k, n] := a[k, n, c] * conj(b[c,k])  threads=false;

julia> f_tul2(a, b);
┌ Debug: KernelAbstractions CPU actor:
│   typeof.(tuple(ℛ::AbstractArray{𝒯}, a, b, 𝒶_n, 𝒶_k, 𝒶_c)) = (Array{Complex{Float64},2}, Array{Complex{Float64},3}, Array{Complex{Float64},2}, UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64})
└ @ Main ~/.julia/dev/Tullio/src/macro.jl:724

julia> ENV["JULIA_DEBUG"] = "none";

julia> @btime f_tul2($a, $b);
  599.003 ms (64 allocations: 1.84 MiB)

# Note that if you just run f_tul threads=false without KernelAbstractions, it takes 1.3sec,
# which is much much slower than @einsum, so perhaps that's another bug.


#########################

julia> using StructArrays

julia> a = randn(ComplexF64, 300, 400, 500);

julia> @time StructArray(a);
  4.967282 seconds (120.00 M allocations: 3.576 GiB, 21.80% gc time)

julia> @time StructArray{ComplexF64}((real(a), imag(a)));
  0.630163 seconds (7 allocations: 915.528 MiB, 10.91% gc time)

julia> @code_warntype StructArray(a)
Variables
  #self#::Type{StructArray}
  v::Array{Complex{Float64},3}
  #46::StructArrays.var"#46#48"

Body::StructArray{Complex{Float64},3,NamedTuple{(:re, :im),Tuple{Array{Float64,3},Array{Float64,3}}},Int64}
1 ─      (#46 = %new(StructArrays.:(var"#46#48")))
│   %2 = #46::Core.Compiler.Const(StructArrays.var"#46#48"(), false)
│   %3 = StructArrays.:(var"#StructArray#45")(%2, #self#, v)::StructArray{Complex{Float64},3,NamedTuple{(:re, :im),Tuple{Array{Float64,3},Array{Float64,3}}},Int64}
└──      return %3
````

## File: benchmarks/01/cosine01.jl
````
using Tullio
cd(joinpath(dirname(pathof(Tullio)), "..", "benchmarks", "01"))
using Pkg; pkg"activate ."

# or

using Pkg; pkg"add LoopVectorization"
using Pkg; pkg"add TensorCast NNlib ForwardDiff Zygote"


# This example is a nice usage:
# https://discourse.julialang.org/t/help-to-improve-performance-of-gradient-calculation-on-tensor-operations/37773/3

using TensorCast

function cosinesim(a, b)
    @reduce similarity[i, j, k] := sum(s) a[s, j, k] * b[s, i, k] /
        sqrt( @reduce [_, j, k] := sum(s') a[s', j, k]^2) /
        sqrt( @reduce [_, i, k] := sum(s'') b[s'', i, k]^2)
end

function mysoftmax(a)
    @cast submax[i, j, k] := a[i, j, k] - @reduce [_, j, k] := maximum(i) a[i, j, k]
    @cast r[i, j, k] := exp(submax[i, j, k]) / @reduce [_, j, k] := sum(i) exp(submax[i, j, k])
end

pairwise2(a) = mysoftmax(cosinesim(a))

N, W, R, B = 16, 64, 4, 16;
a = rand(Float32, W, R, B);
b = rand(Float32, W, N, B);

#=
First look just at the numerator. This should be easy with `NNlib.batched_mul`,
but there is [this issue](https://github.com/FluxML/Zygote.jl/issues/552)
so I'm using [my PR](https://github.com/FluxML/NNlib.jl/pull/191)...
but the end result is a gradient 4x faster.

Almost as fast as Tullio! (which is faster without multi-threading today, something is broken)
=#

using Zygote, NNlib

reducemul(a, b) = @reduce z[i, j, k] := sum(s) a[s, j, k] * b[s, i, k]
reducemul(a, b) ≈ batched_mul(batched_transpose(b), a) # true

@btime reducemul($a, $b); # 98.262 μs (52 allocations: 342.41 KiB)
@btime batched_mul(batched_transpose($b), $a); # 13.605 μs (5 allocations: 4.20 KiB) -> 6.447 μs (2 allocations: 4.14 KiB) with MKL

using Tullio, LoopVectorization, ForwardDiff

tullio_mul(a, b) = @tullio z[i, j, k] := a[s, j, k] * b[s, i, k]
tullio_mul2(a, b) = @tullio z[i, j, k] := a[s, j, k] * b[s, i, k]  grad=Dual
reducemul(a, b) ≈ tullio_mul(a, b) # true

@btime tullio_mul($a, $b); # 2.696 μs (1 allocation: 4.13 KiB) -- improved!

#=
For gradients, I've inserted `(x->x).()` into these because sum creates a `FillArray`
which some methods don't like, but any useful calculation would have something else after this.

Now not needed for @tullio!
=#

grad_a = gradient((a,b) -> sum(reducemul(a, b)), a, b)[1];
grad_a ≈ gradient((a,b) -> sum((x->x), batched_mul(PermutedDimsArray(b, (2,1,3)), a)), a, b)[1] # true
grad_a ≈ gradient((a,b) -> sum(tullio_mul(a, b)), a, b)[1] # true
grad_a ≈ gradient((a,b) -> sum(tullio_mul2(a, b)), a, b)[1] # true

@btime gradient((a,b) -> sum(reducemul(a, b)), $a, $b);       # 307.836 μs (193 allocations: 1.17 MiB)
@btime gradient((a,b) -> sum(x->x, reducemul(a, b)), $a, $b); # 378.264 μs (3316 allocations: 1.23 MiB)

@btime gradient((a,b) -> sum(x->x, batched_mul(PermutedDimsArray(b, (2,1,3)), a)), $a, $b); # 101.782 μs (3159 allocations: 151.52 KiB) -> 218.180 μs (3334 allocations: 161.70 KiB) MKL?

@btime gradient((a,b) -> sum(tullio_mul(a, b)), $a, $b);       # 21.190 μs (41 allocations: 85.89 KiB)
@btime gradient((a,b) -> sum(x->x, tullio_mul(a, b)), $a, $b); # 79.291 μs (3156 allocations: 151.19 KiB)

# version with ForwardDiff:
@btime gradient((a,b) -> sum(tullio_mul2(a, b)), $a, $b);      # 25.168 μs (37 allocations: 85.70 KiB)

#=
Tullio ought to make it easy to fuse that multiplication with the division, perhaps?
But this is slow!
=#

function cosine_nnlib(a, b)
    @reduce den1[j, k] := sum(s) a[s, j, k]^2
    @reduce den2[i, k] := sum(s) b[s, i, k]^2
    bmm = batched_mul(PermutedDimsArray(b, (2,1,3)), a)
    @cast similarity[i, j, k] := bmm[i, j, k] / sqrt(den1[j, k] * den2[i, k])
end

function cosine_fused(a, b)
    @tullio den1[j, k] := a[s, j, k]^2
    @tullio den2[i, k] := b[s, i, k]^2
    @tullio similarity[i, j, k] := a[s, j, k] * b[s, i, k] / sqrt(den1[j, k] * den2[i, k])
end

function cosine_separated(a, b)
    @tullio bmm[i, j, k] := a[s, j, k] * b[s, i, k]

    @tullio den1[j, k] := a[s, j, k]^2
    @tullio den2[i, k] := b[s, i, k]^2

    @tullio f1[j, k] := 1/sqrt(den1[j, k])
    @tullio f2[i, k] := 1/sqrt(den2[i, k])
    @tullio similarity[i, j, k] := bmm[i, j, k] * f1[j, k] * f2[i, k]
end

cosinesim(a, b) ≈ cosine_nnlib(a, b) # true
cosinesim(a, b) ≈ cosine_fused(a, b) # true
cosinesim(a, b) ≈ cosine_separated(a, b) # true

@btime cosinesim($a, $b);    # 318.341 μs (87 allocations: 425.14 KiB)
@btime cosine_nnlib($a, $b); #  34.407 μs (42 allocations: 91.20 KiB) -> 112.308 μs (179 allocations: 99.05 KiB) MKL
@btime cosine_fused($a, $b);   #  7.678 μs (3 allocations: 5.59 KiB)
@btime cosine_separated($a, $b);   #  4.672 μs (6 allocations: 11.19 KiB)

grad_a2 = gradient((a,b) -> sum(cosinesim(a, b)), a, b)[1];
grad_a2 ≈ gradient((a,b) -> sum(cosine_nnlib(a, b)), a, b)[1] # true
grad_a2 ≈ gradient((a,b) -> sum(cosine_fused(a, b)), a, b)[1] # true
grad_a2 ≈ gradient((a,b) -> sum(cosine_separated(a, b)), a, b)[1] # true

@btime gradient((a,b) -> sum(cosinesim(a, b)), $a, $b);    #  984.351 μs (1414 allocations: 3.51 MiB)
@btime gradient((a,b) -> sum(cosine_nnlib(a, b)), $a, $b); #  232.991 μs (6375 allocations: 579.78 KiB) -> 280.876 μs (3442 allocations: 520.61 KiB) MKL
@btime gradient((a,b) -> sum(cosine_fused(a, b)), $a, $b);     # 80.207 μs (129 allocations: 252.86 KiB)
@btime gradient((a,b) -> sum(cosine_separated(a, b)), $a, $b); # 80.367 μs (207 allocations: 267.08 KiB)

# cosine_fused might end up computing / sqrt N^3 times, instead of 2N^2 times,
# so it's not obviously a great idea.
# But I like LoopVectorization is smart enough to avoid that? Gradient is as fast now.

# Its gradient now looks like this:

Tullio.@printgrad x/sqrt(y*z)   x y z


#=
Aside, things about softmax:
=#

mysoftmax(a) ≈ softmax(a, dims=1)

@btime mysoftmax($a);       # 77.517 μs (37 allocations: 50.17 KiB)
@btime softmax($a, dims=1); # 45.405 μs (8 allocations: 33.03 KiB)

@btime gradient(a -> sum(mysoftmax(a)), $a);      # 640.619 μs
@btime gradient(a -> sum(softmax(a, dims=1)), $a); # 95.622 μs
@btime gradient(a -> sum(x->x, mysoftmax(a)), $a);      #
@btime gradient(a -> sum(x->x, softmax(a, dims=1)), $a); # 322.103 μs (12358 allocations: 356.17 KiB)

# add avx to versions from https://github.com/FluxML/NNlib.jl/pull/135/
function NNlib.softmax(xs::Array; dims=1)
    max_ = maximum(xs, dims=dims)
    out = @avx exp.(xs .- max_)
    @avx out .= out ./ sum!(max_, out)
end

@btime softmax($a, dims=1); # 16.084 μs -- 3x quicker

function NNlib.∇softmax(Δ::Array, xs::Array; dims=1)
    sf = softmax(xs, dims=dims)
    @avx sf .* (Δ .- sum(Δ .* sf; dims=dims))
end

@btime gradient(a -> sum(x->x, softmax(a, dims=1)), $a); # 260.441 μs (12358 allocations: 323.36 KiB)
# also 3x quicker, once you subtract the big cost of broadcasting (x->x)!

using Tracker # quicker at broadcasting, not sure it has NNlib gradient at all

@btime Tracker.gradient(a -> sum(mysoftmax(a)), $a);       # 324.806 μs
@btime Tracker.gradient(a -> sum(softmax(a, dims=1)), $a); # 252.686 μs


using ReverseDiff

@btime ReverseDiff.gradient(a -> sum(mysoftmax(a)), $a); # 6.359 ms
````

## File: benchmarks/01/distances01.jl
````
# A Julia-vs-Jax game, from here:
# https://twitter.com/cgarciae88/status/1254269041784561665
# https://discourse.julialang.org/t/improving-an-algorithm-that-compute-gps-distances/38213/19
# https://gist.github.com/cgarciae/a69fa609f8fcd0aacece92660b5c2315

# These versions are updated to use Float32, which is what Jax is using (by default).
# It helped a little to avoid Float64 constants, and to add @inbounds in a few places.

using Pkg; pkg"add LoopVectorization Einsum TensorCast https://github.com/mcabbott/Tullio.jl"
using LoopVectorization, Tullio, Einsum, TensorCast, Test, BenchmarkTools

a = -100 .+ 200 .* rand(Float32, 5000, 2);
b = -100 .+ 200 .* rand(Float32, 5000, 2);

const None = [CartesianIndex()]

function distances(data1, data2)
    data1 = deg2rad.(data1)
    data2 = deg2rad.(data2)
    lat1 = @view data1[:, 1]
    lng1 = @view data1[:, 2]
    lat2 = @view data2[:, 1]
    lng2 = @view data2[:, 2]
    diff_lat = @view(lat1[:, None]) .- @view(lat2[None, :])
    diff_lng = @view(lng1[:, None]) .- @view(lng2[None, :])
    data = (
        @. sin(diff_lat / 2)^2 +
        cos(@view(lat1[:, None])) * cos(@view(lat2[None,:])) * sin(diff_lng / 2)^2
    )
    data .= @. 2.0 * 6373.0 * atan(sqrt(abs(data)), sqrt(abs(1.0 - data)))
    return reshape(data, (size(data1, 1), size(data2, 1)))
end

res = distances(a, b);
@test eltype(res) == Float32

function distances_threaded(data1, data2)
    lat1 = [deg2rad(data1[i,1]) for i in 1:size(data1, 1)]
    lng1 = [deg2rad(data1[i,2]) for i in 1:size(data1, 1)]
    lat2 = [deg2rad(data2[i,1]) for i in 1:size(data2, 1)]
    lng2 = [deg2rad(data2[i,2]) for i in 1:size(data2, 1)]
    # data = Matrix{Float64}(undef, length(lat1), length(lat2))
    data = Matrix{eltype(data1)}(undef, length(lat1), length(lat2))
    @inbounds Threads.@threads for i in eachindex(lat2)
        lat, lng = lat2[i], lng2[i]
        data[:, i] .= @. sin((lat1 - lat) / 2)^2 + cos(lat1) * cos(lat) * sin((lng1 - lng) / 2)^2
    end
    Threads.@threads for i in eachindex(data)
        # data[i] = 2.0 * 6373.0 * atan(sqrt(abs(data[i])), sqrt(abs(1.0 - data[i])))
        @inbounds data[i] = 2 * 6373 * atan(sqrt(abs(data[i])), sqrt(abs(1 - data[i])))
    end
    return data
end

function distances_threaded_simd(data1, data2) # @baggepinnen
    lat1 = [deg2rad(data1[i,1]) for i in 1:size(data1, 1)]
    lng1 = [deg2rad(data1[i,2]) for i in 1:size(data1, 1)]
    lat2 = [deg2rad(data2[i,1]) for i in 1:size(data2, 1)]
    lng2 = [deg2rad(data2[i,2]) for i in 1:size(data2, 1)]
    # data = Matrix{Float64}(undef, length(lat1), length(lat2))
    data = similar(data1, length(lat1), length(lat2))
    cos_lat1 = @avx cos.(lat1)
    Threads.@threads for i in eachindex(lat2)
        # lat, lng = lat2[i], lng2[i]
        @inbounds lat, cos_lat, lng = lat2[i], cos(lat2[i]), lng2[i]
        # @avx data[:, i] .= @. sin((lat1 - lat) / 2)^2 + cos(lat1) * cos(lat) * sin((lng1 - lng) / 2)^2
        @avx data[:, i] .= @. sin((lat1 - lat) / 2)^2 + cos_lat1 * cos_lat * sin((lng1 - lng) / 2)^2
    end
    Threads.@threads for i in eachindex(data)
        # @avx data[i] = 2.0 * 6373.0 * atan(sqrt(abs(data[i])), sqrt(abs(1.0 - data[i])))
        @avx data[i] = 2 * 6373 * atan(sqrt(abs(data[i])), sqrt(abs(1 - data[i])))
    end
    return data
end

@test res ≈ distances_threaded(a, b)
@test eltype(distances_threaded(a, b)) == Float32
@test res ≈ distances_threaded_simd(a, b)
@test eltype(distances_threaded_simd(a, b)) == Float32

function distances_bcast(data1, data2) # @DNF
    data1 = deg2rad.(data1)
    data2 = deg2rad.(data2)
    lat1 = @view data1[:, 1]
    lng1 = @view data1[:, 2]
    lat2 = @view data2[:, 1]
    lng2 = @view data2[:, 2]
    data = sin.((lat1 .- lat2') ./ 2).^2 .+ cos.(lat1) .* cos.(lat2') .* sin.((lng1 .- lng2') ./ 2).^2
    @. data = 2 * 6373 * atan(sqrt(abs(data)), sqrt(abs(1 - data)))
    return data
end

function distances_bcast_simd(data1, data2)
    data1 = deg2rad.(data1)
    data2 = deg2rad.(data2)
    lat1 = @view data1[:, 1]
    lng1 = @view data1[:, 2]
    lat2 = @view data2[:, 1]
    lng2 = @view data2[:, 2]
    @avx data = sin.((lat1 .- lat2') ./ 2).^2 .+ cos.(lat1) .* cos.(lat2') .* sin.((lng1 .- lng2') ./ 2).^2
    @. data = 2 * 6373 * atan(sqrt(abs(data)), sqrt(abs(1 - data)))
    return data
end

@test res ≈ distances_bcast(a, b)
@test eltype(distances_bcast(a, b)) == Float32
@test res ≈ distances_bcast_simd(a, b)
@test eltype(distances_bcast_simd(a, b)) == Float32

function distances_einsum(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @einsum cd1[n] := cos(data1[n,1])
    @einsum cd2[m] := cos(data2[m,1])

    @einsum data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

    @einsum data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

function distances_vielsum(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @vielsum cd1[n] := cos(data1[n,1])
    @vielsum cd2[m] := cos(data2[m,1])

    @vielsum data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

    @vielsum data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

@test res ≈ distances_einsum(a, b)
@test eltype(distances_einsum(a, b)) == Float32
@test res ≈ distances_vielsum(a, b)
@test eltype(distances_vielsum(a, b)) == Float32

function distances_cast(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @cast cd1[n] := cos(data1[n,1]) # pulling these out is worth 25%
    @cast cd2[m] := cos(data2[m,1])

    @cast data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

    @cast data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

function distances_cast_avx(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @cast cd1[n] := cos(data1[n,1])  avx
    @cast cd2[m] := cos(data2[m,1])  avx

    @cast data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2  avx

    @cast data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))  avx
end

@test res ≈ distances_cast(a, b)
@test eltype(distances_cast(a, b)) == Float32
@test res ≈ distances_cast_avx(a, b)
@test eltype(distances_cast_avx(a, b)) == Float32

function distances_tullio(data1deg, data2deg)
    data1 = deg2rad.(data1deg)
    data2 = deg2rad.(data2deg)

    @tullio data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
        cos(data1[n,1]) * cos(data2[m,1]) * sin((data1[n,2] - data2[m,2])/2)^2

    @tullio data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
end

# function distances_tullio2(data1deg, data2deg)
#     data1 = deg2rad.(data1deg)
#     data2 = deg2rad.(data2deg)

#     @tullio cd1[n] := cos(data1[n,1]) # has no effect
#     @tullio cd2[m] := cos(data2[m,1])

#     @tullio data[n,m] := sin((data1[n,1] - data2[m,1])/2)^2 +
#         cd1[n] * cd2[m] * sin((data1[n,2] - data2[m,2])/2)^2

#     @tullio data[n,m] = 2 * 6373 * atan(sqrt(abs(data[n,m])), sqrt(abs(1 - data[n,m])))
# end

@test res ≈ distances_tullio(a, b)
@test eltype(distances_tullio(a, b)) == Float32



##### laptop (2 cores, 4 threads)

julia> a = -100 .+ 200 .* rand(Float32, 5000, 2);
julia> b = -100 .+ 200 .* rand(Float32, 5000, 2);

julia> @btime distances($a, $b);
  1.522 s (26 allocations: 286.18 MiB)

julia> @btime distances_threaded($a, $b);
  516.937 ms (64 allocations: 95.45 MiB)

julia> @btime distances_threaded_simd($a, $b);
  215.938 ms (66 allocations: 95.47 MiB)

julia> @btime distances_bcast($a, $b);
  1.352 s (10 allocations: 95.44 MiB)

julia> @btime distances_bcast_simd($a, $b);
  641.506 ms (43 allocations: 95.44 MiB)

julia> @btime distances_einsum($a, $b);
  983.168 ms (10 allocations: 95.48 MiB)

julia> @btime distances_vielsum($a, $b);
  389.831 ms (103 allocations: 95.49 MiB)

julia> @btime distances_cast($a, $b); # unlike distances_bcast, this pulls out cos(...)
  1.034 s (16 allocations: 95.48 MiB)

julia> @btime distances_cast_avx($a, $b); # and this applies more @avx than bcast_simd
  137.557 ms (43 allocations: 190.85 MiB)

julia> @btime distances_tullio($a, $b);
  51.442 ms (636 allocations: 95.47 MiB)



##### desktop (6 cores, 12 threads)

julia> a = -100 .+ 200 .* rand(Float32, 5000, 2);
julia> b = -100 .+ 200 .* rand(Float32, 5000, 2);

julia> @btime distances($a, $b);
  1.166 s (26 allocations: 286.18 MiB)

julia> @btime distances_threaded($a, $b);
  140.062 ms (144 allocations: 95.46 MiB)

julia> @btime distances_threaded_simd($a, $b);
  64.382 ms (147 allocations: 95.48 MiB)

julia> @btime distances_bcast($a, $b);
  1.033 s (10 allocations: 95.44 MiB)

julia> @btime distances_bcast_simd($a, $b);
  501.002 ms (43 allocations: 95.44 MiB)

julia> @btime distances_einsum($a, $b);
  756.749 ms (10 allocations: 95.48 MiB)

julia> @btime distances_vielsum($a, $b);
  108.200 ms (262 allocations: 95.51 MiB)

julia> @btime distances_cast($a, $b); # unlike distances_bcast, this pulls out cos(...)
  795.199 ms (16 allocations: 95.48 MiB)

julia> @btime distances_cast_avx($a, $b); # and this applies more @avx than bcast_simd
  112.824 ms (43 allocations: 190.85 MiB)

julia> @btime distances_tullio($a, $b);
  28.151 ms (788 allocations: 95.48 MiB)




julia> a = -100 .+ 200 .* rand(Float64, 5000, 2); ##### repeat everythinng in Float64
julia> b = -100 .+ 200 .* rand(Float64, 5000, 2);

julia> @btime distances($a, $b);
  1.308 s (26 allocations: 572.36 MiB)

julia> @btime distances_threaded($a, $b);
  146.374 ms (144 allocations: 190.90 MiB)

julia> @btime distances_threaded_simd($a, $b);
  92.097 ms (146 allocations: 190.94 MiB)

julia> @btime distances_bcast($a, $b);
  1.134 s (10 allocations: 190.89 MiB)

julia> @btime distances_bcast_simd($a, $b);
  728.564 ms (43 allocations: 190.89 MiB)

julia> @btime distances_einsum($a, $b);
  874.312 ms (10 allocations: 190.96 MiB)

julia> @btime distances_vielsum($a, $b);
  123.725 ms (262 allocations: 191.00 MiB)

julia> @btime distances_cast($a, $b);
  902.447 ms (16 allocations: 190.96 MiB)

julia> @btime distances_cast_avx($a, $b);
  431.136 ms (43 allocations: 381.70 MiB)

julia> @btime distances_tullio($a, $b);
  75.608 ms (786 allocations: 190.93 MiB)



##### GPU (an ancient one!)

julia> using CuArrays, KernelAbstractions # and then re-run defn. of distances_tullio

julia> CuArrays.allowscalar(false)

julia> ca = cu(a); cb = cu(b); # Float32

julia> cres = distances_bcast(ca, cb);

julia> @test cres ≈ distances_tullio(ca, cb)
Test Passed

julia> @test cres ≈ distances_cast(ca, cb)
Test Passed

julia> @btime CuArrays.@sync distances_bcast($ca, $cb);
  31.558 ms (420 allocations: 18.42 KiB)

julia> @btime CuArrays.@sync distances_cast($ca, $cb);
  29.728 ms (546 allocations: 22.48 KiB)

julia> @btime CuArrays.@sync distances_tullio($ca, $cb);
  187.258 ms (173551 allocations: 2.66 MiB)


##### Python
# From here, verbatim:
# https://gist.github.com/cgarciae/a69fa609f8fcd0aacece92660b5c2315

import typing as tp
from jax import numpy as jnp
import jax
import numpy as np
import time
@jax.jit
def distances_jax(data1, data2):
    # data1, data2 are the data arrays with 2 cols and they hold
    # lat., lng. values in those cols respectively
    np = jnp
    data1 = np.deg2rad(data1)
    data2 = np.deg2rad(data2)
    lat1 = data1[:, 0]
    lng1 = data1[:, 1]
    lat2 = data2[:, 0]
    lng2 = data2[:, 1]
    diff_lat = lat1[:, None] - lat2
    diff_lng = lng1[:, None] - lng2
    d = (
        np.sin(diff_lat / 2) ** 2
        + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(diff_lng / 2) ** 2
    )
    data = 2 * 6373 * np.arctan2(np.sqrt(np.abs(d)), np.sqrt(np.abs(1 - d)))
    return data.reshape(data1.shape[0], data2.shape[0])
def distances_np(data1, data2):
    # data1, data2 are the data arrays with 2 cols and they hold
    # lat., lng. values in those cols respectively
    data1 = np.deg2rad(data1)
    data2 = np.deg2rad(data2)
    lat1 = data1[:, 0]
    lng1 = data1[:, 1]
    lat2 = data2[:, 0]
    lng2 = data2[:, 1]
    diff_lat = lat1[:, None] - lat2
    diff_lng = lng1[:, None] - lng2
    d = (
        np.sin(diff_lat / 2) ** 2
        + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(diff_lng / 2) ** 2
    )
    data = 2 * 6373 * np.arctan2(np.sqrt(np.abs(d)), np.sqrt(np.abs(1 - d)))
    return data.reshape(data1.shape[0], data2.shape[0])
a = np.random.uniform(-100, 100, size=(5000, 2)).astype(np.float32)
b = np.random.uniform(-100, 100, size=(5000, 2)).astype(np.float32)
def dist_np_test():
    return distances_np(a, b)
# enforce eager evaluation
def dist_jax_test():
    return distances_jax(a, b).block_until_ready()


##### Times on the same laptop as above:

Python 3.7.5 (default, Nov  6 2019, 19:41:43)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.9.0 -- An enhanced Interactive Python. Type '?' for help.


In [2]: dist_np_test()
Out[2]:
array([[ 4011.349 , 11679.735 ,  1918.837 , ...,  2963.0593, 13176.956 ,
        15359.288 ],
       ...,
       [10144.612 , 18684.783 ,  6158.844 , ..., 10165.801 , 13639.45  ,
         8931.506 ]], dtype=float32)

In [3]: %timeit dist_np_test()
988 ms ± 3.14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [4]: %timeit dist_jax_test()
/Users/me/.pyenv/versions/3.7.5/lib/python3.7/site-packages/jax/lib/xla_bridge.py:114: UserWarning: No GPU/TPU found, falling back to CPU.
  warnings.warn('No GPU/TPU found, falling back to CPU.')
372 ms ± 14.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [5]: jax.__version__
Out[5]: '0.1.50'

In [6]: np.__version__
Out[6]: '1.17.3'

# Closest versions above are probably these:
# distances_einsum 983.168 ms -- simple loops, single-threaded
# distances_vielsum 389.831 ms -- ditto, multi-threaded

##### Times on the same desktop as above:

Python 3.6.5 (default, Jun 17 2018, 12:13:06)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.13.0 -- An enhanced Interactive Python. Type '?' for help.

In [3]: %timeit dist_np_test()
822 ms ± 4.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [4]: %timeit dist_jax_test()
/Users/me/code/jax19/.direnv/python-3.6.5/lib/python3.6/site-packages/jax/lib/xla_bridge.py:116: UserWarning: No GPU/TPU found, falling back to CPU.
  warnings.warn('No GPU/TPU found, falling back to CPU.')
107 ms ± 1.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [5]: jax.__version__
Out[5]: '0.1.65'

# Again these nearly exactly match the following:
# distances_einsum 756.749 ms
# distances_vielsum 108.200 ms
# and distances_tullio is still faster by a factor of 3.
````

## File: benchmarks/02/matmul.jl
````
using Pkg; Pkg.activate("~/.julia/dev/Tullio/benchmarks/02/")
pkg"add Tullio, LoopVectorization, Compat, PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools, Libdl, MKL_jll, OpenBLAS_jll"
pkg"up"

Threads.nthreads() == 6

using Tullio, LoopVectorization, Compat

TVERSION = VersionNumber(Pkg.TOML.parsefile(joinpath(pkgdir(Tullio), "Project.toml"))["version"])

tmul!(C,A,B) = @tullio C[i,j] := A[i,k] * B[k,j]

# Adapted from:
# https://github.com/chriselrod/PaddedMatrices.jl/blob/master/benchmark/blasbench.jl

using PaddedMatrices, StructArrays, LinearAlgebra, BenchmarkTools, Libdl

randa(::Type{T}, dim...) where {T} = rand(T, dim...)
randa(::Type{T}, dim...) where {T <: Signed} = rand(T(-100):T(200), dim...)

using MKL_jll, OpenBLAS_jll

const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const DGEMM_MKL = Libdl.dlsym(libMKL, :dgemm)
const SGEMM_MKL = Libdl.dlsym(libMKL, :sgemm)
const DGEMV_MKL = Libdl.dlsym(libMKL, :dgemv)
const MKL_SET_NUM_THREADS = Libdl.dlsym(libMKL, :MKL_Set_Num_Threads)

const libOpenBLAS = Libdl.dlopen(OpenBLAS_jll.libopenblas)
const DGEMM_OpenBLAS = Libdl.dlsym(libOpenBLAS, :dgemm_64_)
const SGEMM_OpenBLAS = Libdl.dlsym(libOpenBLAS, :sgemm_64_)
const DGEMV_OpenBLAS = Libdl.dlsym(libOpenBLAS, :dgemv_64_)
const OPENBLAS_SET_NUM_THREADS = Libdl.dlsym(libOpenBLAS, :openblas_set_num_threads64_)

istransposed(x) = 'N'
istransposed(x::Adjoint{<:Real}) = 'T'
istransposed(x::Adjoint) = 'C'
istransposed(x::Transpose) = 'T'

for (lib,f) ∈ [(:GEMM_MKL,:gemmmkl!), (:GEMM_OpenBLAS,:gemmopenblas!)]
    for (T,prefix) ∈ [(Float32,:S),(Float64,:D)]
        fm = Symbol(prefix, lib)
        @eval begin
            function $f(C::AbstractMatrix{$T}, A::AbstractMatrix{$T}, B::AbstractMatrix{$T})
                transA = istransposed(A)
                transB = istransposed(B)
                M, N = size(C); K = size(B, 1)
                pA = parent(A); pB = parent(B)
                ldA = stride(pA, 2)
                ldB = stride(pB, 2)
                ldC = stride(C, 2)
                α = one($T)
                β = zero($T)
                ccall(
                    $fm, Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{$T}, Ref{$T},
                     Ref{Int64}, Ref{$T}, Ref{Int64}, Ref{$T}, Ref{$T}, Ref{Int64}),
                    transA, transB, M, N, K, α, pA, ldA, pB, ldB, β, C, ldC
                )
            end
        end
    end
end
mkl_set_num_threads(N::Integer) = ccall(MKL_SET_NUM_THREADS, Cvoid, (Int32,), N % Int32)
# mkl_set_num_threads(1)
openblas_set_num_threads(N::Integer) = ccall(OPENBLAS_SET_NUM_THREADS, Cvoid, (Int64,), N)
# openblas_set_num_threads(1)

function benchmark_fun!(f!, C, A, B, force_belapsed = false, reference = nothing)
    tmin = @elapsed f!(C, A, B)
    if force_belapsed || 2tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @belapsed $f!($C, $A, $B))
    elseif tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @elapsed f!(C, A, B))
        if tmin < 2BenchmarkTools.DEFAULT_PARAMETERS.seconds
            tmin = min(tmin, @elapsed f!(C, A, B))
        end
    end
    isnothing(reference) || @assert C ≈ reference
    tmin
end

function runbench(::Type{T}, sizes = [2:255..., round.(Int, range(57.16281374121401, length=200) .^ 1.3705658916944428)...]) where {T}
    (StructVector ∘ map)(sizes) do sz
        n, k, m = sz, sz, sz
        C1 = Matrix{T}(undef, n, m)
        C2 = similar(C1);
        C3 = similar(C1);
        C4 = similar(C1);
        # C5 = similar(C1);
        # C6 = similar(C1);
        A  = randa(T, n, k)
        B  = randa(T, k, m)
        # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.05

        tmlt = benchmark_fun!(tmul!, C1, A, B, sz == first(sizes))

        jmlt = benchmark_fun!(PaddedMatrices.jmul!, C1, A, B, sz == first(sizes))
        # res = if T <: Integer
        #     (matrix_size=sz, MaBLAS_24x9=ma24, MaBLAS_32x6=ma32, MaBLAS_40x5=ma40, PaddedMatrices=jmlt)
        # else
            opbt = benchmark_fun!(gemmopenblas!, C2, A, B, sz == first(sizes), C1)
            mklbt= benchmark_fun!(gemmmkl!, C3, A, B, sz == first(sizes), C1)

            res = (matrix_size=sz, OpenBLAS=opbt, MKL=mklbt, PaddedMatrices=jmlt, Tullio=tmlt)
        # end
        @show res
    end
end

b5 = runbench(Float32, [3,10,30,100,300])

using Plots

function makeplot(res, title="")
    plt = plot()
    for lab in propertynames(res)[2:end]
        times = getproperty(res, lab)
        flops = 2e-9 * res.matrix_size.^3 ./ times
        str = lab==:MKL ? "MKL $(Compat.get_num_threads())" :
            lab==:OpenBLAS ? "OpenBLAS $(Compat.get_num_threads())" :
            lab==:Tullio ? "Tullio $(Threads.nthreads())" :
            string(lab)
        lab==:Tullio ?
            plot!(res.matrix_size, flops, lab=str, m=:circle) :
            plot!(res.matrix_size, flops, lab=str)
    end
    plot!(yaxis=("gigaflops", ([12.5,25,50,100,200,400],["12.5",25,50,100,200,400]), :log10), xaxis=("size", :log10), legend=:bottomright)
    # plot!(1:0, 1:0, c=:white, lab="i7-8700 + $VERSION")
    plot!(1:0, 1:0, c=:white, lab="Intel " * split(Sys.cpu_info()[1].model, " ")[3], title=title * "Julia " * string(VERSION))
end

makeplot(b5, "warmup ")

# Threads.nthreads() <= 6 || error("expected to run with at most 6 threads")
mkl_set_num_threads(Threads.nthreads())
openblas_set_num_threads(Threads.nthreads())

for Ty in [Float64, Float32]
    global b36

    b36 = runbench(Ty, [10,11,12, 20,21, 30,31,32,33, 49,50,51, 63,64,65,66, 77,78, 100,101,102, 127,128,129, 200, 255,256,257, 300, 400, 500, 511,512,513, 600, 700, 800, 999,1000,1024,1025, 1600,1601, 1999,2000])

    p36 = makeplot(b36, "$Ty, Tullio $TVERSION, ")
    savefig(p36, joinpath("~/.julia/dev/Tullio", "benchmarks/02/matmul-$TVERSION-$Ty-$VERSION.png"))

end

# Summary:

# it's fine at small sizes, sometimes beating OpenBLAS, thanks entirely to @avx
# Threading helps at large sizes, but it still ends up about half the speed.
# Which is OK, the goal isn't to replace BLAS, it's to do other things!
# This is just a test to see how much is left on the table.
````

## File: benchmarks/02/permute.jl
````
using Pkg; Pkg.activate("~/.julia/dev/Tullio/benchmarks/02/")
pkg"add Tullio, LoopVectorization, LinearAlgebra, BenchmarkTools, Einsum, TensorOperations, MKL_jll, https://github.com/haampie/FastTranspose.jl, https://github.com/mcabbott/ArrayMeta.jl, StructArrays"
pkg"up"

Threads.nthreads() == 6

using Tullio, LoopVectorization, Einsum, TensorOperations, FastTranspose, ArrayMeta,Libdl, Test, StructArrays, BenchmarkTools, Plots

TVERSION = VersionNumber(Pkg.TOML.parsefile(joinpath(pkgdir(Tullio), "Project.toml"))["version"])

# adapted from https://github.com/haampie/FastTranspose.jl/blob/master/benchmark/mkl.jl
using MKL_jll
const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const MKL_domatcopy = Libdl.dlsym(libMKL, :mkl_domatcopy)
const MKL_somatcopy = Libdl.dlsym(libMKL, :mkl_somatcopy)
function mkl_matcopy!(B::Matrix{Float64}, A::Matrix{Float64}, alpha = 1.0)
    m, n = size(A)
    ordering = 'C'
    trans = 'T'
    ccall(MKL_domatcopy, Cvoid, (Ref{Cchar}, Ref{Cchar}, Ref{Csize_t}, Ref{Csize_t}, Ref{Cdouble}, Ptr{Float64}, Ref{Csize_t}, Ptr{Float64}, Ref{Csize_t}),
           ordering, trans, m, n, alpha, A, m, B, n)
    return nothing
end
function mkl_matcopy!(B::Matrix{Float32}, A::Matrix{Float32}, alpha = 1.0)
    m, n = size(A)
    ordering = 'C'
    trans = 'T'
    ccall(MKL_somatcopy, Cvoid, (Ref{Cchar}, Ref{Cchar}, Ref{Csize_t}, Ref{Csize_t}, Ref{Cdouble}, Ptr{Float32}, Ref{Csize_t}, Ptr{Float32}, Ref{Csize_t}),
           ordering, trans, m, n, alpha, A, m, B, n)
    return nothing
end

for T in [Float64, Float32]
    local A,B = rand(T, 3,3), rand(T, 3,3)
    mkl_matcopy!(B,A)
    # @test B == A' # fails for Float32?
end

# matrix transpose

base_transpose!(y,x) = permutedims!(y, x, (2,1))
base_lazy_transpose!(y,x) = copyto!(y, transpose(x))
einsum_transpose!(y,x) = @einsum y[i,j] = x[j,i]
arraymeta_transpose!(y,x) = @arrayop y[i,j] = x[j,i]
tensor_transpose!(y,x) = @tensor y[i,j] = x[j,i]
avx_transpose!(y,x) = @tullio y[i,j] = x[j,i] threads=false tensor=false
tullio_transpose!(y,x) = @tullio y[i,j] = x[j,i] tensor=false

functions2 = (base_transpose!, base_lazy_transpose!,
    einsum_transpose!, arraymeta_transpose!, tensor_transpose!,
    avx_transpose!, tullio_transpose!,
    recursive_transpose!, mkl_matcopy!)
sizes2 = sort(vcat(vec((2 .^ (4:12))' .+ [-1,0,1]), [25, 50, 100, 150, 200, 1500]))

# 3-array permutedims

base_312!(y, x) = permutedims!(y, x, (3,1,2))
base_lazy_312!(y, x) = copyto!(y, PermutedDimsArray(x, (3,1,2)))
einsum_312!(y, x) = @einsum y[c,a,b] = x[a,b,c]
arraymeta_312!(y, x) = @arrayop  y[c,a,b] = x[a,b,c]
tensor_312!(y, x) = @tensor y[c,a,b] = x[a,b,c]
avx_312!(y, x) = @tullio y[c,a,b] = x[a,b,c] threads=false tensor=false
tullio_312!(y, x) = @tullio y[c,a,b] = x[a,b,c] tensor=false

functions3 = (base_312!, base_lazy_312!,
    einsum_312!, arraymeta_312!, tensor_312!,
    avx_312!, tullio_312!)
sizes3 = sort(vcat(vec((2 .^ (3:9))' .+ [-1,0,1] ), [25, 50, 100, 150, 200]))

# 4-array permutedims

base_4321!(y, x) = permutedims!(y, x, (4,3,2,1))
base_lazy_4321!(y, x) = copyto!(y, PermutedDimsArray(x, (4,3,2,1)))
einsum_4321!(y, x) = @einsum y[d,c,b,a] = x[a,b,c,d]
arraymeta_4321!(y, x) = @arrayop y[d,c,b,a] = x[a,b,c,d]
tensor_4321!(y, x) = @tensor y[d,c,b,a] = x[a,b,c,d]
avx_4321!(y, x) = @tullio y[d,c,b,a] = x[a,b,c,d] threads=false tensor=false
tullio_4321!(y, x) = @tullio y[d,c,b,a] = x[a,b,c,d] tensor=false

functions4 = (base_4321!, base_lazy_4321!,
    einsum_4321!, arraymeta_4321!, tensor_4321!,
    avx_4321!, tullio_4321!)
sizes4 = sort(vcat(vec((2 .^ (2:6))' .+ [-1,0,1]), [25, 50, 99, 100, 101])) # crashes on 129!

# based on https://github.com/chriselrod/PaddedMatrices.jl/blob/master/benchmark/blasbench.jl

function benchmark_fun!(f!, C, A, force_belapsed = false, reference = nothing)
    tmin = @elapsed f!(C, A)
    if force_belapsed || 2tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @belapsed $f!($C, $A))
    elseif tmin < BenchmarkTools.DEFAULT_PARAMETERS.seconds
        tmin = min(tmin, @elapsed f!(C, A))
        if tmin < 2BenchmarkTools.DEFAULT_PARAMETERS.seconds
            tmin = min(tmin, @elapsed f!(C, A))
        end
    end
    isnothing(reference) || @assert C ≈ reference
    tmin
end

function runbench(funs::Tuple, dims::Int=2, sizes=[10,30,100])
    (StructVector ∘ map)(sizes) do n
        A = rand(ntuple(_->n, dims)...)
        C0, C = similar(A), similar(A)
        funs[1](C0, A)
        times = map(funs) do f!
            t = benchmark_fun!(f!, C, A)
            C ≈ C0 || @error "disagreement for $f! at size $n"
            t
        end
        nt = NamedTuple{map(Symbol, funs)}(times)
        res = (size=n, length=length(C), nt...)
        @show res
    end
end

r0 = runbench((base_transpose!, mkl_matcopy!, tullio_transpose!), 2, [10,30,100])

function makeplot(res, title="")
    plt = plot()
    for lab in propertynames(res)[3:end]
        times = getproperty(res, lab)
        flops = 1e-9 * res.length ./ times # was wrong in 0.2.0 plots

        if startswith(string(lab), "tullio")
            plot!(plt, res.size, flops, lab=string(lab), m=:circle)
        else
            plot!(plt, res.size, flops, lab=string(lab))
        end
    end
    plot!(plt, yaxis=("numbers / ns", ([0.5,1,2,4,8], ["1/2","1",2,4,8]), :log10), xaxis=("size per dimension", :log10), legend=:bottomleft)
    plot!(plt, title = title * "Julia " * string(VERSION) * ", Intel " * split(Sys.cpu_info()[1].model)[3])
end

makeplot(r0)
# savefig(joinpath("~/.julia/dev/Tullio", "benchmarks/02/trash.png"))

res2 = runbench(functions2, 2, sizes2)
plot2 = makeplot(res2, "Tullio $TVERSION, ")
savefig(plot2, joinpath("~/.julia/dev/Tullio", "benchmarks/02/transpose-$TVERSION-$VERSION.png"))

res3 = runbench(functions3, 3, sizes3)
plot3 = makeplot(res3, "Tullio $TVERSION, ")
savefig(plot3, joinpath("~/.julia/dev/Tullio", "benchmarks/02/permute3-$TVERSION-$VERSION.png"))

res4 = runbench(functions4, 4, sizes4)
plot4 = makeplot(res4, "Tullio $TVERSION, ")
savefig(plot4, joinpath("~/.julia/dev/Tullio", "benchmarks/02/permute4-$TVERSION-$VERSION.png"))
````

## File: benchmarks/02/Project.toml
````toml
[deps]
ArrayMeta = "ff776444-5e7a-51eb-9a1c-dda7296cec34"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Einsum = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
FastTranspose = "c672cb4e-dd10-4a00-b22f-8edf27ea5534"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
MKL_jll = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
TensorOperations = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
Tullio = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
````

## File: ext/TullioChainRulesCoreExt.jl
````
module TullioChainRulesCoreExt

using Tullio, ChainRulesCore

function ChainRulesCore.rrule(ev::Tullio.Eval, args...)
    Z = ev.fwd(args...)
    Z, function tullio_back(Δ)
        isnothing(ev.rev) && error("no gradient definition here!")
        dxs = map(ev.rev(unthunk(Δ), Z, args...)) do dx
            dx === nothing ? ChainRulesCore.ZeroTangent() : dx
        end
        tuple(ChainRulesCore.ZeroTangent(), dxs...)
    end
end

end
````

## File: ext/TullioCUDAExt.jl
````
module TullioCUDAExt

if !isdefined(Base, :get_extension)
    using ..Tullio, ..CUDA
else
    using Tullio, CUDA
end

Tullio.threader(fun!::F, ::Type{T},
    Z::AbstractArray, As::Tuple, Is::Tuple, Js::Tuple,
    redfun, block=0, keep=nothing) where {F<:Function, T<:CUDA.CuArray} =
    fun!(T, Z, As..., Is..., Js..., keep)

Tullio.∇threader(fun!::F, ::Type{T},
    As::Tuple, Is::Tuple, Js::Tuple, block=0) where {F<:Function, T<:CUDA.CuArray} =
    fun!(T, As..., Is..., Js...,)

# Tullio.thread_scalar ... ought to work? Was never fast.

end
````

## File: ext/TullioFillArraysExt.jl
````
module TullioFillArraysExt

if !isdefined(Base, :get_extension)
    using ..Tullio, ..FillArrays
else
    using Tullio, FillArrays
end

Tullio.promote_storage(::Type{T}, ::Type{F}) where {T, F<:FillArrays.Fill} = T
Tullio.promote_storage(::Type{F}, ::Type{T}) where {T, F<:FillArrays.Fill} = T

end
````

## File: ext/TullioTrackerExt.jl
````
module TullioTrackerExt

if !isdefined(Base, :get_extension)
    using ..Tullio, ..Tracker
else
    using Tullio, Tracker
end

(ev::Tullio.Eval)(A::Tracker.TrackedArray, args...) = Tracker.track(ev, A, args...)
(ev::Tullio.Eval)(A, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)
(ev::Tullio.Eval)(A::Tracker.TrackedArray, B::Tracker.TrackedArray, args...) = Tracker.track(ev, A, B, args...)

Tracker.@grad function (ev::Tullio.Eval)(args...)
    Z = ev.fwd(Tracker.data.(args)...)
    Z, Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        tuple(ev.rev(Tracker.data(Δ), Z, Tracker.data.(args)...)...)
    end
end

end
````

## File: src/grad/avxdual.jl
````
#========== making ForwardDiff work with LoopVectorization ==========#

using .LoopVectorization
using Core: VecElement

using .ForwardDiff
using .ForwardDiff: Dual, Partials, partials

#=
# using Tullio.LoopVectorization: LoopVectorization, SVec, vconvert, SIMDPirates

s1 = SVec{2,Float64}(5.5, 6.6) # SVec{2,Float64}<5.5, 6.6>
# dump(s1)
# SVec{2,Float64}
#   data: Tuple{VecElement{Float64},VecElement{Float64}}
#     1: VecElement{Float64}
#       value: Float64 5.5
#     2: VecElement{Float64}
#       value: Float64 6.6
s1[2]
s1 |> typeof |> parentmodule # VectorizationBase

# @inline svec(tup::NTuple{N,T}) where {N,T} = SVec{N,T}(tup...)

d1 = Dual(1.23, (4,0,0))
typeof(d1) # Dual{Nothing,Float64,3}
# dump(d1)
# Dual{Nothing,Float64,2}
#   value: Float64 1.23
#   partials: Partials{2,Float64}
#     values: Tuple{Float64,Float64}
#       1: Float64 4.0
#       2: Float64 0.0
#       3: Float64 0.0
d1.partials # Partials{3,Float64}
d1.partials[1]

partials(d1, 1)
# @inline val(d::Dual) = d.value

=#

ForwardDiff.can_dual(::Type{<:SVec}) = true

@inline function Base.:+(x::Dual{Z,T,D}, sv::SVec{N,S}) where {Z,T<:Number,D,N,S}
    y = x.value + sv
    ps = ntuple(d -> x.partials.values[d] + zero(sv), Val(D))
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,TS,D}(y, Partials{D,TS}(ps))
end
@inline function Base.:+(sv::SVec{N,S}, x::Dual{Z,T,D}) where {Z,T<:Number,D,N,S}
    y = x.value + sv
    ps = ntuple(d -> x.partials.values[d] + zero(sv), Val(D))
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,TS,D}(y, Partials{D,TS}(ps))
end

@inline function Base.:*(x::Dual{Z,SVec{N,T},D}, sv::SVec{N,S}) where {Z,T,D,N,S}
    y = x.value * sv
    ps = ntuple(d -> x.partials.values[d] * sv, Val(D))
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,typeof(y),D}(y, Partials{D,typeof(y)}(ps))
end
@inline function Base.:*(sv::SVec{N,S}, x::Dual{Z,SVec{N,T},D}) where {Z,T,D,N,S}
    y = sv * x.value
    ps = ntuple(d -> sv * x.partials.values[d], Val(D))
    TS = SVec{N,promote_type(T,S)}
    Dual{Z,TS,D}(y, Partials{D,TS}(ps))
end

@inline function Base.:*(p::Partials{D,SVec{N,T}}, sv::SVec{N,S}) where {T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> p.values[d] * sv, Val(D)))
end
@inline function Base.:*(sv::SVec{N,S}, p::Partials{D,SVec{N,T}}) where {T,D,N,S}
    TS = SVec{N,promote_type(T,S)}
    Partials{D,TS}(ntuple(d -> sv * p.values[d], Val(D)))
end

#========== the end ==========#
````

## File: src/grad/reverse.jl
````
using .ReverseDiff

(ev::Eval)(A::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, args...)
(ev::Eval)(A, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, B, args...)
(ev::Eval)(A::ReverseDiff.TrackedArray, B::ReverseDiff.TrackedArray, args...) = ReverseDiff.track(ev, A, B, args...)

ReverseDiff.@grad function (ev::Eval)(args...)
    Z = ev.fwd(ReverseDiff.value.(args)...)
    Z, Δ -> begin
        isnothing(ev.rev) && error("no gradient definition here!")
        ev.rev(ReverseDiff.value(Δ), Z, ReverseDiff.value.(args)...)
    end
end
````

## File: src/precompile/precompile_Base.jl
````
const __bodyfunction__ = Dict{Method,Any}()

# Find keyword "body functions" (the function that contains the body
# as written by the developer, called after all missing keyword-arguments
# have been assigned values), in a manner that doesn't depend on
# gensymmed names.
# `mnokw` is the method that gets called when you invoke it without
# supplying any keywords.
function __lookup_kwbody__(mnokw::Method)
    function getsym(arg)
        isa(arg, Symbol) && return arg
        @assert isa(arg, GlobalRef)
        return arg.name
    end

    f = get(__bodyfunction__, mnokw, nothing)
    if f === nothing
        fmod = mnokw.module
        # The lowered code for `mnokw` should look like
        #   %1 = mkw(kwvalues..., #self#, args...)
        #        return %1
        # where `mkw` is the name of the "active" keyword body-function.
        ast = Base.uncompressed_ast(mnokw)
        if isa(ast, Core.CodeInfo) && length(ast.code) >= 2
            callexpr = ast.code[end-1]
            if isa(callexpr, Expr) && callexpr.head == :call
                fsym = callexpr.args[1]
                if isa(fsym, Symbol)
                    f = getfield(fmod, fsym)
                elseif isa(fsym, GlobalRef)
                    if fsym.mod === Core && fsym.name === :_apply
                        f = getfield(mnokw.module, getsym(callexpr.args[2]))
                    elseif fsym.mod === Core && fsym.name === :_apply_iterate
                        f = getfield(mnokw.module, getsym(callexpr.args[3]))
                    else
                        f = getfield(fsym.mod, fsym.name)
                    end
                else
                    f = missing
                end
            else
                f = missing
            end
        else
            f = missing
        end
        __bodyfunction__[mnokw] = f
    end
    return f
end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(cat),Expr,Vararg{Any}})
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(cat),Vector{Symbol},Vararg{Any}})
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat_t)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(Base.cat_t),Type{Any},Expr,Vararg{Any}})
    Base.precompile(Tuple{Core.kwftype(typeof(Base.cat_t)),NamedTuple{(:dims,), Tuple{Val{1}}},typeof(Base.cat_t),Type{Any},Vector{Symbol},Vararg{Any}})
    Base.precompile(Tuple{Type{Dict{Symbol, Any}},NTuple{37, Pair{Symbol, Any}}})
    Base.precompile(Tuple{Type{Dict{Symbol, Any}},Pair{Symbol, Symbol},Vararg{Pair}})
    Base.precompile(Tuple{Type{Dict},Pair{Symbol, Any},Vararg{Pair{Symbol, Any}}})
    Base.precompile(Tuple{Type{Pair},Int64,Expr})
    Base.precompile(Tuple{Type{Pair},Int64,Symbol})
    Base.precompile(Tuple{typeof(==),Bool,Symbol})
    Base.precompile(Tuple{typeof(==),Symbol,Bool})
    Base.precompile(Tuple{typeof(>),Bool,Int64})
    Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Int64,Base.OneTo{Int64}})
    Base.precompile(Tuple{typeof(Base.Broadcast.broadcasted),Function,Tuple{Expr, Expr}})
    Base.precompile(Tuple{typeof(Base.__cat),Vector{Any},Tuple{Int64},Tuple{Bool},Expr,Vararg{Any}})
    Base.precompile(Tuple{typeof(Base.__cat),Vector{Any},Tuple{Int64},Tuple{Bool},Vector{Symbol},Vararg{Any}})
    Base.precompile(Tuple{typeof(Base._any),Base.Fix2{typeof(==), Symbol},Vector{Symbol},Colon})
    Base.precompile(Tuple{typeof(Base._array_for),Type{Expr},Base.Iterators.Enumerate{Vector{Any}},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._array_for),Type{Expr},Base.Iterators.Enumerate{Vector{Expr}},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._array_for),Type{LineNumberNode},Vector{Any},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._array_for),Type{Symbol},Base.Iterators.Enumerate{Vector{Any}},Base.HasShape{1}})
    Base.precompile(Tuple{typeof(Base._cat),Val{1},Vector{Symbol},Vararg{Any}})
    Base.precompile(Tuple{typeof(Base._nt_names),Type{NamedTuple{(:redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), Tuple{Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}}}}})
    Base.precompile(Tuple{typeof(Base._shrink),Function,Vector{Symbol},Tuple{Vector{Symbol}}})
    Base.precompile(Tuple{typeof(Base.cat_indices),Symbol,Int64})
    Base.precompile(Tuple{typeof(Base.cat_similar),Expr,Type,Tuple{Int64}})
    Base.precompile(Tuple{typeof(Base.cat_similar),Vector{Symbol},Type,Tuple{Int64}})
    Base.precompile(Tuple{typeof(Base.cat_size),Symbol,Int64})
    Base.precompile(Tuple{typeof(Base.indexed_iterate),Tuple{Nothing, Nothing},Int64})
    Base.precompile(Tuple{typeof(Base.indexed_iterate),Tuple{Symbol, Symbol, Expr, Expr},Int64})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Expr,Vector{Symbol},Vararg{Vector{Symbol}}})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Symbol,Vector{Symbol},Vararg{Vector{Symbol}}})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Vector{Symbol},Expr,Vararg{Any}})
    Base.precompile(Tuple{typeof(Base.promote_eltypeof),Vector{Symbol}})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Expr},LineNumberNode,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Expr},Symbol,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{LineNumberNode},Expr,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Nothing},LineNumberNode,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Symbol},Expr,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Symbol},Int64,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Symbol},QuoteNode,Int64})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{Union{Nothing, LineNumberNode}},Expr,Int64})
    Base.precompile(Tuple{typeof(Base.vectorfilter),Function,Vector{Symbol}})
    Base.precompile(Tuple{typeof(Core.Compiler.eltype),Type{Vector{Base.HasShape{1}}}})
    Base.precompile(Tuple{typeof(allunique),Vector{Symbol}})
    Base.precompile(Tuple{typeof(any),Function,Vector{Symbol}})
    Base.precompile(Tuple{typeof(append!),Vector{Expr},Vector{Expr}})
    Base.precompile(Tuple{typeof(append!),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(Base.cat_shape),Tuple{Bool},NTuple{4, Tuple{Int64}}})
    Base.precompile(Tuple{typeof(Base.cat_shape),Tuple{Bool},NTuple{6, Tuple{Int64}}})
    Base.precompile(Tuple{typeof(collect),Tuple{Symbol, Symbol}})
    Base.precompile(Tuple{typeof(convert),Type{Vector{Any}},Vector{Expr}})
    Base.precompile(Tuple{typeof(convert),Type{Vector{Any}},Vector{Symbol}})
    Base.precompile(Tuple{typeof(convert),Type{Vector{Symbol}},Vector{Expr}})
    Base.precompile(Tuple{typeof(enumerate),Vector{Expr}})
    Base.precompile(Tuple{typeof(enumerate),Vector{Symbol}})
    Base.precompile(Tuple{typeof(get!),Dict{Symbol, Vector{T} where T},Symbol,Vector{Any}})
    Base.precompile(Tuple{typeof(getindex),Dict{Symbol, Vector{T} where T},Symbol})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, Expr},UInt64})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, Int64},UInt64})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, String},UInt64})
    Base.precompile(Tuple{typeof(hash),Pair{Int64, Symbol},UInt64})
    Base.precompile(Tuple{typeof(haskey),Dict{Symbol, Any},Symbol})
    Base.precompile(Tuple{typeof(haskey),Dict{Symbol, Vector{T} where T},Symbol})
    Base.precompile(Tuple{typeof(in),Symbol,Set{Any}})
    Base.precompile(Tuple{typeof(in),Tuple{Expr, Expr},Set{Any}})
    Base.precompile(Tuple{typeof(intersect),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(isassigned),Vector{Symbol}})
    Base.precompile(Tuple{typeof(isequal),Expr})
    Base.precompile(Tuple{typeof(isequal),Int64})
    Base.precompile(Tuple{typeof(isequal),String})
    Base.precompile(Tuple{typeof(isequal),Symbol})
    Base.precompile(Tuple{typeof(iterate),Base.Iterators.Pairs{Symbol, Any, NTuple{37, Symbol}, NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}},Int64})
    Base.precompile(Tuple{typeof(iterate),Base.Iterators.Pairs{Symbol, Any, NTuple{37, Symbol}, NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}}})
    Base.precompile(Tuple{typeof(map),Function,Base.Iterators.Enumerate{Vector{Any}}})
    Base.precompile(Tuple{typeof(map),Function,Base.Iterators.Enumerate{Vector{Expr}}})
    Base.precompile(Tuple{typeof(map),Function,Base.Iterators.Enumerate{Vector{Symbol}}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Any},Vector{Symbol}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Expr}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(map),Function,Vector{Symbol}})
    Base.precompile(Tuple{typeof(map),typeof(Base.cat_size),Tuple{Expr, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}}})
    Base.precompile(Tuple{typeof(map),typeof(Base.cat_size),Tuple{Vector{Symbol}, Expr, Symbol, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}}})
    Base.precompile(Tuple{typeof(merge),NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}}},NamedTuple{(:flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}})
    Base.precompile(Tuple{typeof(merge),NamedTuple{(:mod,), Tuple{Module}},NamedTuple{(:redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), Tuple{Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}}}})
    Base.precompile(Tuple{typeof(pairs),Base.Iterators.Pairs{Symbol, Any, NTuple{37, Symbol}, NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}}})
    Base.precompile(Tuple{typeof(pairs),NamedTuple{(:mod, :redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd, :flags, :redind, :init, :leftraw, :leftind, :leftarray, :leftscalar, :leftnames, :right, :finaliser, :rightind, :sharedind, :unsafeleft, :unsaferight, :arrays, :scalars, :cost, :constraints, :pairconstraints, :notfree, :shiftedind, :axisdefs, :padmodclamp, :outpre, :outex), Tuple{Module, Symbol, Symbol, Symbol, Bool, Bool, Bool, Symbol, Bool, Int64, Bool, Vector{Symbol}, Set{Symbol}, Vector{Symbol}, Nothing, Vector{Any}, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Nothing, Nothing, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Vector{Symbol}, Int64, Dict{Symbol, Vector{T} where T}, Vector{Tuple}, Vector{Symbol}, Vector{Symbol}, Vector{Expr}, Bool, Vector{Expr}, Vector{Expr}}}})
    Base.precompile(Tuple{typeof(promote_type),Type{Symbol},Type{Any}})
    Base.precompile(Tuple{typeof(push!),Set{Any},Expr})
    Base.precompile(Tuple{typeof(push!),Set{Any},Tuple{Expr, Expr}})
    Base.precompile(Tuple{typeof(push!),Vector{Symbol},Symbol,Symbol})
    Base.precompile(Tuple{typeof(push!),Vector{Tuple},Tuple{Symbol, Symbol, Expr, Expr}})
    Base.precompile(Tuple{typeof(setdiff),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Dict{Symbol, Vector{T} where T},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Nothing,Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Set{Symbol},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Any},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Expr},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Symbol},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Dict{Symbol, Any},Vector{Tuple},Symbol})
    Base.precompile(Tuple{typeof(setindex!),Vector{Any},Vector{Symbol},UnitRange{Int64}})
    Base.precompile(Tuple{typeof(union),Vector{Symbol},Vector{Symbol}})
    Base.precompile(Tuple{typeof(unique!),Vector{Expr}})
    Base.precompile(Tuple{typeof(vcat),Expr,Vector{Symbol},Vector{Symbol},Vararg{Vector{Symbol}}})
    Base.precompile(Tuple{typeof(vcat),Vector{Symbol},Expr,Symbol,Vararg{Any}})
    Base.precompile(Tuple{typeof(vcat),Vector{Symbol},Vector{Symbol}})
    isdefined(Base, Symbol("#73#74")) && Base.precompile(Tuple{getfield(Base, Symbol("#73#74")),Expr})
    isdefined(Base, Symbol("#73#74")) && Base.precompile(Tuple{getfield(Base, Symbol("#73#74")),Int64})
    isdefined(Base, Symbol("#73#74")) && Base.precompile(Tuple{getfield(Base, Symbol("#73#74")),Symbol})
    let fbody = try __lookup_kwbody__(which(any, (Function,Vector{Symbol},))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Function,typeof(any),Function,Vector{Symbol},))
        end
    end
end
````

## File: src/precompile/precompile_Core.jl
````
function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Type{NamedTuple{(:mod,), T} where T<:Tuple},Tuple{Module}})
    Base.precompile(Tuple{Type{NamedTuple{(:redfun, :initkeyword, :padkeyword, :verbose, :fastmath, :threads, :grad, :avx, :cuda, :tensor, :nograd), T} where T<:Tuple},Core.Tuple{Core.Symbol, Core.Symbol, Core.Symbol, Core.Bool, Core.Bool, Core.Bool, Core.Symbol, Core.Bool, Core.Int64, Core.Bool, Base.Vector{Core.Symbol}}})
    Base.precompile(Tuple{typeof(Core.Compiler.getindex),Base.Vector{Core.Compiler.CallMeta},Int64})
    Base.precompile(Tuple{typeof(Core.Compiler.length),Base.Vector{Core.Compiler.CallMeta}})
end
````

## File: src/precompile/precompile_Tullio.jl
````
function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(Tullio.Type)),Any,Type{Tullio.DotDict}})
    Base.precompile(Tuple{Type{Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple}, Axes, F, Args} where Args<:Tuple where F where Axes},typeof(Tullio.dollarstrip),Tuple{Tuple{Expr, Expr}}})
    Base.precompile(Tuple{typeof(Base.Broadcast.instantiate),Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple}, Nothing, typeof(Tullio.dollarstrip), Tuple{Tuple{Expr, Expr}}}})
    Base.precompile(Tuple{typeof(Base.setindex_widen_up_to),Vector{typeof(Tullio.subranges)},Expr,Int64})
    Base.precompile(Tuple{typeof(Tullio._trymatch),Any,Any})
    Base.precompile(Tuple{typeof(Tullio._trymatch),Expr,Union{Val{:ref}, Val{:curly}}})
    Base.precompile(Tuple{typeof(Tullio._tullio),Any,Vararg{Any}})
    Base.precompile(Tuple{typeof(Tullio._tullio),Any})
    Base.precompile(Tuple{typeof(Tullio.action_functions),Any})
    Base.precompile(Tuple{typeof(Tullio.arrayplusepsilon),Symbol,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.callcost),Symbol,Tullio.DotDict})
    Base.precompile(Tuple{typeof(Tullio.csewalk),Any,Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.csewalk),Expr,Dict,Vector{T} where T,Set})
    Base.precompile(Tuple{typeof(Tullio.illegal),Any,Any})
    Base.precompile(Tuple{typeof(Tullio.illegal),Expr,Any})
    Base.precompile(Tuple{typeof(Tullio.index_ranges),Any})
    Base.precompile(Tuple{typeof(Tullio.make_many_actors),Any,Any,Any,Vector{T} where T,Any,Vector{T} where T,Any,Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.make_many_actors),Any,Any,Any,Vector{T} where T,Any,Vector{T} where T,Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.output_array),Any})
    Base.precompile(Tuple{typeof(Tullio.padmodclamp_ind),Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.padmodclamp_ind),Expr,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.padmodclamp_pair),Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.padmodclamp_replace),Expr,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.parse_input),Any,Any})
    Base.precompile(Tuple{typeof(Tullio.parse_ranges),Any,Any})
    Base.precompile(Tuple{typeof(Tullio.postwalk),Any,Any})
    Base.precompile(Tuple{typeof(Tullio.range_expr_walk),Any,Expr,Any})
    Base.precompile(Tuple{typeof(Tullio.resolveintersect),Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.resolvestrict),Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.saveconstraints),Any,Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.simplitimes),Number,Any})
    Base.precompile(Tuple{typeof(Tullio.tidyleftraw),Any,Any})
    Base.precompile(Tuple{typeof(Tullio.walk),Any,Any,Any})
    Base.precompile(Tuple{typeof(Tullio.walk),Expr,Any,Any})
    Base.precompile(Tuple{typeof(copy),Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple}, Nothing, typeof(Tullio.dollarstrip), Tuple{Tuple{Expr, Expr}}}})
    isdefined(Tullio, Symbol("#113#114")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#113#114")),Any})
    isdefined(Tullio, Symbol("#139#141")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#139#141")),Any})
    isdefined(Tullio, Symbol("#140#142")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#140#142")),Any})
    isdefined(Tullio, Symbol("#23#24")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#23#24")),Any})
    isdefined(Tullio, Symbol("#25#26")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#25#26")),Any})
    isdefined(Tullio, Symbol("#31#34")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#31#34")),Any})
    isdefined(Tullio, Symbol("#37#38")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#37#38")),Any})
    isdefined(Tullio, Symbol("#39#40")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#39#40")),Any})
    isdefined(Tullio, Symbol("#59#62")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#59#62")),Any})
    isdefined(Tullio, Symbol("#61#64")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#61#64")),Any,Any})
    isdefined(Tullio, Symbol("#65#67")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#65#67")),Any})
    isdefined(Tullio, Symbol("#66#68")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#66#68")),Any})
    isdefined(Tullio, Symbol("#75#77")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#75#77")),Any})
    isdefined(Tullio, Symbol("#76#78")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#76#78")),Any})
    isdefined(Tullio, Symbol("#79#84")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#79#84")),Any})
    isdefined(Tullio, Symbol("#80#85")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#80#85")),Any})
    isdefined(Tullio, Symbol("#81#86")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#81#86")),Any})
    isdefined(Tullio, Symbol("#82#87")) && Base.precompile(Tuple{getfield(Tullio, Symbol("#82#87")),Any,Any})
end
````

## File: src/einsum.jl
````
"""
    Tullio.@einsum  A[i,j] += B[i] * C[j]

Since this package is almost superset of `Einsum.jl`, you can probable drop that and
write `using Tullio: @einsum` to use the new macro under the old name.

Differences:
* Updating `A` with weird things like `*=` uses an awful hack which may be less efficient,
  but does make tests pass!
* Options `threads=false, avx=false, grad=false` are selected for you.
"""
macro einsum(ex::Expr)
    if ex.head in [:(:=), :(=), :(+=)]
        _tullio(ex, :(avx=false), :(threads=false), :(grad=false); mod=__module__)

    elseif ex.head in [:(-=), :(*=), :(/=)]
        @gensym tmp

        if @capture_(ex.args[1], Z_[ijk__]) # array *= ...
            act = Expr(:(:=), :($tmp[$(ijk...)]), ex.args[2:end]...)
            res = _tullio(act, :(avx=false), :(threads=false), :(grad=false); mod=__module__).args[1]
            Expr(Symbol(string(".", ex.head)), Z, res) |> esc

        elseif ex.args[1] isa Symbol # scalar case
            Z = ex.args[1]
            act = Expr(:(:=), :($tmp), ex.args[2:end]...)
            res = _tullio(act, :(avx=false), :(threads=false), :(grad=false); mod=__module__).args[1]
            Expr(ex.head, Z, res) |> esc

        end
    end
end
````

## File: src/eval.jl
````
#========== master evaluator ==========#

"""
    Eval(fwd, rev)
    (e::Eval)(A,B) = fwd(A,B)

This holds the functions `$MAKE` which creates the output array
(before calling `threader($ACT!,...)` to fill it)
and the function `∇$MAKE` which does the reverse pass for differentiation.

It exists so that gradient hooks for various packages can be attached to this,
once. Then `$MAKE` need only be defined in local scope.
"""
struct Eval{F,R}
    fwd::F
    rev::R
end

(ev::Eval)(args...) = ev.fwd(args...)

#========== scalar struct ==========#

"""
    OneBox(val)

Trivial 1-element vector, used for scalar redutcions,
to pass the eltype to `∇$ACT!(AT, 𝛥A, ::AbstractArray{$TYP}, ...)`
"""
struct OneBox{T} <: AbstractVector{T}
    val::T
end
Base.size(::OneBox) = (1,)
Base.getindex(o::OneBox, i::Integer...) = o.val

#========== gradient hooks ==========#
# Macros like @adjoint need to be hidden behind include(), it seems:

# @init @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include("grad/reverse.jl")

if !isdefined(Base, :get_extension)
    using Requires
    include("../ext/TullioChainRulesCoreExt.jl")
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("../ext/TullioTrackerExt.jl")
        @require FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b" include("../ext/TullioFillArraysExt.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/TullioCUDAExt.jl")
    end
end

#========== vectorised gradients ==========#

@inline onlyone(cond::Bool) = cond
@inline onlyone(cond::Bool, seen::Int) = cond && iszero(seen)

@inline anyone(cond::Bool) = cond

#=

@init @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" begin
    using .LoopVectorization # version 0.9+ only now
    using .LoopVectorization.VectorizationBase: Vec, Mask, prevpow2
    SVec{N,T} = Vec{N,T}
    end
    # Functions needed for safe vectorised max gradient
    @inline Tullio.onlyone(cond::Bool, seen::SVec) = cond && allzero(seen)

    @inline Tullio.onlyone(cond::Mask{W}) where {W} = Mask{W}(prevpow2(cond.u))
    @inline Tullio.onlyone(cond::Mask, seen::Union{Int,SVec}) =
        Tullio.allzero(seen) ? Tullio.onlyone(cond) : zero(cond)

    @inline allzero(seen::Integer) = iszero(seen)
    @inline allzero(seen::SVec) = iszero((!iszero(seen)).u)

    @inline Tullio.anyone(cond::Mask) = !iszero(cond.u)
end

=#

#========== storage unwrapper ==========#

"""
    storage_type(adjoint(view(A,...))) == Array{Int,2}
    storage_type(A, B, C) == Array{Int,N} where N

Recursively unwraps wrappers, and combines approximately with `promote_type`.
(Used as the trait to send CuArray to KernelAbstractions and Array{Float or Int}
to LoopVectorization.)
"""
function storage_type(A::AbstractArray)
    P = parent(A)
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end
storage_type(A) = typeof(A)
storage_type(A, Bs...) = promote_storage(storage_type(A), storage_type(Bs...))
storage_type() = AbstractArray

promote_storage(::Type{A}, ::Type{B}) where {A <: Array{T,N}, B <: Array{S,M}} where {T,N,S,M} =
    N==M ? Array{promote_type(T,S), N} : Array{promote_type(T,S)}
promote_storage(::Type{A}, ::Type{B}) where {A <: Array{T,N}, B <: AbstractRange{S}} where {T,N,S} =
    N==1 ? Vector{promote_type(T,S)} : Array{promote_type(T,S)}
promote_storage(::Type{A}, ::Type{B}) where {A <: AbstractRange{T}, B <: Array{S,M}} where {T,S,M} =
    M==1 ? Vector{promote_type(T,S)} : Array{promote_type(T,S)}
promote_storage(T, S) = promote_type(T, S)

#========== fin ==========#
````

## File: src/forward.jl
````
#========== backward gradient using ForwardDiff ==========#

function insert_forward_gradient(axislist, store)
    store.finaliser == :identity || error("can't use grad=Dual with |> finaliser")

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, ACT!)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)
    nonshared = setdiff(vcat(store.leftind, store.redind), store.sharedind)

    epsilondict = Dict{Symbol,Expr}()

    epsilonright = MacroTools_postwalk(epsilonwalk(epsilondict, store), store.right)
    # epsilonright = MacroTools_postwalk(epsilonwalk(epsilondict, store.scalars), store.right)

    defineepsilons, readepsilons = [], []
    for (d, (Aepsilon, Aex)) in enumerate(epsilondict)
        basis = [i==d ? :($one($TYP)) : :($zero($TYP)) for i in 1:length(epsilondict)]
        push!(defineepsilons, :($Aepsilon = ForwardDiff.Dual($zero($TYP), ($(basis...),))))
        push!(readepsilons, :($Aex = $Aex + ForwardDiff.partials($ZED, $d) * $dZ[$(store.leftraw...)]))
    end

    if isempty(defineepsilons) # short-circuit
        push!(store.outpre, :(local @inline $∇act!(::Type, args...) = nothing ))
        store.verbose > 0 && @info "no gradient to calculate"
        return nothing
    end

    ex_iter = :($ZED = $(epsilonright); $(readepsilons...))

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), ZED, store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        :(($(defineepsilons...);)), store.sharedind, nothing, nonshared, ex_iter, nothing, store, "(gradient using ForwardDiff)")

    if isdefined(store.mod, :Zygote) && !(store.scalar)
        ex_iter2 = fillarrayreplace(ex_iter, dZ)
        ex_value = :($(Symbol(dZ, :_value)) = $dZ.value)

        make_many_actors(∇act!,
            vcat(gradarrays, :($dZ::Zygote.Fill{$TYP}), ZED, store.arrays, store.scalars, axislist),
            :(($(defineepsilons...); $ex_value)), store.sharedind, nothing, nonshared, ex_iter2, nothing, store, "(gradient method for FillArrays)")

        # push!(store.outeval, quote
        #     Tullio.promote_storage(T::Type, ::Type{<:Zygote.Fill}) = T
        #     Tullio.promote_storage(::Type{<:Zygote.Fill}, T::Type) = T
        # end)
    end

end

epsilonwalk(dict, store) = ex -> begin
# epsilonwalk(dict, scalars) = ex -> begin
#         ex isa Symbol && ex in scalars && return scalarplusepsilon(ex, dict)
        @capture_(ex, A_[inds__]) || return ex
        A in store.nograd && return ex
        return arrayplusepsilon(A, inds, dict)
    end

arrayplusepsilon(A::Symbol, inds, dict) = begin # the same array may occur twice!
    Aepsilon = Symbol(EPS, A)
    while haskey(dict, Aepsilon)
        Aepsilon = Symbol(Aepsilon, "′")
    end
    dict[Aepsilon] = :( $(Symbol(DEL, A))[$(inds...)] )
    :(( $A[$(inds...)] + $Aepsilon ))
end
arrayplusepsilon(A, inds, dict) = begin
    @debug string("expression ", A, " is why you can't use ForwardDiff here")
    :🐳
end

# scalarplusepsilon(A::Symbol, dict) = begin
#     Aepsilon = Symbol(EPS, A)
#     dict[Aepsilon] = Symbol(DEL, A)
#     :(( $A + $Aepsilon ))
# end

#========== the end ==========#
````

## File: src/macro.jl
````
#========== the macro! ==========#

"""
    @tullio C[i,k] := A[i,j] * B[j,k]
    @tullio F[i,k] := \$α * D[i].field[j] * E[col=k, row=j] + \$β

This is a replacement for `@einsum` which understands a bit more syntax.
The expression on the right is summed over all possible valued of the free index `k`,
and `:=` makes a new array `C`, while `=` and `+=` would write into an existing one.
Scalar arguments should have a dollar sign, like `\$α` or `A[i,\$γ]`.

    @tullio G[i,j] := M[i+x+1, j+y+1] * K[x,y]
    @tullio H[i,j] := M[2i+x, 2j+y]  (x in -1:1, y in -1:1)

Shifts and scaling of indices are allowed, including shifts by other indices.
Ranges can be provided as shown, for under-constrained indices.
If they are over-constrained, shifted indices run over the intersection allowed by all constraints,
while un-shifted indices demand agreement between them (e.g. `axes(A,2) == axes(B,1)` above).

    @tullio (*) L[i] := A[J[k]+2, i] / B[k]^2

This is a product instead of a sum, which could also enabled by writing `L[i] *= ...` (in-place).
You can use any reduction function such as `@tullio (max) M[i,j] := ...`.
Indexing by `J[k]+2` here demands `issubset(J, axes(A,1) .- 2)`.

    @tullio N[j] := sqrt <| M[i,j]^2

Pipe operators `|>` and `<|` apply a function after the sum, here `N ≈ map(norm, eachcol(M))`.
Underscores create functions, e.g. `|> sqrt(_ / V[i])` where clearly `i` must not have been summed.

See the readme for further options.
"""
macro tullio(exs...)
    _tullio(exs...; mod=__module__)
end

function _tullio(exs...; mod=Main)

    opts, ranges, ex = parse_options(exs...)
    if isnothing(ex) # then we simply updated global settings
        return (verbose=_VERBOSE[], fastmath=_FASTMATH[], threads=_THREADS[], grad=_GRAD[], avx=_AVX[], cuda=_CUDA[])
    end

    store = DotDict(; mod = mod, opts...,
    # Reduction
        redind = Symbol[],
        init = nothing,
    # Everything writes into leftarray[leftraw...], sometimes with a generated name
        leftraw = [],
        leftind = Symbol[],    # vcat(leftind, redind) is the complete list of loop indices
        leftarray = nothing,
        leftscalar = nothing, # only defined for scalar reduction
        leftnames = Symbol[],  # for NamedDims
        zero = false,
        scalar = false,
        newarray = false,
        plusequals = false,
    # Whole RHS, without finaliser, plus things extracted:
        right = nothing,
        finaliser = nothing,
        rightind = Symbol[],
        sharedind = Symbol[], # indices appearing on every RHS array, safe for ∇thread
        unsafeleft = Symbol[], # k in A[J[k]] never written to by different threads
        unsaferight = Symbol[], # same for gradient
        arrays = Symbol[],
        scalars = Symbol[],
        cost = 1,
    # Index ranges: first save all known constraints
        constraints = Dict{Symbol,Vector}(), # :k => [:(axes(A,2)), :(axes(B,1))] etc.
        constraintpairs = Tuple[], # (:i, :j, entangled range_i, range_j) from A[i+j] etc.
        notfree = Symbol[], # indices assigned values i = clamp(j, 1,3) within RHS
        shiftedind = Symbol[],
        axisdefs = Expr[],
        padmodclamp = false,
    # Expressions:
        outpre = Expr[],  # preliminary steps
        outex = Expr[],   # the rest!
    )

    parse_input(ex, store)

    parse_ranges(ranges, store)

    index_ranges(store)

    output_array(store)

    ex = action_functions(store)

    opts.verbose > 1 && verboseprint(store)

    ex |> esc
end

#========== options, etc ==========#

OPTS = Dict(
    :verbose => Any[true, false, 2, 3],
    :fastmath => [true, false],
    :threads => Integer,
    :grad => [false, :Base, :Dual],
    :avx => Integer,
    :cuda => Integer,
    :tensor => [true, false],
    )

_VERBOSE = Ref{Any}(false)
_FASTMATH = Ref(true)
_THREADS = Ref{Any}(true)
_GRAD = Ref{Any}(:Base)
_AVX = Ref{Any}(true)
_CUDA = Ref{Any}(true)

function parse_options(exs...)
    opts = Dict{Symbol,Any}(
        :redfun => :+,
        :init => TYP, # this means "auto"
        :pad => TYP,
        :verbose => _VERBOSE[],
        :fastmath => _FASTMATH[],
        :threads => _THREADS[],
        :grad => _GRAD[],
        :avx => _AVX[],
        :cuda => _CUDA[],
        :tensor => false,
        )
    expr = nothing
    nograd = Symbol[]
    ranges = Tuple[]
    for ex in exs
        # Actual options:
        if isexpr(ex, :(=)) && haskey(OPTS, ex.args[1])
            checklegal(ex.args[1], ex.args[2])
            opts[ex.args[1]] = ex.args[2]

        # Init & pad keyword
        elseif isexpr(ex, :(=)) && ex.args[1] == :init
            opts[:init] = ex.args[2]
        elseif isexpr(ex, :(=)) && ex.args[1] == :pad
            opts[:pad] = ex.args[2]

        # Nograd keyword
        elseif isexpr(ex, :(=)) && ex.args[1] == :nograd
            if ex.args[2] isa Symbol
                push!(nograd, ex.args[2])
            elseif isexpr(ex.args[2], :tuple)
                append!(nograd, ex.args[2].args)
            else
                throw("this accepts nograd=A or nograd=(A,B,C)")
            end

        # Ranges specified outside:
        elseif isexpr(ex, :call) && ex.args[1] in [:in, :∈]
            push!(ranges, (ex.args[2], ex.args[3]))
        elseif isexpr(ex, :tuple) && isexpr(ex.args[1], :call) && ex.args[1].args[1] in [:in, :∈]
            for el in ex.args
                isexpr(el, :call) && el.args[1] in [:in, :∈] || throw("expected (i ∈ 1:3) but got $el")
                push!(ranges, (el.args[2], el.args[3]))
            end

        # Reduction function
        elseif ex isa Symbol
            opts[:redfun] = ex

        # The main course!
        elseif ex isa Expr
            isnothing(expr) || throw("too many expressions! recognised keywords are $(vcat(:nograd, keys(opts)...))")
            expr = ex
        else
            throw("not sure what to do with input $ex")
        end
    end
    if isnothing(expr) # if run with no expression, it updates global options
        _VERBOSE[] = opts[:verbose]
        _FASTMATH[] = opts[:fastmath]
        _THREADS[] = opts[:threads]
        _GRAD[] = opts[:grad]
        _AVX[] = opts[:avx]
        _CUDA[] = opts[:cuda]
    end
    opts[:tensor] == false || @warn "option tensor=true is deprecated, try Tullio.@tensor"
    (redfun=opts[:redfun],
        initkeyword=opts[:init], # surely there is a tidier way...
        padkeyword=opts[:pad],
        verbose=opts[:verbose],
        fastmath=opts[:fastmath],
        threads=opts[:threads],
        grad=opts[:grad],
        avx=opts[:avx],
        cuda=opts[:cuda],
        nograd=nograd,
    ), ranges, expr
end

checklegal(opt, val) =
    if OPTS[opt] isa Vector
        val in OPTS[opt] || throw("keyword $opt accepts values [" * join(OPTS[opt], ", ") * "]")
    elseif val isa Expr || val isa Symbol
        # allows threads=64^3 to work
    elseif OPTS[opt] == Integer
        val isa Integer && val >= 0 || throw("keyword $opt accepts false or a positive integer")
    end

#========== symbols ==========#

# These only need not to clash with symbols in the input:
RHS, AXIS = :𝓇𝒽𝓈, :𝒶𝓍
ZED, TYP, ACC, KEEP, FINAL = :ℛ, :𝒯, :𝒜𝒸𝒸, :♻️, :💀
EPS, DEL, EXPR = :𝜀, :𝛥, :ℰ𝓍
MAKE, ACT! = :ℳ𝒶𝓀ℯ, :𝒜𝒸𝓉!

# @gensym RHS MAKE ACT!
# @gensym AXIS ZED TYP ACC KEEP FINAL
# @gensym EPS DEL EXPR

SYMBOLS = [
    RHS, MAKE, ACT!, AXIS, ZED, TYP, ACC, KEEP, EPS, DEL, EXPR,
    Symbol(:∇, MAKE), Symbol(:∇, ACT!), Symbol(DEL, ZED), Symbol(AXIS, :i),
    ] # to test for leaks

#========== input parsing ==========#

function parse_input(expr, store)

    # Equals sign & friends:
    if @capture_(expr, left_ := right_ )
        store.newarray = true
    elseif @capture_(expr, left_ = right_ )
    elseif @capture_(expr, left_ += right_ )
        store.plusequals = true
        store.redfun == :+ || throw("can't use += with reduction $(store.redfun)")
    elseif @capture_(expr, left_ *= right_ )
        store.plusequals = true # slightly abusing the name of the flag!
        if store.redfun == :+ # default, then we change it?
            store.verbose>0 && @info "inferring reduction by *, because of lhs *= rhs"
            store.redfun = :*
        elseif store.redfun == :*
        else
            throw("can't use *= with reduction $(store.redfun)")
        end
    elseif @capture_(expr, left_ ^= right_ )
        store.redfun == :+ && throw("can't use ^= with reduction +, please use +=")
        store.redfun == :* && throw("can't use ^= with reduction *, please use *=")
        store.plusequals = true
    else
        throw("can't understand input, expected A[] := B[] (or with =, or +=, *=, ^=) got $expr")
    end

    # Left hand side:
    if @capture_(left, Z_[leftraw__] )
    elseif @capture_(left, [leftraw__] )
        Base.depwarn("to omit a name for the output, please write `_[i,j] := ...` with an underscore (for Tullio ≥ 0.2.14)", Symbol("@tullio"))
        Z = :_
    elseif left isa Symbol # complete reduction
        store.newarray = true
        store.scalar = true
        store.leftscalar = left # because store.leftarray will be the array
        leftraw = Any[1,] # the gradient still indexes a fake 1D array
        expr.head == :(+=) && push!(store.scalars, left)
        Z = ZED
    else
        throw("can't understand LHS, expected A[i,j,k], got $left")
    end
    leftraw2 = tidyleftraw(leftraw, store)
    store.leftind = filter(i -> i isa Symbol && !is_const(i), leftraw2) # this gives correct outer loop order

    if Z == :_
        store.newarray || throw("can't write into an array whose name isn't given!")
        Z = ZED
    end
    store.leftarray = Z

    store.leftraw = finishleftraw(leftraw2, store)
    if store.newarray && !allunique(store.leftind)
        store.zero = true # making diagonals, etc.
    end
    if !(store.newarray)
        saveconstraints(Z, leftraw, store, false) # this adds to leftind, e.g. A[2i+1] = ..., is that bad??
        if store.plusequals # A[J[k]] += is unsafe, A[J[k]] = is not.
            detectunsafe(left, store.unsafeleft, store)
            store.unsafeleft = setdiff(store.unsafeleft, store.leftraw) # and A[J[k],k] += ... is safe.
        end
    end

    # Right hand side
    detectunsafe(right, store.unsaferight, store)
    right2 = MacroTools_prewalk(rightwalk(store), right)

    if isexpr(right2, :call) && right2.args[1] in (:|>, :<|)
        if right2.args[1] == :|>
            store.finaliser = makefinaliser(right2.args[3], store)
            store.right = MacroTools_postwalk(dollarwalk(store), right2.args[2])
        elseif right.args[1] == :<|
            store.finaliser = makefinaliser(right2.args[2], store)
            store.right = MacroTools_postwalk(dollarwalk(store), right2.args[3])
        end
        if store.scalar
            throw("can't use a finaliser $(right2.args[1]) with scalar output")
        end
    else
        store.right = MacroTools_postwalk(dollarwalk(store), right2)
        store.finaliser = :identity
    end

    unique!(store.scalars)
    unique!(store.arrays)
    unique!(store.leftind)
    store.sharedind = unique!(setdiff(store.sharedind, store.notfree))
    store.rightind = unique!(setdiff(store.rightind, store.notfree))
    store.unsaferight = union(setdiff(store.unsaferight, store.sharedind), store.shiftedind)
    any(==(:_), vcat(store.leftind, store.rightind)) && throw("can't use _ as an index name")

    unique!(store.outpre) # kill mutiple assertions, and evaluate any f(A) only once

    if store.newarray && Z in store.arrays
        throw("can't create a new array $Z when this also appears on the right")
    end
end

rightwalk(store) = ex -> begin
        @nospecialize ex
        # First, this will detect any assignment before it is used:
        if isexpr(ex, :(=))
            if ex.args[1] isa Symbol
                push!(store.notfree, ex.args[1])
            elseif isexpr(ex.args[1], :tuple)
                for i in ex.args[1].args
                    i isa Symbol && push!(store.notfree, i)
                end
            end
        end
        isexpr(ex, :return) && throw("can't use return inside body")

        # Second, alter indexing expr. to pull out functions of arrays:
        @capture_(ex, A_[inds__]) || return ex

        if isnothing(arrayonly(A))
            Anew = Symbol(string("≪", A, "≫"))
            push!(store.outpre, :(local $Anew = $(dollarstrip(A))))
            A = Anew
        end

        # Third, save letter A, and what axes(A) says about indices:
        push!(store.arrays, arrayonly(A))
        inds3 = primeindices(inds)
        saveconstraints(A, inds3, store, true)

        # Finally, re-assemble with new A etc:
        return :($A[$(inds3...)])
    end

arrayonly(A::Symbol) = A   # this is for RHS(i,j,k, A,B,C)
arrayonly(A::Expr) =
    if @capture_(A, B_[inds__])
        return arrayonly(B)
    elseif @capture_(A, B_.field_) && !(B isa Symbol)
        return arrayonly(B)
    end # returns nothing from :(f(A)), signal to pull function out,
        # and now also from :(A.b), but not :(A.b[i])

saveconstraints(A, inds, store, right=true) = begin
    A1 = arrayfirst(A, store)
    is = Symbol[]
    foreach(enumerate(inds)) do (d,ex)
        is_const(ex) && return
        containsany(ex, store.notfree) && return
        axis_i = length(inds)==1 ? :($linearindex($A1)) : :($axes($A1,$d))
        ex_i, axis_i = padmodclamp_ind(ex, axis_i, store) # this may pad the axis, or may make it nothing
        range_i, i = range_expr_walk(axis_i, ex_i)
        range_i = range_fix_end(range_i, axis_i)
        if isnothing(axis_i) # because mod(i) or clamp(i+j). Do save index, don't save range.
            if i isa Symbol
                push!(is, i)
                ex_i isa Symbol || push!(store.shiftedind, i)
            elseif i isa Tuple
                push!(is, filter(!isnothing, collect(i))...)
                push!(store.shiftedind, filter(!isnothing, collect(i))...)
            end
        elseif i isa Symbol
            push!(is, i)
            ex_i isa Symbol || push!(store.shiftedind, i)
            v = get!(store.constraints, i, [])
            push!(v, dollarstrip(range_i))
        elseif i isa Tuple # from things like A[i+j]
            push!(is, filter(!isnothing, collect(i))...) # collect for Julia ⩽ 1.3
            push!(store.shiftedind, filter(!isnothing, collect(i))...)
            push!(store.constraintpairs, (i..., dollarstrip.(range_i)...))
        elseif isnothing(i) # from A[J[k]], but A[J[k]+i] goes via store.constraintpairs, I said.
            str = "extrema of index $ex must fit within $A1"
            # @show range_i axis_i # @tullio C[i,k] := B[J[i]+1,k] verbose=2 grad=false # comes here, wrong check
            push!(store.outpre, :($issubset($range_i, $axis_i) || $throw($str)))
        end
    end
    if right
        append!(store.rightind, is)
        if A1 in store.nograd # then don't care whether it sharesindices
        elseif isassigned(store.sharedind)
            shared = intersect(is, store.sharedind)
            empty!(store.sharedind)
            append!(store.sharedind, shared)
        else
            append!(store.sharedind, is)
        end
    else
        append!(store.leftind, is) # why can's this be the only path for store.leftind??
    end
    n = length(inds)
    if n>1  # one index now means linear indexing
        str = "expected a $n-array $A1" # already arrayfirst(A)
        push!(store.outpre, :( $ndims($A1) == $n || $throw($str) ))
    end
end

arrayfirst(A::Symbol, store) = A  # this is for axes(A,d), axes(first(B),d), etc.
arrayfirst(A::Expr, store) =
    if (@capture_(A, Binds_.field_) && @capture_(Binds, B_[inds__]))
        str = "elements $A must be of uniform size"
        push!(store.outpre, :( $all($ZED -> $axes($ZED.$field) == $axes($first($B).$field), $B) || throw($str) ))
        return :( $first($B).$field )
    elseif @capture_(A, B_[inds__])
        str = "elements $A must be of uniform size"
        push!(store.outpre, :( $all($AXIS -> $axes($AXIS) == $axes($first($B)), $B) || $throw($str) ))
        return :( first($B) )
    elseif @capture_(A, B_.field_)
        return A
    end

containsany(ex, list) = begin
    out = false
    MacroTools_postwalk(ex) do x
        if x in list
            out = true
        end
        x
    end
    out
end

primeindices(inds) = map(inds) do ex
    isexpr(ex, Symbol("'")) &&
        return Symbol(ex.args[1], "′") # normalise i''
    ex
end

# This function is for range inference
padmodclamp_ind(i, ax_i, store) = i, ax_i
padmodclamp_ind(ex::Expr, ax_i, store) =
    if ex.head == :call && ex.args[1] in [:mod, :clamp, :pad] && length(ex.args) == 2
        store.padmodclamp = true
        return ex.args[2], nothing # nothing means that range inference is discarded

    elseif ex.head == :call && ex.args[1] == :pad && length(ex.args) == 3
        store.padmodclamp = true
        _, a, p = ex.args
        return ex.args[2], :($padrange($ax_i, $p, $p)) # padrange() is in shifts.jl
    elseif ex.head == :call && ex.args[1] == :pad && length(ex.args) == 4
        store.padmodclamp = true
        _, a, lo, hi = ex.args
        return ex.args[2], :($padrange($ax_i, $lo, $hi))
    else
        return ex, ax_i
    end

padmodclamp_replace(s, store, inside=false) = s
padmodclamp_replace(ex::Expr, store, inside=false) =
    if ex.head == :(=) && @capture_(ex.args[1], A_[inds__])
        # This tricky case is 𝛥A[pad(i,2)] = 𝛥A[pad(i,2)] + ...
        Aex, fun = padmodclamp_pair(A, inds, store, true)
        right = if fun != identity
            padmodclamp_replace(ex.args[2], store, true)
        else
            padmodclamp_replace(ex.args[2], store, inside)
        end
        return fun(:($Aex = $right))
    elseif @capture_(ex, A_[inds__])
        Aex, fun = padmodclamp_pair(A, inds, store)
        return inside ? Aex : fun(Aex)
    else
        args = map(x -> padmodclamp_replace(x, store, inside), ex.args)
        Expr(ex.head, args...)
    end

padmodclamp_pair(A, inds, store, assign=false) = begin
    nopadif = []
    inds4 = map(enumerate(inds)) do (d,ex)
        isexpr(ex, :call) || return ex
        if ex.args[1] == :mod && length(ex.args) == 2
            i = ex.args[2]
            return :($mod($i, $axes($A,$d)))
        elseif ex.args[1] == :clamp && length(ex.args) == 2
            i = ex.args[2]
            return :($clamp($i, $first($axes($A,$d)), $last($axes($A,$d))))
        elseif ex.args[1] == :pad && length(ex.args) >= 2
            i = ex.args[2]
            if !all(==(0), ex.args[3:end]) || length(ex.args) == 2
                # push!(nopadif, :($i >= first(axes($A,$d))), :($i <= last(axes($A,$d)))) # allows avx
                push!(nopadif, :($i >= first(axes($A,$d))), :($i <= Base.last(axes($A,$d)))) # allows avx... but LV 0.8, Julia 1.4, needs Base?
            end
            return i
        end
        ex
    end
    Aex = :($A[$(inds4...)])
    fun = if isempty(nopadif)
        identity
    else
        cond = first(nopadif)
        for c2 in nopadif[2:end]
            cond = :($cond & $c2)
        end
        if assign # for gradients, this wraps 𝛥A[pad(i,2)] = 𝛥A[pad(i,2)] + ...
            ex -> :($cond && $ex)
        elseif store.padkeyword == TYP # default, pad with zero
            ex -> :($cond ? $ex : zero(eltype($A)))
        else
            ex -> :($cond ? $ex : $convert($eltype($A), $(store.padkeyword)))
        end
    end
    Aex, fun # fun(Aex), but also fun(Aex = ...)
end

dollarwalk(store) = ex -> begin
        if isexpr(ex, :call)
            callcost(ex.args[1], store) # cost model for threading
        elseif isexpr(ex, :$) # interpolation of $c things:
            ex.args[1] isa Symbol || throw("you can only interpolate single symbols, not $ex")
            push!(store.scalars, ex.args[1])
            return ex.args[1]
        end
        ex
    end

dollarstrip(expr) = MacroTools_postwalk(expr) do ex
        isexpr(ex, :$) && return ex.args[1]
        ex
    end

tidyleftraw(leftraw, store) = begin
    step1 = map(leftraw) do i
        if isexpr(i, :kw) && store.newarray # then NamedDims wrapper is put on later
                push!(store.leftnames, i.args[1])
                return i.args[2]
        elseif i === :_ # underscores on left denote trivial dimensions
            return 1
        end
        i
    end
    primeindices(step1) # normalise i' to i′
end

finishleftraw(leftraw, store) = map(enumerate(leftraw)) do (d,i)
    is_const(i) && store.newarray && (i != 1)  &&
        throw("can't fix indices on LHS when making a new array")

    if isexpr(i, :$)
        i.args[1] isa Symbol || throw("you can only interpolate single symbols, not $ex")
        push!(store.scalars, i.args[1])
        return i.args[1]

    elseif isexpr(i, :call) && i.args[1] == :+ &&
            length(i.args)==3 && i.args[3] == :_ # magic un-shift A[i+_, j] := ...
        i = primeindices(i.args)[2]
        i isa Symbol || throw("index ($i + _) is too complicated, sorry")
        push!(store.leftind, i)
        deli = Symbol(DEL, i)
        push!(store.scalars, deli) # calculating this must be done later
        return :($i + $deli)

    elseif @capture_(i, J_[inds__]) # scatter operation, A[i,J[j,k]] := ...
        push!(store.nograd, J)
        rightwalk(store)(i) # array J viewed as part of RHS, and provides a range for j,k
        inds2 = filter(j->j isa Symbol, tidyleftraw(inds, store))
        append!(store.leftind, inds2) # but j,k aren't to be summed

        ex = :($J[$(tidyleftraw(inds, store)...)])
        if store.newarray
            ax_i = Symbol(AXIS, string("≪", ex, "≫")) # fake index name, to which to attach a size?
            push!(store.axisdefs, :(local $ax_i = $extremerange($J)))
            store.zero = true
        end

        return ex # has primes dealt with
    end
    i
end

detectunsafe(expr, list, store) = MacroTools_postwalk(expr) do ex
        @capture_(ex, A_[inds__]) || return ex
        for i in inds
            MacroTools_postwalk(i) do x
                @capture_(x, B_[inner__]) || return x
                # Now we have found an array which indexes another one, mark its indices unsafe
                append!(list, filter(j -> j isa Symbol, inner))
                unique!(list)
                # and don't compute a gradient for the inner array
                B isa Symbol && push!(store.nograd, B)
                x
            end
        end
        ex
    end

makefinaliser(s::Symbol, store) = s
makefinaliser(expr::Expr, store) = begin
    underscore = false
    out = MacroTools_postwalk(expr) do ex
        if ex == :_
            underscore = true
            return RHS
        elseif @capture_(ex, A_[inds__])
            for i in inds
                i isa Symbol || continue
                i in store.leftind || throw("index $i can't be used in finaliser")
            end
        end
        ex
    end
    if underscore
        return dollarstrip(:($RHS -> $out))
    else
        return dollarstrip(ex)
    end
end

function parse_ranges(ranges, store) # now runs after parse_input
    for (i,r) in ranges
        if isexpr(i, Symbol("'")) # catch primes!
            i = Symbol(i.args[1], "′")
        end
        push!(store.rightind, i)
        v = get!(store.constraints, i, [])
        if isexpr(r, :call) && r.args[1] == :(:) && length(r.args) == 3
            # for a literal range, write OneTo(10) or 0:9 directly into constraints
            if r.args[2] == 1 && r.args[3] isa Integer
                push!(v, :(Base.OneTo($(r.args[3]))))
                continue
            elseif r.args[2] isa Integer && r.args[3] isa Integer
                push!(v, r)
                continue
            end
        end
        # for axes(A,2) where A is already available, just save it
        if isexpr(r, :call) && r.args[1] in (:axes, :eachindex) && r.args[2] in store.arrays
            push!(v, r)
            continue
        end
        # for anything else, treat it as a scalar argument
        if r isa Symbol
            push!(store.scalars, r)
            push!(v, r)
        else
            s = Symbol(string("≪", r, "≫"))
            push!(store.outpre, :(local $s = $r))
            str = "expected a range for ($i in $r), got "
            push!(store.outpre, :($s isa AbstractRange || throw($str * string($r))))
            push!(store.scalars, s)
            push!(v, s)
        end
    end
    unique!(store.rightind)
    unique!(store.scalars)
    store.redind = setdiff(store.rightind, store.leftind)
end

#========== index ranges ==========#

function index_ranges(store)

    todo = Set(vcat(store.leftind, store.redind))
    done = Dict{Symbol,Any}()

    for (i,j,r_i,r_j) in store.constraintpairs

        if isnothing(i) # case of A[j + I[k]]
            v = get!(store.constraints, j, [])
            push!(v, r_j)
        elseif isnothing(j)
            v = get!(store.constraints, i, [])
            push!(v, r_i)

        elseif haskey(store.constraints, i) && i in todo
            resolveintersect(i, store, done) # use existing knowledge to fix i's range
            pop!(todo, i)
            v = get!(store.constraints, j, []) # and then allow j's range to depend on that
            push!(v, r_j)
        elseif haskey(store.constraints, j) && j in todo
            resolveintersect(j, store, done)
            pop!(todo, j)
            v = get!(store.constraints, i, [])
            push!(v, r_i)
        end
    end

    for i in todo
        haskey(store.constraints, i) || throw("unable to infer range of index $i")
        if i in store.shiftedind
            resolveintersect(i, store, done)
        else
            resolvestrict(i, store, done)
        end
        deli = Symbol(DEL,i)
        if deli in store.scalars # magic shift on LHS
            axi, axi_del = Symbol(AXIS,i), Symbol(AXIS,i,DEL)
            push!(store.axisdefs, :($deli = 1 - $first($axi)),
                :(local $axi_del = Base.OneTo($length($axi))))
            # You can't compute deli inside Act! as doesn't always see full range of i.
            # But if you make it a scalar argument, then it's an argument of Make, hence
            push!(store.outpre, :(local $deli = 0)) # ... this awful hack.
        end
    end

    append!(store.outex, store.axisdefs)

    if store.verbose > 0
        if !isempty(store.leftind)
            lex = map(i -> Expr(:(=), i, done[i]), store.leftind)
            push!(store.outex, :(@info "left index ranges" $(lex...)))
        end
        if !isempty(store.redind)
            rex = map(i -> Expr(:(=), i, done[i]), store.redind)
            push!(store.outex, :(@info "reduction index ranges" $(rex...)))
        end
    end
end

resolvestrict(i, store, done) = begin
    res = first(store.constraints[i])
    ax_i = Symbol(AXIS, i)
    push!(store.axisdefs, :( local $ax_i = $res ))
    done[i] = res
    for alt in store.constraints[i][2:end] # in which case it shouldn't be a Set
        str = "range of index $i must agree"
        push!(store.axisdefs, :( $alt == $res || $throw($str) ))
    end
end

resolveintersect(i, store, done) = begin
    res = if isempty(store.constraints[i])
        throw("unable to infer range of index $i")
    elseif length(store.constraints[i])==1
        first(store.constraints[i])  # because intersect(1:3) isa Vector, wtf?
    else
        :( intersect($(store.constraints[i]...)) )
    end
    ax_i = Symbol(AXIS, i)
    push!(store.axisdefs, :( local $ax_i = $res ))
    done[i] = res
end


#========== output array + eltype ==========#

function output_array(store)

    # Initialisation needs to be worked out somewhere...
    if store.initkeyword == TYP # then auto
        store.init = store.redfun == :+ ? :(zero($TYP)) :
                    store.redfun == :* ? :(one($TYP)) :
                    store.redfun == :max ? :(typemin($TYP)) :
                    store.redfun == :min ? :(typemax($TYP)) :
                    store.redfun == :& ? :(true) :
                    store.redfun == :| ? :(false) :
                    begin
                        store.verbose>0 && @warn "guessing init=zero(T) for unknown reduction $(store.redfun)"
                        :(zero($TYP))
                    end
    else
        if store.initkeyword isa Number
            store.init = store.initkeyword
        else
            init_sy = Symbol(string("≪", store.initkeyword, "≫"))
            push!(store.outpre, :(local $init_sy = $(store.initkeyword)))
            push!(store.scalars, init_sy)
            store.init = init_sy
        end
    end

    # And some not-compltely-unrelated errors:
    if isempty(store.redind) && !(store.plusequals)
        store.redfun == :+ || throw("nothing to reduce over using $(store.redfun)")
        store.finaliser == :identity || throw("can't apply finaliser without a reduction")
    end
    if isempty(store.redind)
        store.initkeyword == TYP || throw("nothing to reduce over, so won't use init = $(store.initkeyword)")
    elseif store.plusequals && !(store.scalar)
        store.initkeyword == TYP || throw("in-place update will not use init = $(store.initkeyword)")
    end

    if store.newarray # this includes scalar case!

        ex_right = padmodclamp_replace(:($(store.finaliser)($(store.right))), store)
        push!(store.outex, :( local $RHS($(store.arrays...), $(store.rightind...)) = $ex_right))

        # Try inference first, usually fine, and avoids scalar evaluation on GPU
        allfirst = map(i -> :($first($(Symbol(AXIS, i)))), store.rightind)
        T1 = Symbol(TYP,1)
        T2 = Symbol(TYP,2)
        T3 = Symbol(TYP,3)
        warn = store.verbose>0 ? :(@warn "unable to infer eltype from RHS") : nothing
        push!(store.outex, quote
            local $T1 = Core.Compiler.return_type($RHS, $typeof(($(store.arrays...), $(allfirst...))))
            local $T2 = if Base.isconcretetype($T1)
                $T1
            else
                $warn
                $typeof($RHS($(store.arrays...), $(allfirst...)))
            end
        end)

        # Init. usually depends on type, but sometimes widens type
        if store.initkeyword == TYP
            push!(store.outex, :(local $T3 = $T2))
        else
            push!(store.outex, :(local $T3 = Base.promote_type($T2, $typeof($(store.init)))))
        end

        # Oh, also scalar += might widen type...
        if store.scalar && store.plusequals
            push!(store.outex, :(local $TYP = Base.promote_type($T3, $typeof($(store.leftscalar)))))
        else
            push!(store.outex, :(local $TYP = $T3))
        end

        # This now checks for OffsetArrays, and allows A[i,1] := rhs. Pulls out scatterers.
        outaxes = map(store.leftraw) do i
            i isa Integer && i==1 && return :(Base.OneTo(1))
            i isa Symbol && return Symbol(AXIS, i)
            i isa Expr && @capture_(i, J_[inds__]) && return Symbol(AXIS, string("≪", i, "≫"))
            i isa Expr && i.head == :call && length(i.args)==3 && i.args[1] == :+ &&
                startswith(string(i.args[3]), string(DEL)) && return Symbol(AXIS, i.args[2], DEL)
            throw("can't use index $i on LHS for a new array")
        end

        if !isdefined(store.mod, :OffsetArrays)
            outaxes = map(store.leftraw, outaxes) do i, ax
                ax == :(Base.OneTo(1)) && return ax
                i in store.shiftedind || @capture_(i, J_[inds__]) || return ax
                push!(store.outex, :( $first($ax) == 1 || $throw("to allow indices not starting at 1, OffsetArrays must be visible in the caller's module. Otherwise write `A[i+_] := ...` to remove offset")))
                return :(Base.OneTo($ax)) # This doesn't apply to offsets caused by pad(i+j,3), sadly?
            end
        end

        simex = if store.scalar && store.plusequals
            :( convert($TYP, $(store.leftscalar)) ) # here init is needed only if threading
        elseif store.scalar
            :( convert($TYP, $(store.init)) )
        elseif isempty(store.arrays)
            :( similar(1:0, $TYP, tuple($(outaxes...))) )
        else
            :( similar($(store.arrays[1]), $TYP, tuple($(outaxes...),)) )
        end
        if store.scalar
            push!(store.outex, :( local $ZED = $simex ))
        elseif isempty(store.leftnames)
            push!(store.outex, :( local $(store.leftarray) = $simex ))
        else
            nex = :(tuple($(QuoteNode.(store.leftnames)...)))
            push!(store.outex, :( local $(store.leftarray) = NamedDims.NamedDimsArray($simex, $nex) ))
        end

        if store.scalar && store.threads != false && store.initkeyword != TYP
            msg = "init=$(store.init) must be compatible with $(store.redfun), for possibly-threaded scalar reduction"
            push!(store.outex, :($(store.redfun)($(store.init), $(store.init)) == $(store.init) || $throw($msg)))
        end
    end

    if store.zero # allow pad=NaN to control this too
        # push!(store.outex, :( $(store.leftarray) .= false )) # zero($TYP) won't work in-place
        if store.padkeyword == TYP # default
            push!(store.outex, :($(store.leftarray) .= zero(eltype($(store.leftarray)))))
        else
            push!(store.outex, :($(store.leftarray) .= $(store.padkeyword)))
        end
    end

    ex_pre = quote $(store.outpre...) end # before act! gets pushed into store.outpre
    store.verbose==2 && @info ">>>>> Preliminary expressions" verbosetidy(ex_pre)
end

#========== action functions ==========#

function action_functions(store)

    axisleft = map(i -> Symbol(AXIS, i), setdiff(store.leftind, store.unsafeleft))
    axisred = map(i -> Symbol(AXIS, i), union(store.redind, store.unsafeleft))
    axislist = vcat(axisleft, axisred)
    # Order of these is convenient for threader(), which divides axisleft up freely,
    # divides axisred up with re-starts.
    # This is independent of the grouping inner/outer for make_many_actors().

    #===== constructing loops =====#

    zed_arg, zed_one = if store.scalar
        :($ZED::$TYP), ZED
    else
        :($ZED::AbstractArray{$TYP}), :($ZED[$(store.leftraw...)])
    end

    ex_init = if store.plusequals && !isempty(axisleft) # then always keep=true
        :( $ACC = $zed_one )
    elseif store.scalar && !(store.plusequals) # then always keep=false
        :( $ACC = $(store.init) )
    else # for non-numbers, similar() may leave undef, so avoid ifelse here
        :( $ACC = $KEEP===nothing ? $(store.init) : $zed_one )
    end

    ex_iter = :( $ACC = $(store.redfun)($ACC, $(store.right) ) )

    ex_write = if store.scalar # then we return the value instead, ZED is immutable
        :( $ACC )
    elseif store.finaliser == :identity
        :( $ZED[$(store.leftraw...)] = $ACC )
    else # this branch is moved outside @avx by finalsplit(expr), below.
        :( $ZED[$(store.leftraw...)] = $FINAL===nothing ? $ACC : $(store.finaliser)($ACC) )
    end

    ex_nored = if store.plusequals # implies keep=true directly, and final=true since no J indices in threader.
        :( $ZED[$(store.leftraw...)] =  $(store.finaliser)($(store.redfun)($ZED[$(store.leftraw...)] ,$(store.right))) )
    else # using finaliser without reduction, and without +=, is now an error.
        :( $ZED[$(store.leftraw...)] = $(store.right) )
    end

    if isempty(store.redind)
        make_many_actors(ACT!,
            vcat(zed_arg, store.arrays, store.scalars, axislist),
            nothing, store.leftind, nothing, Symbol[], ex_nored, nothing, store)
    else
        make_many_actors(ACT!,
            vcat(zed_arg, store.arrays, store.scalars, axislist),
            nothing, store.leftind, ex_init, store.redind, ex_iter, ex_write, store)
    end

    ∇make = if store.newarray
        # make_many_actors and backward_definitions both push into store.outpre
        backward_definitions(store)
    else
        nothing
    end

    #===== action! =====#

    keep = store.plusequals ? :true : :nothing
    block = store.threads==false ? nothing :
        store.threads==true ? cld(BLOCK[], store.cost) :
        store.threads
    if store.scalar
        ST = :($storage_type($(store.arrays...)))
        push!(store.outex, :(
            $thread_scalar($ACT!, $ST, $ZED,
                tuple($(store.arrays...), $(store.scalars...),),
                tuple($(axisred...),), $(store.redfun), $block, $keep)
        ))
    else
        ST = :($storage_type($(store.leftarray), $(store.arrays...)))
        push!(store.outex, :(
            $threader($ACT!, $ST, $(store.leftarray),
                tuple($(store.arrays...), $(store.scalars...),),
                tuple($(axisleft...),), tuple($(axisred...),), $(store.redfun), $block, $keep);
            $(store.leftarray)
        ))
    end
    store.verbose>0 && block != nothing && @info "threading threshold (from cost = $(store.cost))" block

    if store.newarray
        # then slurp up outex to make a function:
        ex_make = quote
            local @inline function $MAKE($(store.arrays...), $(store.scalars...), )
                $(store.outex...)
            end
        end
        store.verbose==2 && @info ">>>>> Maker function" verbosetidy(ex_make)
        ex = quote
            let $ACT! = $ACT!
                $ex_make
                $Eval($MAKE, $∇make)($(store.arrays...), $(store.scalars...), )
            end
        end

        # wrap pre and out in one let block so that ACT! doesn't escape:
        ex = :(let
            $(store.outpre...)
            $ex
        end)

        # and assign the result if necc:
        if store.leftarray != ZED
            push!(store.outex, :($(store.leftarray) = $ex ))
            return :($(store.leftarray) = $ex )
        elseif store.scalar
             push!(store.outex, :($(store.leftscalar) = $ex))
             return :($(store.leftscalar) = $ex)
        else # case of [i,j] := ... with no name given
            # push!(store.outex, ex)
            return ex
        end

    else
        # in-place, no MAKE function, but still keep ACT! from escaping
        ex_body = quote $(store.outex...) end
        store.verbose==2 && @info "In-place body" verbosetidy(ex_body)
        return :(let
            $(store.outpre...)
            $(store.outex...)
        end)
    end
end


"""
    make_many_actors(f!, args, ex1, [:i,], ex3, [:k,], ex5, ex6, store)

This makes several functions of this form,
decorated as necessary with `@inbouds` or `@avx` etc,
and with appropriate `storage_type` as the first argument.
```
f!(::Type, args..., keep=nothing, final=true) where {T}
    ex1
    ex2 = (for i in axis_i
        ex3
        ex4 = (for k in axis_k
            ex5
        end)
        ex6
    end)
end
```
"""
function make_many_actors(act!, args, ex1, outer::Vector, ex3, inner::Vector, ex5, ex6, store, note="")

    if store.padmodclamp
        ex5 = padmodclamp_replace(ex5, store)
    end

    ex4 = recurseloops(ex5, inner)
    ex2 = recurseloops(:($ex3; $ex4; $ex6), outer)

    ex_act = if store.fastmath && isempty(store.notfree)
        quote
            local @inline function $act!(::Type, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                @inbounds @fastmath ($ex1; $ex2)
            end
        end
    elseif isempty(store.notfree)
        quote
            local @inline function $act!(::Type, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                @inbounds ($ex1; $ex2)
            end
        end
    else
        quote
            local @inline function $act!(::Type, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                ($ex1; $ex2)
            end
        end
    end
    store.verbose==2 && @info "===== Base actor $note" verbosetidy(ex_act)
    push!(store.outpre, ex_act)

    if act! != ACT! && isempty(store.sharedind) && store.threads != false
        store.verbose>0 && @warn "can't parallelise this gradient, no shared indices $note"
    end

    #===== LoopVectorization =====#

    expre, exloop0, expost = if isempty(outer)
        :($ex1; $ex3), ex4, ex6
    else
        ex1, ex2, nothing
    end
    exloop, exloopfinal = finalsplit(exloop0)

    # Disable @avx for scatter, https://github.com/chriselrod/LoopVectorization.jl/issues/145
    safe = if act! == ACT!
        isempty(store.unsafeleft)
    else # working on ∇act!
        isempty(store.unsaferight)
    end

    if safe && store.avx != false && isdefined(store.mod, :LoopVectorization)
        unroll = store.avx == true ? 0 : store.avx # unroll=0 is the default setting
        info1 = store.verbose>0 ? :(@info "running LoopVectorization actor $($note)" maxlog=3 _id=$(hash(store))) : nothing
        check1 = store.verbose>0 ? :(LoopVectorization.check_args($(store.arrays...)) || @error "rejected by LoopVectorization's check_args! $($note)" maxlog=3 _id=$(hash(store))) : nothing
        try
            act! == ACT! || store.redfun == :+ || throw("use of LoopVectorization for min/max gradients is disabled")
            lex = if isnothing(exloopfinal)
                quote

                    local @inline function $act!(::Type{<:Array{<:LoopVectorization.NativeTypes}}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                        $expre
                        $info1
                        $check1
                        LoopVectorization.@avx unroll=$unroll $exloop
                        $expost
                    end

                end
            else # "isnothing(final) ? exp(rhs) : rhs" does not prevent execution of finaliser within @avx
                quote

                    local @inline function $act!(::Type{<:Array{<:LoopVectorization.NativeTypes}}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                        $expre
                        $info1
                        $check1
                        if $FINAL === nothing
                            LoopVectorization.@avx unroll=$unroll $exloop
                        else
                            LoopVectorization.@avx unroll=$unroll $exloopfinal
                        end
                        $expost
                    end

                end
            end
            store.verbose==2 && @info "=====LV===== LoopVectorization actor $note" verbosetidy(lex)
            push!(store.outpre, macroexpand(store.mod, lex))
            store.verbose==2 && @info "success expanding LoopVectorization.@avx"
        catch err
            store.verbose>0 && @warn "LoopVectorization failed $note" err
        end
    end

    #===== KernelAbstractions =====#

    unsafe = if act! == ACT!
        store.unsafeleft
    else # working on ∇act!
        store.unsaferight
    end
    safeouter = setdiff(outer, unsafe)

    if store.cuda > 0 && isdefined(store.mod, :KernelAbstractions)
        kernel = gensym(:🇨🇺)
        workgroupsize = store.cuda === true ? nothing : store.cuda  # cuda=true means "use auto-tuning"
        axouter = map(i -> Symbol(AXIS, i), safeouter)
        asserts = map(ax -> :( $first($ax)==1 || $throw("KernelAbstractions can't handle OffsetArrays here")), axouter)
        sizes = map(ax -> :(length($ax)), axouter)

        if isempty(safeouter)
            store.verbose>0 && @warn "using KernelAbstractions with no outer indices, this will be slow"
            safeouter = [Symbol(EPS, 1)] # fake index name, only appears in @index
            sizes = [:(one(Int))]    # iterate over 1:1
        end

        kernelbody = recurseloops(:($ex3; $ex4; $ex6), unsafe)
        try
            # const_args = map(args) do a
            #     a isa Symbol || return a  # this skips output ZED::AbstractArray{TYP}
            #     a == store.leftarray && return a  # case A[i] = A[i]^2 / B[i,j]
            #     :(@Const($a))
            # end
            kex1 = quote
                # @Const removed, see https://github.com/mcabbott/Tullio.jl/pull/32
                # KernelAbstractions.@kernel function $kernel($(const_args...), @Const($KEEP), @Const($FINAL)) where {$TYP}
                KernelAbstractions.@kernel function $kernel($(args...), $KEEP, $FINAL) where {$TYP}
                    ($(safeouter...),) = @index(Global, NTuple)
                    $ex1  # This seems dodgy, shouldn't ex1 be outside?
                    $kernelbody
                end

            end
            store.verbose==2 && @info "=====KA===== KernelAbstractions kernel $note" verbosetidy(kex1)
            push!(store.outpre, macroexpand(store.mod, kex1))
            if isdefined(store.mod, :CUDA) && isdefined(store.mod, :CuArray) # new-style, CUDA.jl, with CUDA.CUDABackend()
                info2 = store.verbose>0 ? :(@info "running KernelAbstractions + CUDA actor $($note)" maxlog=3 _id=$(hash(store))) : nothing
                kex2 = quote

                    local @inline function $act!(::Type{<:CuArray}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                        $info2
                        cu_kern! = $kernel(CUDA.CUDABackend())
                        $(asserts...)
                        $ACC = cu_kern!($(args...), $KEEP, $FINAL; ndrange=tuple($(sizes...)), workgroupsize=$workgroupsize)
                    end

                end
                store.verbose==2 && @info "=====KA===== KernelAbstractions CUDA actor $note" verbosetidy(kex2)
                push!(store.outpre, kex2)
            end
            info3 = store.verbose>0 ? :(@info "running KernelAbstractions CPU actor $($note)" maxlog=3 _id=$(hash(store))) : nothing
            kex3 = quote

                local @inline function $act!(::Type{<:Array}, $(args...), $KEEP=nothing, $FINAL=true) where {$TYP}
                    $info3
                    cpu_kern! = $kernel(CPU(), 4)
                    $(asserts...)
                    $ACC = cpu_kern!($(args...), $KEEP, $FINAL; ndrange=tuple($(sizes...)))
                end

            end
            if store.threads==false
                # This CPU kernel can't be called by threader, and so threads=false
                # offers a way to control whether it gets used or not. By default, not.
                push!(store.outpre, kex3)
            end
            store.verbose==2 && @info "success expanding KernelAbstractions.@kernel"
        catch err
            store.verbose>0 && @warn "KernelAbstractions failed $note" err
        end
    end
end


recurseloops(ex, list::Vector) =
    if isempty(list)
        return ex
    else
        i = first(list)
        r = Symbol(AXIS, i)
        ex = :(for $i in $r; $ex; end)
        return recurseloops(ex, list[2:end])
    end

finalsplit(expr) = begin
    yes = false
    ex_1 = MacroTools_postwalk(expr) do ex
        yes |= isifelsefinal(ex)
        isifelsefinal(ex) ? ex.args[2] : ex
    end
    ex_2 = MacroTools_postwalk(expr) do ex
        isifelsefinal(ex) ? ex.args[3] : ex
    end
    if yes
        return ex_1, ex_2
    else
        return ex_1, nothing
    end
end

# This matches ex = :(isnothing(💀) ? 𝒜𝒸𝒸 : tanh(𝒜𝒸𝒸))
# and ex = :(💀===nothing ? 𝒜𝒸𝒸 : tanh(𝒜𝒸𝒸))
isifelsefinal(ex) = isexpr(ex, :if, 3) && isexpr(ex.args[1], :call) &&
        ex.args[1].args[1] in (:isnothing, :(===)) && ex.args[1].args[2] == FINAL


#===== define gradient hooks =====#

function backward_definitions(store)
    store.grad == false && return nothing # no gradient wanted

    axisshared = map(i -> Symbol(AXIS, i), setdiff(store.sharedind, store.unsaferight)) # safe to multi-thread
    loopind = vcat(store.leftind, store.redind)
    axisnonshared = map(i -> Symbol(AXIS, i), union(setdiff(loopind, store.sharedind), store.unsaferight))

    axislist = vcat(axisshared, axisnonshared) # this defines the order of arguments of ∇act!

    ok = false
    if store.grad == :Dual && store.redfun == :+
        try
            insert_forward_gradient(axislist, store)
            ok = true
            store.verbose==2 && @info "using ForwardDiff gradient"
        catch err
            store.verbose>0 && @warn "ForwardDiff gradient failed" err
        end
    elseif store.grad == :Base
        try
            insert_symbolic_gradient(axislist, store)
            ok = true
            store.verbose==2 && @info "success wtih symbolic gradient"
        catch err
            store.verbose>0 && @warn "symbolic gradient failed" err
        end
    end

    ok == false && return nothing # failed to make a gradient

    dZ = Symbol(DEL, ZED)
    ∇make = Symbol(:∇, MAKE)
    ∇act! = Symbol(:∇, ACT!)

    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)
    defineempties = map(store.arrays, gradarrays) do A, dA
        if A in store.nograd
            :(local $dA = nothing)
        else
            :( local $dA = fill!(similar($A, Base.promote_type(eltype($A), $TYP)), 0) )
        end
    end
    # append!(defineempties, map((x,dx) -> :($dx = zero(Base.promote_type(typeof($x), $TYP))), store.scalars, gradscalars))
    returns = vcat(gradarrays, map(_->:nothing, store.scalars)) # ?? needs a test!
    # returns = vcat(gradarrays, gradscalars)

    ST = :($storage_type($(gradarrays...), $(store.arrays...)))
    block = store.threads==false ? nothing :
        store.threads==true ? cld(BLOCK[], store.cost) :
        store.threads
    input, acton = if store.scalar
        :($dZ::$TYP), :( $OneBox($dZ) ) # a hack to minimise changes to ∇Act!, for now??
    else
        :($dZ::AbstractArray{$TYP}), dZ
    end
    ex_make = quote

        local function $∇make($input, $ZED, $(store.arrays...), $(store.scalars...), ) where {$TYP}
            $(defineempties...)
            $(store.axisdefs...)
            $∇threader($∇act!, $ST,
                tuple($(gradarrays...), $acton, $ZED, $(store.arrays...), $(store.scalars...),),
                tuple($(axisshared...),), tuple($(axisnonshared...), ), $block)
            return ($(returns...),)
        end

    end
    store.verbose==2 && @info "<<<<< Gradient maker function" verbosetidy(ex_make)
    push!(store.outpre, quote
        local $∇make = let $∇act! = $∇act!
            $ex_make
        end
    end)

    return ∇make
end

fillarrayreplace(rhs, dZ) = MacroTools_postwalk(rhs) do @nospecialize ex
        @capture_(ex, A_[inds__]) && A==dZ || return ex
        return Symbol(dZ, :_value)
    end

#========== the end ==========#
````

## File: src/precompile.jl
````
include("precompile/precompile_Tullio.jl")
_precompile_()

module _Precompile_Core
    include("precompile/precompile_Core.jl")
end

module _Precompile_Base
    include("precompile/precompile_Base.jl")
end

if VERSION >= v"1.6-"
    _Precompile_Base._precompile_()
    _Precompile_Core._precompile_()
end

#=
# To generate these files, following:
# https://timholy.github.io/SnoopCompile.jl/stable/snoopi/
# For the full benefit, it seems you must first disable them here.

VERSION # 1.6

using Pkg
Pkg.activate(mktempdir())
Pkg.add("SnoopCompile")
using SnoopCompile

using Tullio

inf_timing = @snoopi begin
    Tullio._tullio(:( A[i] := (1:10)[i] ))
    Tullio._tullio(:( A[i+_] := (1:10)[i+j] ), :(i in 1:3))
    Tullio._tullio(:( A[i, J[k]] := B[i] * C[j,k] ), :(grad=Dual))
end

pc = SnoopCompile.parcel(inf_timing)

SnoopCompile.write(joinpath(pkgdir(Tullio), "src", "precompile"), pc)

=#

#=
# Simple test:

using Tullio
@time Tullio._tullio(:( A[i] := (1:10)[i+j] + (1:3)[j]) );

# Julia 1.5.2:
# 7.116702 seconds (17.27 M allocations: 873.755 MiB, 4.21% gc time)
# 5.373509 seconds (5.91 M allocations: 307.566 MiB, 1.39% gc time)

# Julia 1.6-
# 7.703672 seconds (19.13 M allocations: 1.073 GiB, 4.19% gc time)
# 5.538456 seconds (6.53 M allocations: 380.879 MiB, 1.88% gc time)

=#
````

## File: src/shifts.jl
````
#========== linear indexing ==========#

# Not so related to shifts, but has to live somewhere! (Runtime.)

linearindex(A) = Base.OneTo(length(A))  # Tuple, AbstractArray
linearindex(v::AbstractVector) = Base.axes1(v)

#========== adjusting index ranges, runtime ==========#

# This is to get the range of j in A[2j], from axes(A,1):

function divrange(r::AbstractUnitRange, f::Integer)
    if f > 0
        # a = div(first(r), f, RoundUp) # onnly 1.4 it seems?
        a = cld(first(r), f)
        # z = div(last(r), f, RoundDown)
        z = fld(last(r), f)
    elseif f < 0
        # a = div(last(r), f, RoundUp)
        a = cld(last(r), f)
        # z = div(first(r), f, RoundDown)
        z = fld(first(r), f)
    else
        throw("can't scale indices by zero")
    end
    a:z
end

#=
divrange(1:10, 2) .* 2
divrange(0:10, 2) .* 2
divrange(1:11, 2) .* 2
divrange(1:10, 3) .* 3

divrange(1:10, -1) .* -1 |> sort
divrange(1:10, -2) .* -2 |> sort
divrange(0:10, -2) .* -2 |> sort
divrange(0:11, -2) .* -2 |> sort
=#

# Special case of A[-i]:

function minusrange(r::AbstractRange)
    -last(r):-first(r)
end

#=
minusrange(1:11) == divrange(1:11, -1)
minusrange(1:10) == divrange(1:10, -1)
=#

# This is to get the range of j in A[j+k], given axes(A,1) and the range of k

function subranges(r::AbstractUnitRange, s::AbstractRange)
    first(r)-minimum(s) : last(r)-maximum(s)
end

function addranges(r::AbstractUnitRange, s::AbstractRange)
    first(r)+maximum(s) : last(r)+minimum(s)
end

#=
issubset(subranges(1:10, 1:3) .+ 1, 1:10)
issubset(subranges(1:10, 1:3) .+ 3, 1:10)

issubset(addranges(1:10, 1:3) .- 1, 1:10)
issubset(addranges(1:10, 1:3) .- 3, 1:10)
=#

# This is for A[I[j]] (where this range must be a subset of axes(A,1))
# and for A[I[j]+k] (where it enters into the calculation of k's range).

function extremerange(A)
    α, ω = minimum(A), maximum(A)
    α isa Integer && ω isa Integer || throw("expected integers!")
    α:ω
end

# This gives the range of j implied by A[i, pad(j,3)]

function padrange(r::AbstractUnitRange, lo::Integer, hi::Integer)
    first(r)-lo : last(r)+hi
end

#========== functions used by the macro ==========#

@nospecialize

# This is for the bounds check on A[I[j],k]:

function extremeview(ex::Expr)
    @assert ex.head == :ref
    A = ex.args[1]
    if any(is_const, ex.args[2:end])
        ind = map(i -> is_const(i) ? i : (:), ex.args[2:end])
        :(@view $A[$(ind...)])
    else
        A
    end
end

"""
    range_expr_walk(:(axes(A,1)), :(2i+1)) -> range, :i

Given the axis of `A`, and the expression inside `A[2i+1]`,
this returns an expression for the resulting range of index `i`.
Understands operations `+, -, *, ÷`.
(Don't really need `÷`, as this results in a non-`UnitRange`
which can't be a valid index.)

If the expression is from something like `A[2i+j]`, then it returns a tuple of ranges
and a tuple of symbols. The range for `:j` contains `:$(AXIS)i` and v-v.

If the expression is from `A[I[j]]` then it returns `(min:max, nothing)`,
and the caller should check `issubset(min:max, axes(A,1))`.
"""
function range_expr_walk(r, ex::Expr, con=[])
    ex.head == :kw && return range_expr_kw(r, ex)
    if ex.head == :ref # case of M[I[j], k] with r=axes(M,1)
        A = ex.args[1]
        A = extremeview(ex)
        push!(con, :(minimum($A) in $r && maximum($A) in $r || throw("not safe!"))) # not used??
        return (:($extremerange($A)),nothing)
    end
    ex.head == :call || throw("not sure what to do with $ex")
    if ex.args[1] == :pad && length(ex.args) == 3
        _, a, p = ex.args
        return range_expr_walk(:($padrange($r, $p, $p)), a)
    elseif ex.args[1] == :pad && length(ex.args) == 4
        _, a, lo, hi = ex.args
        return range_expr_walk(:($padrange($r, $lo, $hi)), a)
    elseif length(ex.args) == 2
        op, a = ex.args
        if op == :+
            return range_expr_walk(r, a)
        elseif op == :-
            return range_expr_walk(:($minusrange($r)), a)
        end
    elseif length(ex.args) == 3
        op, a, b = ex.args
        if op == :+
            is_const(a) && return range_expr_walk(:($r .- $a), b)
            is_const(b) && return range_expr_walk(:($r .- $b), a)
            # with neither constant, first go outwards from index j to expression b...
            ax_a = range_unwrap(a)
            ax_b = range_unwrap(b)
            #... then use that with given size(A,d) to constrain range of i, and v-v:
            range_a, i_a = range_expr_walk(:($subranges($r, $ax_b)), a)
            range_b, i_b = range_expr_walk(:($subranges($r, $ax_a)), b)
            return (range_a, range_b), (i_a, i_b)

        elseif op == :-
            is_const(a) && return range_expr_walk(:($minusrange($r .- $a)), b)
            is_const(b) && return range_expr_walk(:($r .+ $b), a)
            ax_a = range_unwrap(a)
            ax_b = range_unwrap(b)
            range_a, i_a = range_expr_walk(:($addranges($r, $ax_b)), a)
            range_b, i_b = range_expr_walk(:($minusrange($subranges($r, $ax_a))), b)
            return (range_a, range_b), (i_a, i_b)

        elseif op == :*
            is_const(a) && return range_expr_walk(:($divrange($r, $a)), b)
            is_const(b) && return range_expr_walk(:($divrange($r, $b)), a)
        elseif op in (:÷, :/)
            throw("division using ÷ or / in indexing is not supported")
        end
    elseif length(ex.args) > 3
        op, a, b, c = ex.args[1:4]
        ds = ex.args[5:end]
        if op == :+
            is_const(a) && return range_expr_walk(:($r .- $a), :(+($b, $c, $(ds...))))
            is_const(b) && return range_expr_walk(:($r .- $b), :(+($a, $c, $(ds...))))
            is_const(c) && return range_expr_walk(:($r .- $c), :(+($a, $b, $(ds...))))
        end
    end
    throw("not sure what to do with $ex, sorry")
end

range_expr_walk(range, s::Symbol) = range, s
range_expr_walk(range, n::Integer) = range, nothing

is_const(::Int) = true
is_const(::Any) = false
is_const(s::Symbol) = s in [:(:), :begin, :end] # : for the purposes of saveconstraints setting :intersect
is_const(ex::Expr) = begin
    ex.head == :$ && return true # what's returned by range_expr_walk will still contain $
    if ex.head == :call && ex.args[1] in (:+, :-, :*, :÷)
        return all(is_const, ex.args[2:end])
    end
    false
end

"""
    range_expr_walk(:(axes(A,1)), :(i=j)) -> :(axes(A, :i)), :j

Special case for keyword indexing, `A[i=j, k=j+2]` comes here.
"""
function range_expr_kw(r::Expr, ex::Expr)
    @assert ex.head == :kw
    # @assert r.head == :call && r.args[1] == :axes
    r.args[3] = QuoteNode(ex.args[1])
    range_expr_walk(r, ex.args[2])
end

"""
    range_unwrap(:(2i+1)) -> :(2 .* AXIS_i .+ 1)

This goes in the opposite direction to `range_expr_walk`, and gives
the range of values taken by the expression, in terms of `Symbol($AXIS, i)`.
"""
range_unwrap(i::Symbol) = Symbol(AXIS, i)
range_unwrap(ex::Expr) = begin
    if ex.head == :ref # case of A[I[j]+k] comes here
        A = ex.args[1]
        return :($extremerange($A))
    end
    ex.head == :call || throw("don't know how to handle $ex")
    # if ex.args[1] == :pad or :clamp or :mod --- find test cases.
    if length(ex.args) == 2
        op, a = ex.args
        if op == :-
            return :($minusrange($(range_unwrap(a))))
        end
    elseif length(ex.args) == 3
        op, a, b = ex.args
        if op == :*
            a == -1 && return :($minusrange($(range_unwrap(b))))
            b == -1 && return :($minusrange($(range_unwrap(a))))
            is_const(a) && return :($a .* $(range_unwrap(b)))
            is_const(b) && return :($b .* $(range_unwrap(a)))
        elseif op == :+
            is_const(a) && return :($a .+ $(range_unwrap(b)))
            is_const(b) && return :($b .+ $(range_unwrap(a)))
        elseif op == :-
            is_const(a) && return :($a .- $(range_unwrap(b)))
            is_const(b) && return :($(range_unwrap(a)) .- $b)
        end
    end
    throw("don't know how to handle $ex, sorry")
end

"""
    range_expr_walk(nothing, :(i+2)) -> nothing, :j

Special case used for `A[mod(i+2)]` etc.
"""
range_expr_walk(r::Nothing, s::Symbol) = r, s
range_expr_walk(r::Nothing, n::Integer) = r, nothing
function range_expr_walk(r::Nothing, ex::Expr)
    is_const(ex) && return r, nothing
    ex.head == :ref && return r, nothing
    ex.head == :call || throw("not sure what to do with $ex")
    # if ex.args[1] in [:+, :-, :*, :÷]
        syms = []
        for x in ex.args[2:end]
            _, i = range_expr_walk(r, x)
            i isa Tuple && push!(syms, i...)
            i isa Symbol && push!(syms, i)
        end
        if isempty(syms)
            return r, nothing
        elseif length(syms) == 1
            return r, syms[1]
        else
            return r, Tuple(syms)
        end
    # end
end

"""
    range_fix_end( :(minusrange(axes(A, 1), end), :(axes(A, 1)) )

While `range_expr_walk` knows that `:end` is a constant, it doesn't remove it
from expressions to calculate ranges, since it has by then forgotten the original range.
"""
function range_fix_end(expr, axis_i)
    MacroTools_prewalk(expr) do ex
        ex === :end && return _fix_end(axis_i)
        ex === :begin && return _fix_begin(axis_i)
        return ex
    end
end

_fix_end(ex) =
    if isexpr(ex, :call) && ex.args[1] in (:axes, axes, :(Base.axes))
        _, A, d = ex.args
        :($lastindex($A, $d))
    else
        :($last($ex)) 
    end
_fix_begin(ex) =
    if isexpr(ex, :call) && ex.args[1] in (:axes, axes, :(Base.axes))
        _, A, d = ex.args
        :($firstindex($A, $d))
    else
        :($first($ex)) 
    end

@specialize

#========== the end ==========#
````

## File: src/symbolic.jl
````
#========== backward gradient using symbolic derivatives ==========#

using DiffRules

function insert_symbolic_gradient(axislist, store)

    dZ = Symbol(DEL, ZED)
    ∇act! = Symbol(:∇, ACT!)
    maxflag = Symbol(RHS, :⛰)
    maxdone = Symbol(:🆗, RHS)
    gradarrays = map(A -> Symbol(DEL, A), store.arrays)
    # gradscalars = map(A -> Symbol(DEL, A), store.scalars)

    out_ind, in_ind = if store.redfun == :+
        store.sharedind, setdiff(vcat(store.leftind, store.redind), store.sharedind)
    elseif store.redfun in [:min, :max] # :*,
        store.leftind, store.redind
    else
        throw("can't take gradients with reduction $(store.redfun)")
    end

    targets = []
    MacroTools_postwalk(symbwalk(targets, store), store.right)
    # append!(targets, scalars)

    if isempty(targets) # short-circuit
        push!(store.outpre, :(local @inline $∇act!(::Type, args...) = nothing ))
        store.verbose > 0 && @info "no gradient to calculate"
        return nothing
    end

    inbody, prebody = [], []
    for (dt, t) in unique(targets)
        drdt = leibnitz(store.right, t)
        deltar = if store.finaliser == :identity
            simplitimes(simpliconj(drdt), :($dZ[$(store.leftraw...)]))
        else
            rhs = :($ZED[$(store.leftraw...)])
            dldr = leibfinal(store.finaliser, rhs)
            simplitimes(simpliconj(drdt), simpliconj(dldr), :($dZ[$(store.leftraw...)]))
        end
        if store.redfun == :+
            push!(inbody, :($dt = $dt + $deltar))
        # elseif store.redfun == :*
        #     push!(inbody, :($dt = $deltar * $ZED[$(store.leftraw...)] * inv($(store.right))))
        #     push!(prebody, :($dt = $deltar * $ACC))
        elseif store.redfun in [:min, :max]
            push!(inbody, :($dt += $ifelse($maxflag, $deltar, $zero($TYP))))
        end
    end
    store.verbose>0 && @info "symbolic gradients" inbody
    ex_body = commonsubex(quote $(inbody...) end)

    ex_pre, ex_post = if store.redfun == :* # then nonzero LHS are handled already, but harder cases here:
        product_grad(prebody, store)
    elseif store.redfun in [:min, :max]
        :($maxdone = 0), nothing
    else
        nothing, nothing
    end
    if store.redfun in [:min, :max] # this case really wants sparse 𝛥x!
        ex_body = :(
            $maxflag = Tullio.onlyone($ZED[$(store.leftraw...)] == $(store.right), $maxdone);
            $ex_body;
            $maxdone += Tullio.anyone($maxflag);
            )
    end

    make_many_actors(∇act!,
        vcat(gradarrays, :($dZ::AbstractArray{$TYP}), ZED, store.arrays, store.scalars, axislist),
        # vcat(gradarrays, gradscalars, :($dZ::AbstractArray{$TYP}), store.arrays, store.scalars, axislist),
        nothing, out_ind, ex_pre, in_ind, ex_body, ex_post, store, "(symbolic gradient)")

    if isdefined(store.mod, :Zygote) && !(store.scalar) # special case for FillArrays
        ex_body2 = fillarrayreplace(ex_body, dZ)
        ex_pre2 = fillarrayreplace(ex_pre, dZ)
        ex_value = :($(Symbol(dZ, :_value)) = $dZ.value) # @avx likes this outside the loop

        make_many_actors(∇act!,
            vcat(gradarrays, :($dZ::Zygote.Fill{$TYP}), ZED, store.arrays, store.scalars, axislist),
            ex_value, out_ind, ex_pre2, in_ind, ex_body2, ex_post, store, "(gradient method for FillArrays)")
    end

end

leibfinal(fun::Symbol, res) =
    if fun == :log
        :(exp(-$res)) # this exp gets done at every element :(
        # :(inv(exp($res)))
    else
        _leibfinal(:($fun($RHS)), res)
    end

_leibfinal(out, res) = begin
    grad1 = leibnitz(out, RHS)
    grad2 = MacroTools_postwalk(grad1) do ex
        # @show ex ex == out
        ex == out ? res : ex
    end
    MacroTools_postwalk(grad2) do ex
        ex == RHS ? throw("couldn't eliminate partial sum") : ex
    end
end

leibfinal(ex::Expr, res) = begin
    if ex.head == :call && ex.args[1] isa Expr &&
        ex.args[1].head == :(->) && ex.args[1].args[1] == RHS # then it came from underscores
        inner = ex.args[1].args[2]
        if inner isa Expr && inner.head == :block
            lines = filter(a -> !(a isa LineNumberNode), inner.args)
            length(lines) == 1 && return _leinfinal(first(lines), res)
        end
    end
    throw("couldn't understand finaliser")
end

#=
Tullio.leibfinal(:exp, :res)   # :res
Tullio.leibfinal(:sqrt, :res)  # :(inv(res) / 2)
Tullio.leibfinal(:tanh, :res)  # :(1 - res ^ 2)
Tullio.leibfinal(:log, :res)   # :(exp(-res))
=#

# This works for simple cases, but the general case is more complicatd.
#=
product_grad(prebody, store) = begin
    cnt = Symbol(DEL,:𝒸ℴ𝓊𝓃𝓉,0)

    inds_orig = :(($(store.redind...),))
    inds_prime = :(($(map(i -> Symbol(i,'′',DEL), store.redind)...),))
    inds_zero = :(($(map(i -> 0, store.redind)...),))

    loop_search = recurseloops(:(
        # find and save the index at which RHS is zero
        if iszero($(store.right))
            $cnt += 1
            $inds_prime = $inds_orig
        end
    ), copy(store.redind))

    loop_accum = recurseloops(:(
        # product of RHS at all redind except the one which gives zero
        $ACC = $ACC * ifelse($inds_orig == $inds_prime, 1, $(store.right))
    ), copy(store.redind))

    store.verbose>0 && @info "symbolic gradients extra..." prebody
    ex_prebody = commonsubex(quote $(prebody...) end)

    ex_pre = quote
        if iszero($ZED[$(store.leftraw...)])
            local $cnt = 0
            local $inds_prime = $inds_zero
            $loop_search
            if $cnt == 1
                local $ACC = one($TYP)
                $loop_accum
                let $inds_orig = $inds_prime
                    $ex_prebody
                end
            end # elseif more than one zero, then leave 𝛥x .== 0
            # continue # i.e. skip the ordinary routine, which divides
            @goto JUMP
        end
    end

    ex_post = quote
        @label JUMP
    end

    push!(store.notfree, cnt) # hack to disable @inbounds, avoids ERROR: syntax: misplaced label

    ex_pre, ex_post
end
=#

#========== symbolic differentiation ==========#

# This could probably use https://github.com/dfdx/XGrad.jl
# or https://github.com/SciML/ModelingToolkit.jl
# or https://github.com/JuliaMath/Calculus.jl/blob/master/src/differentiate.jl
# or now I found this: https://github.com/HarrisonGrodin/Simplify.jl
# but seemed simple enough to just write out, using rules from:
# http://www.juliadiff.org/DiffRules.jl/latest/

symbwalk(targets, store) = ex -> begin
        @capture_(ex, A_[inds__]) && A isa Symbol || return ex
        A in store.nograd && return ex
        deltaex = :($(Symbol(DEL, A))[$(inds...)])
        push!(targets, (deltaex, ex))
        return ex
    end

leibnitz(s::Number, target) = 0
leibnitz(s::Symbol, target) = s == target ? 1 : 0
leibnitz(ex::Expr, target) = begin
    ex == target && return 1
    @capture_(ex, B_[ijk__]) && return 0
    if ex.head == Symbol("'")
        ex.head = :call
        pushfirst!(ex.args, :adjoint)
    end
    ex.head == :call || throw("expected a functionn call, got $ex.")
    fun = ex.args[1]
    if fun == :log # catch log(a*b) and especially log(a/b)
        arg = ex.args[2]
        if arg isa Expr && arg.args[1] == :* && length(arg.args) == 3
            newex = :(log($(arg.args[2])) + log($(arg.args[3])))
            return leibnitz(newex, target)
        elseif arg isa Expr && arg.args[1] == :/
            newex = :(log($(arg.args[2])) - log($(arg.args[3])))
            return leibnitz(newex, target)
        end
    end
    if length(ex.args) == 2 # one-arg function
        fx = mydiffrule(fun, ex.args[2])
        dx = leibnitz(ex.args[2], target)
        return simplitimes(fx, dx)
    elseif length(ex.args) == 3  # two-arg function
        fx, fy = mydiffrule(fun, ex.args[2:end]...)
        dx = leibnitz(ex.args[2], target)
        dy = leibnitz(ex.args[3], target)
        return simpliplus(simplitimes(fx, dx), simplitimes(fy, dy))
    elseif fun in [:+, :*]
        fun == :* && return leibnitz(:(*($(ex.args[2]), *($(ex.args[3:end]...)))), target)
        dxs = [leibnitz(x, target) for x in ex.args[2:end]]
        fun == :+ && return simpliplus(dxs...)
    elseif length(ex.args) == 4  # three-arg function such as ifelse
        fx, fy, fz = mydiffrule(fun, ex.args[2:end]...)
        dx = leibnitz(ex.args[2], target)
        dy = leibnitz(ex.args[3], target)
        dz = leibnitz(ex.args[4], target)
        return simpliplus(simplitimes(fx, dx), simplitimes(fy, dy), simplitimes(fz, dz))
    end
    throw("don't know how to handle $ex.")
end

simplitimes(x::Number, y::Number) = x*y
simplitimes(x::Number, y) = x==0 ? 0 : x==1 ? y : x==-1 ? :(-$y) : :($x * $y)
simplitimes(x, y::Number) = y==0 ? 0 : y==1 ? x : y==-1 ? :(-$x) : :($y * $x)
simplitimes(x, y) = :($y * $x)
simplitimes(x, y, zs...) = simplitimes(simplitimes(x, y), zs...)

simpliplus(x::Number, y::Number) = x + y
simpliplus(x::Number, y) = x==0 ? y : :($x + $y)
simpliplus(x, y::Number) = y==0 ? x : :($x + $y)
simpliplus(x, y) = :($x + $y)
simpliplus(x, y, zs...) = simpliplus(simpliplus(x, y), zs...)

simpliconj(x::Number) = conj(x)
simpliconj(x) = :(conj($x))

mydiffrule(f, xs...) = begin
    f == :+ && return map(_->1, xs)
    f == :- && return length(xs)==1 ? -1 : (1,-1)
    f == :^ && return mypowrule(xs...)
    f == :/ && return mydivrule(xs...)
    f == :// && return mydivrule(xs...)
    f == :inv && return mydivrule(1, xs...)[2]
    f == :log && return simpliinv(xs...)
    f == :abs && return myabsrule(xs...)
    f == :sqrt && return mysqrtrule(xs...)
    f == :relu && return myrelurule(xs...)
    f in BASE_NOGRAD && return map(_->0, xs)
    DiffRules.hasdiffrule(:Base, f, length(xs)) &&
        return DiffRules.diffrule(:Base, f, xs...)
    DiffRules.hasdiffrule(:SpecialFunctions, f, length(xs)) &&
        return DiffRules.diffrule(:SpecialFunctions, f, xs...)
    throw("no diffrule found for function $f($(join(map(_->"_",xs),", "))).")
end

BASE_NOGRAD = [:(==), :(!=), :(<), :(<=), :(>), :(>=), :trunc, :round]

# Goals of these rules, besides correctness, are:
# 1. don't cause promotion of Float32, e.g. by factors (1/2)
# 2. make it easy for commonsubex(), e.g. by re-using inv(x)

mydivrule(x, y) = begin # (:(one(x) / y), :(-((x / y) / y)))
    invy = simpliinv(y)
    invy, :( -($x) * $invy * $invy )
end
mydivrule(x, y::Integer) = (y==1 ? 1 : 1//y), 0
mydivrule(x, y::Number) = (y==1 ? 1 : :(one($TYP)/$y)), 0

mydivrule(x::Number, y) = 0, :((-$x)*inv($y)*inv($y))
mydivrule(x::Number, y::Number) = 0, 0
mydivrule(x::Number, y::Integer) = 0, 0

mysqrtrule(x::Number) = sqrt(x)
mysqrtrule(x) = :(inv(sqrt($x))/2)

simpliinv(x) = :(inv($x))
simpliinv(x::Integer) = :(1//$x)
simpliinv(x::Expr) = if x.head == :call && x.args[1] == :/
        :($(x.args[3]) / $(x.args[2]))
    else
        :(inv($x))
    end

mypowrule(x, p) = begin
    dx = simplitimes(p, simplipow(x, simpliplus(p, -1)))
    dp = simplitimes(simplipow(x,p), :(log($x)))
    dx, dp
end

simplipow(x::Number, p::Number) = x^p
simplipow(x, p::Number) = p==1 ? x : p==2 ? :($x*$x) : :($x^$p)
simplipow(x, p) = :($x^$p)

myrelurule(x::Number) = x>0 ? 1 : 0
myrelurule(x) = :(ifelse($x>0, 1, 0))

myabsrule(x::Number) = x<0 ? -1 : 1
myabsrule(x) = :(ifelse($x<0, -1, 1)) # matches DiffRules._abs_deriv, which uses signbit(x)

#========== CSE ==========#

# My approach was to look for things occuring twice, biggest first.
# Then I found https://github.com/rdeits/CommonSubexpressions.jl
# which just pulls everything out, but doesn't like indexing expressions.

function commonsubex(expr::Expr)
    dict, defs, nope = Dict(), [], Set()
    if expr.head == :block
        args = [csewalk(ex, dict, defs, nope) for ex in copy(expr).args]
        quote
            $(defs...)
            $(args...)
        end
    else
        res = csewalk(copy(expr), dict, defs, nope)
        quote
            $(defs...)
            $res
        end
    end
end

csewalk(ex, dict, defs, nope) = ex
csewalk(ex::Expr, dict::Dict, defs::Vector, nope::Set) =
    # The goal is to alter RHS of assignments,
    # this mess is the most common case, A = A + stuff
    if ex.head == :(=) && ex.args[2] isa Expr && ex.args[2].head == :call &&
        ex.args[2].args[1] == :+ && ex.args[2].args[2] == ex.args[1]
        for n in 3:length(ex.args[2].args)
            ex.args[2].args[n] = csewalk(ex.args[2].args[n], dict, defs, nope)
        end
        push!(nope, ex.args[1]) # new Ex3 = ... cannot have this on RHS
        ex
    elseif ex.head in (:(=), :(+=)) # easier case of A = stuff
        push!(nope, ex.args[1])
        ex.args[2] = csewalk(ex.args[2], dict, defs, nope)
        ex

    # Then we work on sub-expressions, replace those we're seen immediately,
    # and don't look inside A[i,j] at all:
    elseif haskey(dict, ex)
        dict[ex]
    elseif ex.head == :ref
        ex

    # Simplest case is the last one, replace a whole expression with Ex5 & work inwards.
    # Can't replace "illegal" expressions, but can look for parts which are safe:
    elseif illegal(ex, nope)
        args = Any[x in nope ? x : csewalk(x, dict, defs, nope) for x in ex.args]
        Expr(ex.head, args...)

    elseif ex.head == :call && ex.args[1] in (:*, :+) && length(ex.args) >= 4 # e.g. 1*2*3
        inner = []
        while length(ex.args) >= 3
            pushfirst!(inner, pop!(ex.args))
        end
        binary = Expr(:call, ex.args..., Expr(:call, ex.args[1], inner...))
        csewalk(binary, dict, defs, nope)

    else
        args = Any[csewalk(x, dict, defs, nope) for x in ex.args]
        sy = Symbol(EXPR, length(defs)+1)
        dict[ex] = sy
        # add defn for the outermost operation:
        push!(defs, Expr(:(=), sy, Expr(ex.head, args...)))
        # and return the name for caller:
        sy
    end

illegal(ex, nope) = ex in nope
illegal(ex::Expr, nope) = ex in nope || any(illegal(x, nope) for x in ex.args)

#========== examination ==========#

"""
    Tullio.@printgrad log(x/y) x y

Prints the symbolic gradient, showing `∂f/∂x` and `∂f/∂y` for `f=log(x/y)`.
Useful to check that simplifications, and common subexpression elimination,
are working OK for a given RHS.
"""
macro printgrad(exs...)
    printgrad(exs...)
end

function printgrad(ex::Expr, ts::Symbol...)
    out = quote end
    for t in ts
        df = leibnitz(ex, t)
        dt = Symbol(:δ, t) # Symbol("∂f_∂", t)
        push!(out.args, :($dt = $df))
    end
    print("Initial:\n   ")
    println(join(filter(x -> !(x isa LineNumberNode), out.args), "\n   "))
    print("After CSE:\n   ")
    done = filter(x -> !(x isa LineNumberNode), commonsubex(out).args)
    println(join(done, "\n   "))
    nothing
end

#=

using Tullio: @printgrad

@printgrad  x * y * z   x y z
@printgrad  x * (y * z)   x y z
@printgrad  x + y * z   x y z

@printgrad  1/x   x
@printgrad  x^-1   x   # could make inv(x) for CSE
@printgrad  inv(x)   x
@printgrad  sqrt(x)   x
@printgrad  1/sqrt(x)   x
@printgrad  inv(sqrt(x))   x
@printgrad  x/sqrt(y)   x y

@printgrad  sqrt(x*y)   x y
@printgrad  sqrt(x) * sqrt(y)   x y # worse than line above

@printgrad  1/sqrt(x*y)   x y       # could use repeated CSE

@printgrad  x/sqrt(y*z)   x y z     # could use repeated CSE
@printgrad  x/(sqrt(y)*sqrt(z))   x y z
@printgrad  x*inv(sqrt(y))*inv(sqrt(z))   x y z

@printgrad  x/2   x
@printgrad  x/y   x y

@printgrad  x^2   x
@printgrad  (x*y)^2   x y
@printgrad  (x+y)^3   x y
@printgrad  x^y   x y
@printgrad  log(x)^2   x

@printgrad  log(x)   x
@printgrad  log(x/2)   x
@printgrad  log(2x)   x
@printgrad  log(k*x)   x

@printgrad  x*log(y)   x y

@printgrad  log(x*y)   x y
@printgrad  log(x) + log(y)   x y  # better, now used for log(x*y)

@printgrad  log(x/y)   x y
@printgrad  log(x*inv(y))   x y
@printgrad  log(x)-log(y)   x y    # much better, now used for log(x/y)

@printgrad  log(x/y) * z   x y z
@printgrad  (log(x) - log(y)) * z   x y z
@printgrad  log(x)*z - log(y)* z   x y z

@printgrad  exp(2x)   x
@printgrad  exp(x/y)   x y
@printgrad  exp((x-y)^2/2)   x y

@printgrad  exp(x) * y   x y
@printgrad  exp(x) / 2y   x y

@printgrad a * b / sqrt(d * e)  a b d e
@printgrad x * z / sqrt(y * z)  x y z

=#


#========== the end ==========#
````

## File: src/tensor.jl
````
#========== use TensorOperations when you can ==========#

"""
    Tullio.@tensor C[i,j] := A[i,k] * B[k,j]

This is a way to run `TensorOperations.@tensor`, whose only advantage is 
that it provides gradient definitions. (This code is part of Tullio.jl a bit by accident.)

This is less flexible than `TensorOperations.@tensor`, and in particular it accepts 
only a single term on the RHS. It is much less flexible that `@tullio`, but will 
sometimes be faster.
"""
macro tensor(exs...)
    opts, ranges, ex = parse_options(exs...)
    
    isempty(ranges) || throw("@tensor does not accept explicit index ranges")
    opts.redfun == :+ || throw("@tensor only reduces over +")
    opts.initkeyword == TYP || throw("@tensor does not accept init keyword")

    res = try_tensor(ex, ranges, DotDict(; mod = __module__, opts...,
        newarray = false, scalar = false,
        arrays = Symbol[], indices = [], scalars = Symbol[]))

    Expr(:block, res...) |> esc
end

function try_tensor(expr, ranges, store)

    fail = nothing
    if isexpr(expr, [:(:=), :(=), :(+=)])
    else
        fail = "@tensor expected left := right etc"
    end
    if @capture_(expr.args[1], Z_[leftind__]) && all(a -> a isa Symbol, leftind)
        if expr.head == :(:=)
            store.newarray = true
        end
    elseif expr.args[1] isa Symbol # scalar output
        store.scalar = true # not used?
        Z, leftind = expr.args[1], nothing
        if expr.head == :(:=)
            store.newarray = true
            expr.head = :(=) # mutate it as @tensor doesn't accept scalar :=
        elseif expr.head == :(=)
            store.newarray = true
        end # for scalars, only += case isn't :newarray
    else
        fail = "@tensor expected A[i,j,k] := ..."
    end
    MacroTools_postwalk(expr.args[2]) do ex
        ex isa Expr || return ex
        if ex.head == :call && ex.args[1] == :* && all(a -> a isa Expr || a isa Number, ex.args[2:end])
            # Todo: allow A[i] * $c
        elseif ex.head == :ref && all(a -> a isa Symbol, ex.args)
        elseif ex.head == :call && ex.args[1] in [:+, :-] && length(ex.args)==2
            # Allows -A[i]. Could allow conj() too, but gradient would be wrong.
        else
            # Disallows anything containing +, since A[i] + B[i,k,k] has differing meanings.
            fail = "Tullio.@tensor can't handle $(ex)"
        end
        ex
    end
    fail != nothing && throw(fail)

    outex = [] # you could simplify, only one expression really
    # try
        tex = macroexpand(store.mod, :(TensorOperations.@tensor $expr))

        if store.newarray
            left, right = expr.args
            #===== new array =====#

            MacroTools_postwalk(right) do ex
                ex isa Expr || return ex
                # Save array and scalar arguments
                if @capture_(ex, A_[ijk__])
                    A1 = arrayonly(A)
                    push!(store.arrays, A1)
                    push!(store.indices, ijk)
                    n = length(ijk)
                    str = "expected a $n-array $A1"
                    push!(outex, :( $ndims($A1) == $n || $error($str) ))
                elseif ex.head == :call && ex.args[1] == :*
                    foreach(ex.args[2:end]) do a
                        a isa Symbol && push!(store.scalars, a)
                    end
                end
                ex
            end

            if store.grad == false
                push!(outex, tex)
            else
                args = unique(vcat(store.arrays, store.scalars))
                push!(outex, quote
                    local function $MAKE($(args...),)
                        local $Z
                        $tex
                    end
                end)

                ∇make, backdefs = tensor_grad(right, leftind, store)
                append!(outex, backdefs)
                outex = [:($Z = let
                    $(outex...)
                    $Eval($MAKE, $∇make)($(args...))
                end)]
            end
        else
            #===== in-place =====#
            push!(outex, tex)
        end

        # @tensor may return "throw(TensorOperations.IndexError("non-matching indices ..."
        for line in outex
            MacroTools_postwalk(line) do ex
                isexpr(ex, :call) && ex.args[1] == :throw && error(string(ex.args[2]))
                ex
            end
        end
        store.verbose>1 && verbose_tensor(outex, store)
        return outex

    # catch err
    #     store.verbose>0 && @warn "TensorOperations failed" err
    #     return nothing
    # end
end

verbose_tensor(outex, store) = begin
    printstyled("TensorOperations outex =\n", color=:blue)
    foreach(ex -> printstyled(Base.remove_linenums!(ex) , "\n", color=:green), outex)
    verboseprint(store)
end



#========== symbolic gradient ==========#
# Originally TensorGrad.jl (an unregistered package),
# all terms are again @tensor expressions.

function tensor_grad(right, leftind, store)
    dZ = Symbol(DEL, ZED)
    ∇make = Symbol(:∇, MAKE)
    backsteps, backseen = [], []

    for (B, Binds) in zip(store.arrays, store.indices)
        deltaB = Symbol(DEL, B)

        newright, extra, ijk = replace_B_with_Δ(B, Binds, right, leftind)

        append!(backsteps, extra)

        if B in backseen
            addon = macroexpand(store.mod, :( @tensor $deltaB[$(ijk...)] = $deltaB[$(ijk...)] + $newright ))
            push!(backsteps, addon)
            store.verbose>0 && @info "gradient @tensor $deltaB[$(join(ijk,','))] += $newright"
        else
            push!(backseen, B)
            symB = Symbol(DEL, B, '_', join(ijk))
            create = macroexpand(store.mod, :( @tensor( $deltaB[$(ijk...)] := $newright ) ))
            push!(backsteps, create)
            store.verbose>0 && @info "gradient @tensor $deltaB[$(join(ijk,','))] := $newright"
        end
    end

    args = unique(vcat(store.arrays, store.scalars))
    backtuple = vcat(
        map(B -> Symbol(DEL, B), unique(store.arrays)),
        map(_ -> nothing, unique(store.scalars)),
        )

    outex = [:(
        local function $∇make($dZ, $ZED, $(args...))
            $(backsteps...)
            return ($(backtuple...),)
        end
    )]

    if !isnothing(leftind) && isdefined(store.mod, :Zygote) # special case for FillArrays
        # backsteps_fill = fillarrayreplace(backsteps, dZ)
        # ex_value = :($(Symbol(dZ, :_value)) = $dZ.value)
        push!(outex, :(
            local $∇make($dZ::Zygote.Fill, $ZED, $(args...)) = $∇make(collect($dZ), $ZED, $(args...))
            # Todo: make this work without collect! Ideally it would write simpler @tensor expr.
            # local function $∇make($dZ::Zygote.Fill, $ZED, $(args...))
            #     $ex_value
            #     $(backsteps_fill...)
            #     return ($(backtuple...),)
            # end
        ))
    end

    ∇make, outex
end

using LinearAlgebra

function replace_B_with_Δ(B, Bijk, right, leftind)
    dZ = Symbol(DEL, ZED)

    # If B[ijk] occurs twice this will be wrong:
    countB = 0

    # Construct the new RHS
    out = MacroTools_postwalk(right) do x
        if @capture_(x, A_[ijk__]) && A==B && ijk == Bijk
            countB += 1
            if isnothing(leftind)
                return :(conj($dZ)) # scalar case
            else
                return :( conj($dZ[$(leftind...)]) )
            end
        else
            return x
        end
    end
    out = :(conj($out))

    # Deal with partial traces -- repeated indices on same array
    extra, deltas = [], []
    newijk = copy(Bijk)
    if !allunique(Bijk)
        for n in 1:length(Bijk)
            i = newijk[n]
            m = findfirst(isequal(i), newijk[n+1:end])
            if m != nothing
                j = Symbol('_',i,'′')
                newijk[n] = j
                delta = Symbol("_δ_",i,j)

                # This definition is added up front:
                push!(extra, quote
                    local $delta = $Diagonal(fill!(similar($B, real(eltype($B)), size($B,$n)),true))
                end)
                # This factor is included in the new RHS:
                push!(deltas, :( $delta[$i,$j] ))
            end
        end
    end
    if length(extra) > 0
        out = :( *($out, $(deltas...)) )
    end

    # I said:
    # Gradient has indices appearing only on LHS... so you need * ones()[i,j]?

    countB > 1 && throw("can't handle case of $B appearing twice with same indices")
    # Could also multiply by countB, and replace just once, would that be safe?

    return out, extra, newijk
end

#========== the end ==========#
````

## File: src/threads.jl
````
#========== cost "model" ==========#

const BLOCK = Ref(2^18)
# matmul: crossover about 70x70 on my laptop, 70^3 = 343_000, log2(70^3) = 18.3, but only 30% effect at 100^3=10^6
# batchmul: crossover between 20 & 30, log2(20^4) == 17.3, log2(30^4) == 19.6
# contract01: 1500 * 100, length 15_000, doesn't want threading
# cosine01: block 65_536, not sure if it wants
# log: vector crossover about length 10_000

"""
    COSTS = Dict(:* => 0, :log =>10, ...)

Initial cost is `1`, and every other function call adds the value from this dictionary.
Then `n = BLOCK[] ÷ cost` is the number of iterations at which the macro thinks it
worthwhile to turn on threading; you can override this with keyword `threads=n`.
"""
const COSTS = Dict(:+ => 0, :- => 0, :* => 0,
    :conj => 0, :adjoint => 0, :abs =>0, abs2 => 0,
    :getindex => 0, :getproperty => 0, :getfield => 0,
    :^ => 2, :/ => 2, :div =>2, :rem =>2, :mod =>2,
    :log => 10, :exp => 10) # and all others 10, plus 1 initially

callcost(sy, store) = store.cost += get(COSTS, sy, 10)

#========== runtime functions ==========#

"""
    threader(f!,T, Z, (A,B), (1:5,1:6), (1:7), +, block=100, keep=nothing)

Calling `f!(T, Z,A,B, 1:5,1:6, 1:7, nothing)` should do the work.
But if there are enough elements (meaning `5*6*7 > 100`)
then this will call `f!` many times in different threads.
(`block=nothing` turns the whole thing off.)

The first tuple of ranges are supposed to be safe to thread over,
probably the axes of the output `Z`.
It will subdivide the longest until either there are too few elements,
or it has spent its spawning budget, `nthreads()`.

For a scalar reduction the first tuple will be empty, and `length(Z)==1`.
Then it divides up the other axes, each accumulating in its own copy of `Z`.

`keep=nothing` means that it overwrites the array, anything else (`keep=true`) adds on.
"""
@inline function threader(fun!::F, ::Type{T}, Z::AbstractArray, As::Tuple, I0s::Tuple, J0s::Tuple, redfun, block, keep=nothing) where {F <: Function, T}
    if isnothing(block) # then threading is disabled
        fun!(T, Z, As..., I0s..., J0s..., keep)
        return nothing
    elseif !all(r -> r isa AbstractUnitRange, I0s) || !all(r -> r isa AbstractUnitRange, J0s)
        # don't thread ranges like 10:-1:1, and disable @avx too
        fun!(Array, Z, As..., I0s..., J0s..., keep)
        return nothing
    end

    Is = map(UnitRange, I0s)
    Js = map(UnitRange, J0s)
    Ielements = productlength(Is)
    Jelements = productlength(Js)
    threads = min(Threads.nthreads(), cld(Ielements * Jelements, block), Ielements)

    if length(Is) >= 1 && threads>1
        thread_halves(fun!, T, (Z, As...), Is, Js, threads, keep)
    else
        tile_halves(fun!, T, (Z, As...), Is, Js, keep)
    end
    nothing
end


"""
    ∇threader(f!,T, (dA,dB,dZ,A,B), (1:5), (1:6,1:7), block)

Again, calling `f!(T, dA,dB,dZ,A,B, 1:5,1:6, 1:7)` should do the work.

The first tuple of ranges should be safe to thread over, e.g. those in common
to all output arrays.

If there are none, then it should to take a second strategy
of dividing up the other ranges into tiles disjoint in every index,
and giving those to different threads. But this was only right for 2 indices,
and is now disabled.
"""
function ∇threader(fun!::F, ::Type{T}, As::Tuple, I0s::Tuple, J0s::Tuple, block) where {F <: Function, T}
    if isnothing(block) # then threading is disabled
        fun!(T, As..., I0s..., J0s...)
        return nothing
    elseif !all(r -> r isa AbstractUnitRange, I0s) || !all(r -> r isa AbstractUnitRange, J0s)
        # don't thread ranges like 10:-1:1, and disable @avx too
        fun!(Array, As..., I0s..., J0s...)
        return nothing
    end

    Is = map(UnitRange, I0s)
    Js = map(UnitRange, J0s)
    Ielements = productlength(Is)
    Jelements = productlength(Js)
    threads = min(Threads.nthreads(), cld(Ielements * Jelements, block), Ielements)

    if threads > 1
        thread_halves(fun!, T, As, Is, Js, threads)
    else
        tile_halves(fun!, T, As, Is, Js)
    end
    nothing
end

function thread_halves(fun!::F, ::Type{T}, As::Tuple, Is::Tuple, Js::Tuple, threads::Int, keep=nothing) where {F <: Function, T}
    if threads > 2 && rem(threads,3) == 0 # not always halves!
        I1s, I2s, I3s = trisect(Is)
        task1 = Threads.@spawn begin
            thread_halves(fun!, T, As, I1s, Js, threads÷3, keep)
        end
        task2 = Threads.@spawn begin
            thread_halves(fun!, T, As, I2s, Js, threads÷3, keep)
        end
        thread_halves(fun!, T, As, I3s, Js, threads÷3, keep)
        wait(task1)
        wait(task2)
    elseif threads > 1
        I1s, I2s = cleave(Is, maybe32divsize(T))
        task = Threads.@spawn begin
            thread_halves(fun!, T, As, I1s, Js, threads÷2, keep)
        end
        thread_halves(fun!, T, As, I2s, Js, threads÷2, keep)
        wait(task)
    else
        tile_halves(fun!, T, As, Is, Js, keep)
    end
    nothing
end

function tile_halves(fun!::F, ::Type{T}, As::Tuple, Is::Tuple, Js::Tuple, keep=nothing, final=true) where {F <: Function, T}
    # keep == nothing || keep == true || error("illegal value for keep")
    # final == nothing || final == true || error("illegal value for final")
    maxI, maxJ = maximumlength(Is), maximumlength(Js)
    maxL = tile_maxiter(T)
    if maxI < maxL && maxJ < maxL
        fun!(T, As..., Is..., Js..., keep, final)
    elseif maxI > maxJ
        I1s, I2s = cleave(Is)
        tile_halves(fun!, T, As, I1s, Js, keep, final)
        tile_halves(fun!, T, As, I2s, Js, keep, final)
    else
        J1s, J2s = cleave(Js)
        tile_halves(fun!, T, As, Is, J1s, keep, nothing)
        tile_halves(fun!, T, As, Is, J2s, true, final)
    end
    nothing
end

"""
    TILE[] = $(TILE[])
    tile_maxiter(Array{Float64}) == 64?

This now sets the maximum length of iteration of any index,
before it gets broken in half to make smaller tiles.
`TILE[]` is in bytes.
"""
const TILE = Ref(512) # this is now a length, in bytes!

function tile_maxiter(::Type{<:AbstractArray{T}}) where {T}
    isbitstype(T) || return TILE[] ÷ 8
    max(TILE[] ÷ sizeof(T), 4)
end
tile_maxiter(::Type{AT}) where {AT} = TILE[] ÷ 8 # treat anything unkown like Float64

#=

using Tullio
Z = zeros(Int, 11,9);
cnt = 0
f!(::Type, Z, i, j, ♻️, 💀) = begin
    global cnt
    Z[i,j] .= (global cnt+=1)
end
Tullio.tile_halves(f!, Array, (Z,), UnitRange.(axes(Z)), (), 4, nothing, true)
Z

  1   1   3   3   5   5   7   7   7
  1   1   3   3   5   5   7   7   7
  2   2   4   4   6   6   8   8   8
  2   2   4   4   6   6   8   8   8
  9   9  10  10  13  13  14  14  14
  9   9  10  10  13  13  14  14  14
  9   9  10  10  13  13  14  14  14
 11  11  11  11  15  15  16  16  16
 11  11  11  11  15  15  16  16  16
 12  12  12  12  15  15  16  16  16
 12  12  12  12  15  15  16  16  16

using TiledIteration
function colour!(A, n=1)
    for (i,t) in enumerate(TileIterator(axes(A), ntuple(_->n, ndims(A))))
        A[t...] .= i
    end
    A
end;
colour!(zeros(Int, 11,9), 2)

 1  1   7   7  13  13  19  19  25
 1  1   7   7  13  13  19  19  25
 2  2   8   8  14  14  20  20  26
 2  2   8   8  14  14  20  20  26
 3  3   9   9  15  15  21  21  27
 3  3   9   9  15  15  21  21  27
 4  4  10  10  16  16  22  22  28
 4  4  10  10  16  16  22  22  28
 5  5  11  11  17  17  23  23  29
 5  5  11  11  17  17  23  23  29
 6  6  12  12  18  18  24  24  30

=#

#========== scalar case ==========#

"""
    thread_scalar(f,T, z, (A,B), (1:5,1:6), +, block=100, keep=nothing)

Just like `threader`, but doesn't take any safe indices `Is`.
And `f` doesn't actually mutate anything, it returns the value...
and `z` is now a scalar, the `init` value for the reduction.
"""
@inline function thread_scalar(fun!::F, ::Type{T}, z::RT, As::Tuple, J0s::Tuple, redfun, block, keep=nothing)::RT where {F <: Function, T, RT}
    if isnothing(block) # then threading is disabled
        return fun!(T, z, As..., J0s..., keep)
    elseif !all(r -> r isa AbstractUnitRange, J0s)
        # don't thread ranges like 10:-1:1, and disable @avx too
        return fun!(Array, z, As..., J0s..., keep)
    end

    Js = map(UnitRange, J0s)
    Jelements = productlength(Js)
    threads = min(Threads.nthreads(), cld(Jelements, block), Jelements)

    if threads < 2
        return fun!(T, z, As..., Js..., keep)
    else
        return scalar_halves(fun!, T, z, As, Js, redfun, threads, keep)
    end
end

function scalar_halves(fun!::F, ::Type{T}, z::RT, As::Tuple, Js::Tuple, redfun, threads, keep=nothing)::RT where {F <: Function, T, RT}
    if threads < 1
        return fun!(T, z, As..., Js..., keep)
    else
        J1s, J2s = cleave(Js)
        S1 = z # scope
        task = Threads.@spawn begin
            S1 = scalar_halves(fun!, T, z, As, J1s, redfun, threads÷2, nothing)
        end
        S2 = scalar_halves(fun!, T, z, As, J2s, redfun, threads÷2, keep)
        wait(task)
        return redfun(S1, S2)
    end
end

#========== tuple functions ==========#

@inline productlength(Is::Tuple) = prod(length.(Is))
@inline productlength(Is::Tuple, Js::Tuple) = productlength(Is) * productlength(Js)

@inline maximumlength(Is::Tuple) = max(length.(Is)...)
@inline maximumlength(::Tuple{}) = 0

@inline maybe32divsize(::Type{<:AbstractArray{T}}) where T<:Number = max(1, 32 ÷ sizeof(T))
@inline maybe32divsize(::Type) = 4

"""
    cleave((1:10, 1:20, 5:15)) -> lo, hi
Picks the longest of a tuple of ranges, and divides that one in half.
"""
@inline cleave(::Tuple{}, n::Int=4) = (), ()
@inline function cleave(ranges::Tuple{UnitRange}, step::Int=4)
    r1 = first(ranges)
    cleft = findcleft(r1, step)
    tuple(first(r1):cleft), tuple(cleft+1:last(r1))
end
@inline function cleave(ranges::Tuple{UnitRange,UnitRange}, step::Int=4)
    r1, r2 = ranges
    if length(r1) > length(r2)
        cleft = findcleft(r1, step)
        return tuple(first(r1):cleft, r2), tuple(cleft+1:last(r1), r2)
    else
        cleft = findcleft(r2, step)
        return tuple(r1, first(r2):cleft), tuple(r1, cleft+1:last(r2))
    end
end
@inline @generated function cleave(ranges::Tuple{Vararg{UnitRange,N}}, step::Int=4) where {N}
    ex_finds = [quote
        li = length(ranges[$i])
        if li>l
            c = $i
            l = li
        end
    end for i in 1:N]
    ex_alpas = [:($i==c ? (first(ranges[$i]):cleft) : (ranges[$i])) for i in 1:N]
    ex_betas = [:($i==c ? (cleft+1:last(ranges[$i])) : (ranges[$i])) for i in 1:N]
    quote
        c, l = 0, 0
        $(ex_finds...)
        cleft = findcleft(ranges[c], step)
        tuple($(ex_alpas...)), tuple($(ex_betas...))
    end
end

@inline function findcleft(r::UnitRange, step::Int)
    if length(r) >= 2*step
        minimum(r) - 1 + step * div(length(r), step * 2)
    else
        # minimum(r) - 1 + div(length(r), 2, RoundNearest) # not in Julia 1.3
        minimum(r) - 1 + round(Int, length(r)/2)
    end
end

#=
@btime Tullio.cleave(z[],4)  setup=(z=Ref((1:200, 1:500, 1:300)))
@btime Tullio.cleave(z[],4)  setup=(z=Ref((1:200, 1:50)))
@btime Tullio.cleave(z[],4)  setup=(z=Ref((5:55,)))
=#

"""
    trisect((1:10, 1:20, 5:15)) -> lo, mid, hi

Just like `cleave`, but makes 3 pieces, for 6-core machines.
"""
@inline trisect(::Tuple{}) = (), (), ()
@inline trisect(ranges::Tuple{UnitRange}) = map(tuple, findthree(first(ranges)))
@inline function trisect(ranges::Tuple{UnitRange,UnitRange})
    r1, r2 = ranges
    if length(r1) > length(r2)
        a,b,c = findthree(r1)
        return (a,r2), (b,r2), (c,r2)
    else
        a,b,c = findthree(r2)
        return (r1,a), (r1,b), (r1,c)
    end
end
@inline @generated function trisect(ranges::Tuple{Vararg{UnitRange,N}}) where {N}
    ex_finds = [quote
        li = length(ranges[$i])
        if li>l
            c = $i
            l = li
        end
    end for i in 1:N]
    ex_alpas = [:($i==c ? (lo) : (ranges[$i])) for i in 1:N]
    ex_betas = [:($i==c ? (mid) : (ranges[$i])) for i in 1:N]
    ex_gammas = [:($i==c ? (hi) : (ranges[$i])) for i in 1:N]
    quote
        c, l = 0, 0
        $(ex_finds...)
        lo,mid,hi = findthree(ranges[c])
        tuple($(ex_alpas...)), tuple($(ex_betas...)), tuple($(ex_gammas...))
    end
end

@inline function findthree(r::UnitRange)
    d = div(length(r), 3)
    i0 = first(r)
    (i0 : i0+d-1), (i0+d : i0+2d-1), (i0+2d : i0+length(r)-1)
end

#========== the end ==========#
````

## File: src/tools.jl
````
#========== a mutable, typeless, almost-namedtuple ==========#

struct DotDict
    store::Dict{Symbol,Any}
end
DotDict(;kw...) = DotDict(Dict(pairs(kw)...))

Base.parent(x::DotDict) = getfield(x, :store)

Base.propertynames(x::DotDict) = Tuple(sort(collect(keys(parent(x)))))
Base.getproperty(x::DotDict, s::Symbol) = getindex(parent(x), s)
function Base.setproperty!(x::DotDict, s::Symbol, v)
    s in propertynames(x) || throw("DotDict has no field $s")
    T = typeof(getproperty(x, s))
    if T == Nothing
        setindex!(parent(x), v, s)
    else
        setindex!(parent(x), convert(T, v), s)
    end
end

function Base.show(io::IO, x::DotDict)
    print(io, "DotDict(")
    strs = map(k -> string(k, " = ", getproperty(x, k)), propertynames(x))
    print(io, join(strs, ", "), ")")
end

verboseprint(store) = begin
    printstyled("┌ store.\n", color=:blue)
    foreach(propertynames(store)) do k
        printstyled("│   $k = ", color=:blue)
        r = getproperty(store, k)
        if k ∉ [:outpre, :outex]
            println(repr(r))
        else
            str = repr(verbosetidy.(r))
            println(first(str, 150), length(str)>150 ? " ..." : "")
        end
    end
    printstyled("└\n", color=:blue)
    if store.verbose == 3
        if haskey(store, :outpre)
            printstyled("store.outpre = \n", color=:blue)
            printstyled(verbosetidy(store.outpre) , "\n", color=:green)
        end
        if haskey(store, :outex)
            printstyled("\nstore.outex = \n", color=:blue)
            printstyled(verbosetidy(store.outpre) , "\n", color=:green)
        end
    end
end

#========== capture macro ==========#
# My faster, more limited, version:

"""
    @capture_(ex, A_[ijk__])

Faster drop-in replacement for `MacroTools.@capture`, for a few patterns only:
* `A_[ijk__]` and `A_{ijk__}`
* `[ijk__]`
* `A_.field_`
* `A_ := B_` and  `A_ = B_` and `A_ += B_` etc.
* `f_(x_)`
"""
macro capture_(ex, pat::Expr)

    H = QuoteNode(pat.head)

    A,B = if pat.head in [:ref, :curly] && length(pat.args)==2 &&
        _endswithone(pat.args[1]) && _endswithtwo(pat.args[2]) # :( A_[ijk__] )
        _symbolone(pat.args[1]), _symboltwo(pat.args[2])

    elseif pat.head == :. && pat.args[2] isa QuoteNode &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2].value) # :( A_.field_ )
        _symbolone(pat.args[1]), _symbolone(pat.args[2].value)

    elseif pat.head == :call  && length(pat.args)==2 &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2]) # :( f_(x_) )
        _symbolone(pat.args[1]), _symbolone(pat.args[2])

    elseif pat.head in [:call, :(=), :(:=), :+=, :-=, :*=, :/=, :^=] &&
        _endswithone(pat.args[1]) && _endswithone(pat.args[2]) # :( A_ += B_ )
        _symbolone(pat.args[1]), _symbolone(pat.args[2])

    # elseif pat.head == :call  && length(pat.args)==3 && pat.args[1] == :!= &&
    #     _endswithone(pat.args[2]) && _endswithone(pat.args[3]) # :( A_ != B_ )
    #     H = QuoteNode(pat.args[1])
    #     _symbolone(pat.args[2]), _symbolone(pat.args[3])

    elseif pat.head == :vect && _endswithtwo(pat.args[1]) # :( [ijk__] )
        _symboltwo(pat.args[1]), gensym(:ignore)

    else
        throw("@capture_ doesn't work on pattern $pat")
    end

    @gensym res
    quote
        $A, $B = nothing, nothing
        $res = $_trymatch($ex, Val($H))
        if $res === nothing
            false
        else
            $A, $B = $res
            true
        end
    end |> esc
end

_endswithone(ex) = endswith(string(ex), '_') && !_endswithtwo(ex)
_endswithtwo(ex) = endswith(string(ex), "__")

_symbolone(ex) = Symbol(string(ex)[1:end-1])
_symboltwo(ex) = Symbol(string(ex)[1:end-2])

_getvalue(::Val{val}) where {val} = val

_trymatch(s, v) = nothing # Symbol, or other Expr
_trymatch(ex::Expr, pat::Union{Val{:ref}, Val{:curly}}) = # A_[ijk__] or A_{ijk__}
    if ex.head === _getvalue(pat)
        ex.args[1], ex.args[2:end]
    else
        nothing
    end
_trymatch(ex::Expr, ::Val{:.}) = # A_.field_
    if ex.head === :. && ex.args[2] isa QuoteNode
        ex.args[1], ex.args[2].value
    else
        nothing
    end
_trymatch(ex::Expr, pat::Val{:call}) =
    if ex.head === _getvalue(pat) && length(ex.args) == 2
        ex.args[1], ex.args[2]
    else
        nothing
    end
_trymatch(ex::Expr, pat::Union{Val{:(=)}, Val{:(:=)}, Val{:(+=)}, Val{:(-=)}, Val{:(*=)}, Val{:(/=)}, Val{:(^=)}}) =
    if ex.head === _getvalue(pat)
        ex.args[1], ex.args[2]
    else
        nothing
    end
# _trymatch(ex::Expr, pat::Val{:!=}) =
#     if ex.head === :call && length(ex.args) == 3 && ex.args[1] == :!=
#         ex.args[2], ex.args[3]
#     else
#         nothing
#     end
_trymatch(ex::Expr, ::Val{:vect}) = # [ijk__]
    if ex.head === :vect
        ex.args, nothing
    else
        nothing
    end


# Cases for Tullio:
# @capture(ex, B_[inds__].field_) --> @capture_(ex, Binds_.field_) && @capture_(Binds, B_[inds__])


#========== postwalk ==========#
# Copied verbatim from here:
# https://github.com/MikeInnes/MacroTools.jl/blob/master/src/utils.jl

walk(x, inner, outer) = outer(x)
walk(x::Expr, inner, outer) = outer(Expr(x.head, map(inner, x.args)...))

"""
    postwalk(f, expr)
Applies `f` to each node in the given expression tree, returning the result.
`f` sees expressions *after* they have been transformed by the walk. See also
`prewalk`.
"""
postwalk(f, x) = walk(x, x -> postwalk(f, x), f)

"""
    prewalk(f, expr)
Applies `f` to each node in the given expression tree, returning the result.
`f` sees expressions *before* they have been transformed by the walk, and the
walk will be applied to whatever `f` returns.
This makes `prewalk` somewhat prone to infinite loops; you probably want to try
`postwalk` first.
"""
prewalk(f, x)  = walk(f(x), x -> prewalk(f, x), identity)

replace(ex, s, s′) = prewalk(x -> x == s ? s′ : x, ex)

const MacroTools_prewalk = prewalk
const MacroTools_postwalk = postwalk

#========== prettify ==========#

using Base.Meta: isexpr

verbosetidy(expr) = MacroTools_postwalk(expr) do ex
        if isexpr(ex, :block)
            args = filter(x -> !(x isa LineNumberNode || x == nothing), ex.args)
            if length(args) == 1 && Meta.isexpr(args[1], :block)
                # disallow block(block(stuff))
                args[1]
            else
                Expr(ex.head, args...)
            end
        elseif isexpr(ex, :macrocall) && length(ex.args) >= 2
            # line number after macro name can't be dropped, but can be nothing:
            Expr(ex.head, ex.args[1], nothing, filter(x -> !(x isa LineNumberNode), ex.args[3:end])...)
        else
            ex
        end
    end

#========== the end ==========#
````

## File: src/Tullio.jl
````
module Tullio

export @tullio

@nospecialize

include("tools.jl")

include("macro.jl")

include("tensor.jl")

include("symbolic.jl")

include("forward.jl")

include("einsum.jl")

@specialize

include("eval.jl")

include("shifts.jl")

include("threads.jl")

include("precompile.jl")

end # module
````

## File: test/cuda.jl
````
using Tullio, Test
using CUDA, KernelAbstractions
CUDA.allowscalar(false)
using Tracker, ForwardDiff
@tullio grad=Base

# matmul
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);
@test A * B ≈ mul(A, B)
@test cu(A * B) ≈ mul(cu(A), cu(B))

# gradient
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
@test ΔA ≈ ones(3,500) * B'
@test cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1]

# shifts
@tullio D[i,j] := A[i,j+k]  k in 0:10
@test axes(D) == (1:3, 1:30)
@tullio cD[i,j] := cu(A)[i,j+k]  k in 0:10
@test cD isa CuArray
@test cD ≈ cu(D)

#=
# ranges
@tullio E[i,j] := A[i,j+k-1] + (-1:0.5:1)[k]
@test axes(E) == (1:3, 1:36)
@tullio cE[i,j] := cu(A)[i,j+k-1] + (-1:0.5:1)[k]
@test cE isa CuArray
@test cE ≈ cu(E)
=#

# product
@tullio (*) F[j] := A[i,j]
@test F ≈ vec(prod(A, dims=1))
@tullio (*) cF[j] := cu(A)[i,j]
@test cF ≈ cu(F)

# maximum
g(A) = @tullio (max) G[j] := A[i,j]
@test g(A) == vec(maximum(A, dims=1))
A0 = zero(A);
A0[findmax(A, dims=1)[2]] .= 1
@test A0 ≈ Tracker.gradient(sum∘g, A)[1]
@test g(cu(A)) isa CuArray
@test g(cu(A)) ≈ cu(g(A))
@test cu(A0) ≈ Tracker.gradient(sum∘g, cu(A))[1]

# functions
h(A) = @tullio H[j] := exp(A[i,j]) / log(A[i,j])
@test h(cu(A)) isa CuArray
@test h(cu(A)) ≈ cu(h(A))
A1 = Tracker.gradient(sum∘h, A)[1]
@test cu(A1) ≈ Tracker.gradient(sum∘h, cu(A))[1]

#= # broken by https://github.com/mcabbott/Tullio.jl/pull/31
# scalar
@tullio s := cu(A)[i,j]^2
@test s ≈ sum(abs2, A)
@tullio s += cu(B)[i,j]^2
@test s ≈ sum(abs2, A) + sum(abs2, B)
=#

# https://github.com/mcabbott/Tullio.jl/issues/96
A, B, C = CUDA.rand(2,2,2), CUDA.rand(2,2), CUDA.rand(2,2,2);
@tullio A[k,i,a] = tanh(B[i,a] + C[k,i,a]) 
A2 = similar(A)
struct Bee{T}; B::T; end
B2 = Bee(B)
@test A ≈ @tullio A2[k,i,a] = tanh(B2.B[i,a] + C[k,i,a])
````

## File: test/einsum.jl
````
using Test
using LinearAlgebra # dot

# using Einsum
using Tullio: @einsum

@testset "Test that vars in Main aren't overwritten by einsum" begin
    i = -1
    y = randn(10)
    @einsum x[i] := y[i]
    @test i == -1
end

@testset "Test that B is overwritten by := operator" begin
    B = randn(10, 10)
    A = randn(5, 10)
    @einsum B[i, j] := A[i, j] # this should run without a problem
    @test size(B) == size(A)
end

@testset "CP decomposition test case" begin

    # preallocated test case
    A = zeros(5, 6, 7)
    B = similar(A)
    C = similar(A)

    X = randn(5, 2)
    Y = randn(6, 2)
    Z = randn(7, 2)

    @einsum A[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    # @einsimd B[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    # @vielsum C[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]

    for i = 1:5
        for j = 1:6
            for k = 1:7
                s = 0.0
                for r = 1:2
                    s += X[i, r] * Y[j, r] * Z[k, r]
                end
                @test isapprox(A[i, j, k], s)
                # @test isapprox(B[i, j, k], s)
                # @test isapprox(C[i, j, k], s)
            end
        end
    end

    # without preallocation
    @einsum A2[i, j, k] := X[i, r] * Y[j, r] * Z[k, r]
    @test isapprox(A, A2)

end

@testset "Interesting test case, can throw an error that local vars are declared twice." begin
    A = zeros(5, 6, 7)
    X = randn(5, 2)
    Y = randn(6, 2)
    Z = randn(7, 2)
    if true
        @einsum A[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    else
        @einsum A[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    end
end

@testset "From #21: local `T` does not interfer with internal T" begin
    function test(x::Vector{T}, y::Vector{T}) where T
        @einsum z := x[i] * y[i]
        return z
    end
    @test_nowarn test(rand(3), rand(3))
end

@testset "From #20: local `s` does not interfere with internal s" begin
    x = rand(2, 3)
    @test_nowarn @einsum y[i] := x[i, s]
end

@testset "At one point this threw an error because the lhs had no indices/arguments" begin
    x = randn(10)
    y = randn(10)
    @einsum k := x[i] * y[i]
    @test isapprox(k, dot(x, y))
end

@testset "Elementwise multiplication (this should create nested loops with no no summation.)" begin
    x = randn(10)
    y = randn(10)
    @einsum k[i] := x[i] * y[i]
    # @einsimd k2[i] := x[i] * y[i]
    # @vielsum k3[i] := x[i] * y[i]
    @test isapprox(k, x .* y)
    # @test isapprox(k2, x .* y)
    # @test isapprox(k3, x .* y)
end

@testset "Transpose a block matrix" begin
    z = [rand(2, 2) for i = 1:2, j = 1:2]
    @einsum t[i, j] := transpose(z[j, i])
    @test isapprox(z[1, 1],  t[1, 1]')
    @test isapprox(z[2, 2],  t[2, 2]')
    @test isapprox(z[1, 2],  t[2, 1]')
    @test isapprox(z[2, 1],  t[1, 2]')
end

@testset "Mapping functions" begin
    A = randn(10, 10)
    @einsum B[i, j] := exp(A[i, j])
    @test isapprox(exp.(A), B)
end

@testset "Example from numpy" begin
    A = reshape(collect(1:25), 5, 5)
    @einsum B[i] := A[i, i]
    @test all(B .== [1, 7, 13, 19, 25])

    # @vielsum C[i] := A[i, i]
    # @test all(C .== [1, 7, 13, 19, 25])

end

@testset "Adding a scalar, to-done" begin

    A = collect(reshape(1:12, 3,4))
    @einsum A[i, j] = A[i,j] + 50
    @test A == collect(50 .+ reshape(1:12, 3,4))

end

@testset "Test in-place operations" begin
    A = randn(5, 6, 7)
    B = randn(5, 6, 7)
    A1 = copy(A)
    B1 = copy(B)

    X = randn(5, 2)
    Y = randn(6, 2)
    Z = randn(7, 2)

    @einsum A[i, j, k] += X[i, r] * Y[j, r] * Z[k, r]
    # @einsimd B[i, j, k] += X[i, r] * Y[j, r] * Z[k, r]

    for i = 1:5
        for j = 1:6
            for k = 1:7
                s = 0.0
                for r = 1:2
                    s += X[i, r] * Y[j, r] * Z[k, r]
                end
                @test isapprox(A[i, j, k], A1[i, j, k] + s)
                # @test isapprox(B[i, j, k], B1[i, j, k] + s)
            end
        end
    end

end

@testset "scalar += dot" begin

    x = randn(10)
    y = randn(10)
    k0 = randn()
    k = k0
    @einsum k += x[i] * y[i]
    @test isapprox(k, k0 + dot(x, y))

end

@testset "test *= operator" begin

    A = randn(5, 6, 7)
    B = randn(5, 6, 7)
    A1 = copy(A)
    B1 = copy(B)

    X = randn(5, 2)
    Y = randn(6, 2)
    Z = randn(7, 2)

    @einsum A[i, j, k]  *= X[i, r] * Y[j, r] * Z[k, r]
    # @einsimd B[i, j, k] *= X[i, r] * Y[j, r] * Z[k, r]

    for i = 1:5
        for j = 1:6
            for k = 1:7
                s = 0.0
                for r = 1:2
                    s += X[i, r] * Y[j, r] * Z[k, r]
                end
                @test isapprox(A[i, j, k], A1[i, j, k] * s)
                # @test isapprox(B[i, j, k], B1[i, j, k] * s)
            end
        end
    end

    x = randn(10)
    y = randn(10)
    k0 = randn()
    k = k0
    @einsum k *= x[i] * y[i]
    @test isapprox(k, k0 * dot(x, y))
end


@testset "Test offsets" begin
    X = randn(10)

    # without preallocation
    @einsum A[i] := X[i + 5]
    @test_broken size(A) == (5,)  # here Tullio returns an OffsetArray
    @test_broken all(A .== X[6:end])

    # with preallocation
    B = zeros(10)
    @einsum B[i] = X[i + 5]
    @test size(B) == (10,)
    @test all(B[1:5] .== X[6:end])
end

@testset "Test symbolic offsets" begin
    offset = 5
    X = randn(10)

    # without preallocation
    # @einsum A[i] := X[i + :offset] # error on 1.0
    @einsum A[i] := X[i + $offset]  # here Tullio returns an OffsetArray
    @test_broken size(A) == (5,)
    @test_broken all(A .== X[6:end])

    # with preallocation
    B = zeros(10)
    # @einsum B[i] = X[i + :offset] # error on 1.0
    @einsum B[i] = X[i + $offset]
    @test size(B) == (10,)
    @test all(B[1:5] .== X[6:end])
end


@testset "Test adding/subtracting constants" begin
    k = 5
    X = randn(10)

    # without preallocation
    @einsum A[i] := X[i] + k
    @einsum B[i] := X[i] - k
    @test isapprox(A, X .+ k)
    @test isapprox(B, X .- k)

    @einsum A[i] := X[i] + $k  # Tullio prefers $k, it becomes a function argument
    @einsum B[i] := X[i] - $k
    @test isapprox(A, X .+ k)
    @test isapprox(B, X .- k)

    # with preallocation
    C, D = zeros(10), zeros(10)
    @einsum C[i] = X[i] + $k
    @einsum D[i] = X[i] - $k
    @test isapprox(C, X .+ k)
    @test isapprox(D, X .- k)
end

@testset "Test multiplying/dividing constants" begin
    k = 5
    X = randn(10)

    # without preallocation
    @einsum A[i] := X[i] * k
    @einsum B[i] := X[i] / k
    @test isapprox(A, X .* k)
    @test isapprox(B, X ./ k)

    @einsum A[i] := X[i] * $k  # Tullio prefers $k, it becomes a function argument
    @einsum B[i] := X[i] / $k
    @test isapprox(A, X .* k)
    @test isapprox(B, X ./ k)

    # with preallocation
    C, D = zeros(10), zeros(10)
    @einsum C[i] = X[i] * k
    @einsum D[i] = X[i] / k
    @test isapprox(C, X .* k)
    @test isapprox(D, X ./ k)

    @einsum C[i] = X[i] * $k  # Tullio prefers $k, it becomes a function argument
    @einsum D[i] = X[i] / $k
    @test isapprox(C, X .* k)
    @test isapprox(D, X ./ k)
end

@testset "Test indexing with a constant" begin
    A = randn(10, 2)
    j = 2
    # @einsum B[i] := A[i, :j] # error on Julia 1.0, i.e. this broke at some point in Einsum.jl
    @einsum B[i] := A[i, $j]   # Tullio's notation, here the $ is not optional!
    @test all(B .== A[:, j])
    @einsum C[i] := A[i, 1]
    @test all(C .== A[:, 1])

    D = zeros(10, 3)
    # @einsum D[i, 1] = A[i, :j]
    @einsum D[i, 1] = A[i, $j]
    @test isapprox(D[:, 1], A[:, j])
    # @einsum D[i, :j] = A[i, :j]
    @einsum D[i, $j] = A[i, $j]
    @test isapprox(D[:, j], A[:, j])
end

@testset "Better type inference on allocating arrays" begin
    B1 = ones(Int, 5)
    B2 = ones(Float32, 5)
    B3 = ones(5)
    C = randn(5)
    @einsum A1[i, j] := B1[i] * C[j]
    @einsum A2[i, j] := B2[i] * C[j]
    @einsum A3[i, j] := B3[i] * C[j]

    @test eltype(A1) == Float64
    @test eltype(A2) == Float64
    @test eltype(A3) == Float64
    @test isapprox(A1, A3)
    @test isapprox(A2, A3)
end

@testset "Scalar output, issue #37" begin

    a = [1 0; 0 1]
    @einsum b := a[i,i]
    @einsum c[] := a[i,i]

    @test b == c[] == 2
    @test b isa Int
    @test c isa Array{Int,0}

end

#========== some extra things from issues not tests ==========#

@testset "shifts, issue 6" begin
    # https://github.com/ahwillia/Einsum.jl/issues/6
    # discussion of things which there were attempts to make work

    A = zeros(10);
    X = randn(10);
    Y = randn(10);
    @einsum A[j] = X[j]*Y[j+3]
    @test isapprox(A, X.*[Y[4:end];zeros(3)])

    @einsum A2[j] := X[j+3]
    @test axes(A2,1) == -2:7

    offset = 3
    @einsum A3[j] := X[j+$offset]
    @test axes(A3,1) == -2:7

    @einsum A5[i] := X[i-5]
    @test size(A5) == (10,)
    @test_broken all(A5[6:end] .== X[1:5]) # not true in Tullio's conventions

    @einsum A6[i] := X[i+3]*X[i-3]
    # @test size(A) == (7,) # that can't make sense
    # @test isapprox(A[4:7], X[7:end].*X[1:4])

end
@testset "shifts, issue 12" begin
    # https://github.com/ahwillia/Einsum.jl/pull/12

    B = collect(1:10)

    # "produce errors?" -- no!
    A = zeros(5)
    @einsum A[i] = B[i+5]
    intersect(axes(A, 1), axes(B, 1) .- 5) # legal indices 1:5
    @test A[1] == 6

    A = zeros(5)
    @einsum A[i] = B[i-5] #
    intersect(axes(A, 1), axes(B, 1) .+ 5) # empty range, so nothing changed
    @test all(A .== 0)

    # "legal?" -- yes!
    A = zeros(10)
    @einsum A[i] = B[i+5]
    @test A == [6, 7, 8, 9, 10, 0, 0, 0, 0, 0]

    A = zeros(10)
    @einsum A[i] = B[i-5]
    @test A == [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]

#=
    # "Side note. This is silly, but would be kind of cool to support this:"

    B = collect(1:10)
    A = zeros(10)
    @einsum A[i] = B[i>>5]
    A == [6, 7, 8, 9, 10, 1, 2, 3, 4, 5]

    @einsum A[i] = B[i<<2]
    A == [3, 4, 5, 6, 7, 8, 9, 10, 1, 2]

    # Cyclic indices would be neat but this isn't the notation
=#
end

@tullio avx=true
````

## File: test/gradients.jl
````
#=
This file is run several times
* with grad=Base vs grad=Dual
* with Tracker, Zygote
* using KernelAbstractions, LoopVectorization, TensorOperations
=#

using Tullio, Test, ForwardDiff, Random
# using Tracker; _gradient(x...) = Tracker.gradient(x...); GRAD = :Tracker; macro printline() end

function gradtest(f, dims)
    x = randn(dims...)
    grad = ForwardDiff.gradient(x -> sum(sin, f(x)), x)
    grad ≈ _gradient(x -> sum(sin, f(x)), x)[1]
end

@testset "simple" begin

    @test _gradient(x -> sum(@tullio y[i] := 2*x[i]), rand(3))[1] == [2,2,2]
    @test _gradient(x -> sum(@tullio y[i] := 2*x[i] + i), rand(3))[1] == [2,2,2]

    # two contributions
    g2(x) = @tullio y[i, j] := 1 * x[i] + 1000 * x[j]  avx=false
    mat = [1 1 3; 1 1 5; 7 7 7]
    g_fd = ForwardDiff.gradient(x -> sum(mat .* g2(x)), rand(3))
    @test g_fd ≈ _gradient(x -> sum(mat .* g2(x)), rand(3))[1]

    # larger array, no shared indices -- https://github.com/mcabbott/Tullio.jl/issues/14
    r100 = randn(100)
    g_fd = ForwardDiff.gradient(x -> sum(sin, g2(x)), r100)
    @test g_fd ≈ _gradient(x -> sum(sin, g2(x)), r100)[1]

    r100 = randn(100)

    # scalar output
    s2(x) = @tullio s := exp(x[i]) / x[j]
    @test _gradient(s2, r100)[1] ≈ ForwardDiff.gradient(s2, r100)

    # two arrays, and a sum
    h2(x,y) = @tullio z[i] := x[i,j] + y[j,i]
    @test _gradient(sum∘h2, rand(2,3), rand(3,2)) == (ones(2,3), ones(3,2))

    # nontrivial function
    flog(x,y) = @tullio z[i] := log(x[i,j]) / y[j,i]  avx=false  # new failure LoopVectorization v0.12.14? only on CI?
    r_x, r_y = rand(2,3), rand(3,2)
    fx = ForwardDiff.gradient(x -> sum(flog(x, r_y)), r_x)
    fy = ForwardDiff.gradient(y -> sum(flog(r_x, y)), r_y)
    @test fx ≈ _gradient(sum∘flog, r_x, r_y)[1]
    @test fy ≈ _gradient(sum∘flog, r_x, r_y)[2]

    # classic
    mm(x,y) = @tullio z[i,j] := 2 * x[i,k] * y[k,j]  avx=false # new?
    x1 = rand(3,4);
    y1 = rand(4,5);
    z1 = x1 * y1
    dx, dy = _gradient(sum∘mm, x1, y1)
    @test dx ≈ 2 * ones(3,5) * y1'
    @test dy ≈ 2 * x1' * ones(3,5)

    # abs, abs2
    va = [1,-2,3,-4,5]
    abs_grad = ForwardDiff.gradient(v -> sum(abs, 1 .+ v.^2), va)
    @test abs_grad ≈ _gradient(v -> (@tullio s := abs(1 + v[i]^2)), va)[1]
    abs2_grad = ForwardDiff.gradient(v -> sum(abs2, 1 .+ v.^2), va)
    @test abs2_grad ≈ _gradient(v -> (@tullio s := abs2(1 + v[i]^2)), va)[1]

end

@printline

@testset "zero-arrays" begin

    # Using zero-dim arrays fails on ReverseDiff & Tracker
    # Tracker.gradient(x -> x[], fill(1.0))
    # ReverseDiff.gradient(x -> x[], fill(1.0)) # is ambiguous
    if GRAD in [:Tracker, :ReverseDiff]
        @test_skip _gradient(x -> sum(@tullio y[] := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)
    else
        @test _gradient(x -> sum(@tullio y[] := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)
    end
    # one-element vectors are fine:
    @test _gradient(x -> sum(@tullio y[1] := log(x[i]) avx=false), collect(1:3.0))[1] == 1 ./ (1:3)  # new failure LoopVectorization v0.12.14? only on CI?
    # which is what's now used for this:
    @test _gradient(x -> (@tullio y := log(x[i])), collect(1:3.0))[1] == 1 ./ (1:3)

end
@testset "gather/scatter" begin

    inds = vcat(1:3, 1:2)
    @test _gradient(x -> sum(@tullio y[i] := x[inds[i]]), rand(3))[1] == [2,2,1]

    _gradient(x -> sum(@tullio y[inds[i]] := x[i]), rand(5))[1] == [1,1,1,1,1]
    ForwardDiff.gradient(x -> sum(@tullio y[inds[i]] := x[i]), rand(5)) == [0,0,1,1,1]
    # This difference may be another edge case like multiple maxima?

    ind2 = rand(1:10, 1024) # many repeats
    dx2 = ForwardDiff.gradient(x -> sum(@tullio y[i] := x[ind2[i]] + x[i]), rand(1024))
    @test_skip dx2 ≈ _gradient(x -> sum(@tullio y[i] := x[ind2[i]] + x[i]), rand(1024))[1]

    ind3 = vcat(unique(rand(2:1024, 10)), 1) # many missing, no repeats, but always includes 1
    g3 = ForwardDiff.gradient(x -> sum(@tullio y[ind3[i]] := i^2 * x[i]), ones(size(ind3)))
    @test g3 ≈ _gradient(x -> sum(@tullio y[ind3[i]] := i^2 * x[i]), ones(size(ind3)))[1]
    # You get weird errors here if indices of y don't start at 1.

    # 1.6 failure on CI, with rand(1:1024, 10)
    # [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 0.0, 64.0, 81.0, 100.0, 121.0] ≈ [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0]

end

@printline

@testset "shifts, etc" begin

    c1(N,K) = @tullio M[x,y,c] := N[x+i-1, y+j-1,c] * K[i,j]
    m1 = rand(10,10,2)
    k1 = rand(3,3)
    g_m = ForwardDiff.gradient(N -> sum(sin, c1(N, k1)), m1)
    g_k = ForwardDiff.gradient(K -> sum(sin, c1(m1, K)), k1)
    @test_skip g_m ≈ _gradient(N -> sum(sin, c1(N, k1)), m1)[1]  atol=0.01 # works at repl, fails in tests
    @test g_k ≈ _gradient(K -> sum(sin, c1(m1, K)), k1)[1]  atol=0.01

    c2(mat, kern) = @tullio out[x,y,n] := begin
            i = mod(x+a, axes(mat,1))
            j = mod(y+b, axes(mat,2))
            @inbounds mat[i,j,n] * abs(kern[a,b])
        end (x in axes(mat,1), y in axes(mat,2)) grad=Dual

    if Tullio._GRAD[] == :Dual
        g_m = ForwardDiff.gradient(N -> sum(sin, c2(N, k1)), m1)
        g_k = ForwardDiff.gradient(K -> sum(sin, c2(m1, K)), k1)
        @test g_m ≈ _gradient(N -> sum(sin, c2(N, k1)), m1)[1]  atol=0.01
        @test g_k ≈ _gradient(K -> sum(sin, c2(m1, K)), k1)[1]  atol=0.01
    end

end
@testset "mod, clamp, pad" begin

    fmod(x) = @tullio y[i] := x[mod(i)]  i in 1:5  avx=false # fails on 1.4, LV 0.8
    fclamp(x) = @tullio y[i] := x[clamp(i)]  i in 1:5  avx=false
    fpad(x) = @tullio y[i] := x[pad(i-2,2)]
    @test _gradient(sum∘fmod, ones(3))[1] == [2,2,1]
    @test _gradient(sum∘fclamp, ones(3))[1] == [1,1,3]
    @test _gradient(sum∘fpad, ones(3))[1] == [1,1,1]

end
@testset "@inferred" begin

    h2(x,y) = @tullio z[i] := x[i,j] + y[j,i]  # as above
    flog(x,y) = @tullio z[i] := log(x[i,j]) / y[j,i]  avx=false  # new failure LoopVectorization v0.12.14? only on CI?

    mat = rand(3,3)
    @test @inferred(h2(mat, mat)) ≈ vec(sum(mat .+ mat', dims=2))
    @test @inferred(flog(mat, mat)) isa Vector

    if GRAD == :Zygote
        @test_broken @inferred(_gradient(sum∘h2, rand(2,3), rand(3,2))) isa Tuple
        @test_broken @inferred(_gradient(sum∘flog, mat, mat)) isa Tuple
    else
        @test @inferred(_gradient(sum∘h2, rand(2,3), rand(3,2))) isa Tuple
        @test @inferred(_gradient(sum∘flog, mat, mat)) isa Tuple
    end

end

@printline

@testset "from TensorTrace" begin
    # These can all be handled using TensorOperations

    triv1(x) = @tullio A[i,j] := 2 * x[i,j]
    @test gradtest(triv1, (2,3))

    r32 = randn(3,2);
    r312 = randn(3,1,2);

    ## trace!
    tr1(x) = @tullio T[k] := 22 * x[i,i,k]
    @test gradtest(tr1, (3,3,4))

    tr2(x) = @tullio T[k] := 22 * x[i,i,k,j,j]
    @test gradtest(tr2, (3,3,4,7,7))

    ## contract! A
    con1(x) = @tullio C[i,j] := 5 * x[i,k] * r32[k,j]  avx=false  # https://github.com/mcabbott/Tullio.jl/pull/144
    @test gradtest(con1, (2,3))

    r22 = rand(2,2);

    con3(x) = @tullio C[i,j,m,n] := x[i,j,k] * r312[k,m,n]  avx=false # I think leading size-1 dims are the problem
    @test gradtest(con3, (1,2,3))

    con4(x) = @tullio C[i,m] := x[i,kk,k] * r312[k,m,kk]  avx=false
    @test gradtest(con4, (1,2,3))

    con5(x) = @tullio C[j,i,n,m] := 44 * x[i,j,k] * r312[k,m,n]  avx=false
    @test gradtest(con5, (1,2,3))

    r392 = randn(3,9,2);
    con6(x) = @tullio C[n,i,m,j] := x[i,j,k] * r392[k,m,n]
    @test gradtest(con6, (9,2,3))

    con7(x) = @tullio C[m,n,j,i] := 44 * x[i,j,k] * r392[k,m,n]
    @test gradtest(con7, (9,2,3))

    @printline

    ## contract! B
    con8b(x) = @tullio K[i,j] := 5 * r32[i,k] * x[k,j]  avx=false
    @test gradtest(con8b, (2,3))

    con9b(x) = @tullio K[i,j,m,n] := r312[i,j,k] * x[m,k,n]  avx=false  # https://github.com/mcabbott/Tullio.jl/pull/144
    @test gradtest(con9b, (1,2,3))

    con10b(x) = @tullio K[n,j,m,i] := r392[i,j,k] * x[m,k,n]  avx=false
    @test gradtest(con10b, (9,2,3))

    r3399 = randn(3,3,9,9);

    con13(x) = @tullio K[i,j] := r3399[s,s,j,k] * x[t,t,k,i]  avx=false  # https://github.com/mcabbott/Tullio.jl/pull/144
    @test gradtest(con13, (3,3,9,9))

    r33 = rand(3,3);
    con14(x) = @tullio K[i,j] := r3399[a,b,j,k] * x[b,c,k,i] * r33[a,c]  avx=false
    @test gradtest(con14, (3,3,9,9))

    @printline

    ## scalar -- one with :=, one without
    sc1(x) = @tullio s = r22[b,β] * x[a,b,c] * r312[c,a,β]  avx=false verbose=true
    @test gradtest(sc1, (1,2,3)) # UndefVarError: ####op#798_0 not defined

    @printline

    sc2(x) = @tullio s := x[γ,c] * r3399[c,γ,i,i]  avx=false verbose=true
    @test gradtest(sc2, (3,3))

end

@printline

if Tullio._GRAD[] != :Dual
#=
    @testset "products" begin

        p1(x) = @tullio (*) z = x[i]
        @test _gradient(p1, 1:4)[1] == ForwardDiff.gradient(p1, 1:4)
        @test _gradient(p1, -1:3)[1] == ForwardDiff.gradient(p1, -1:3) # one zero
        @test _gradient(p1, [1,0,2,0])[1] == ForwardDiff.gradient(p1, [1,0,2,0])

        p2(m,v) = @tullio (*) y[i] := (m[i,j] + 3*v[j])^2 # / sqrt(v[i])
        p2(m,v) = @tullio (*) y[i] := m[i,j] * v[j]
        m1 = rand(4,4) .+ 1
        v1 = rand(4) .+ 1
        dm = ForwardDiff.gradient(m -> sum(p2(m,v1)), m1)
        @test dm ≈ _gradient(sum∘p2, m1, v1)[1]
        dv = ForwardDiff.gradient(v -> sum(p2(m1,v)), v1)
        @test_broken dv ≈ _gradient(sum∘p2, m1, v1)[2]

        m1[2,3] = 0
        p3(m) = @tullio (*) y[i] := 4 * m[i,j]
        @test _gradient(sum∘p3, m1)[1] ≈ ForwardDiff.gradient(sum∘p3, m1)
        m1[3,4] = -1
        p4(m) = @tullio (*) y[i] := sin(1 + m[i,j])
        @test _gradient(sum∘p4, m1)[1] ≈ ForwardDiff.gradient(sum∘p4, m1)

    end
=#
    @testset "min/max" begin

        f1(x) = @tullio (max) z = x[i]
        f2(x) = @tullio (min) z = x[i] # avx=false

        @test _gradient(f1, 1:4)[1] == ForwardDiff.gradient(f1, 1:4)
        @test _gradient(f2, 1:4)[1] == ForwardDiff.gradient(f2, 1:4)

        @test _gradient(f1, [2,2,3,3])[1] in ([0,0,1,0], [0,0,0,1]) # changes with @avx
        ForwardDiff.gradient(f1, [2,2,3,3]) == [0,0,0,1] # different sub-gradient, OK
        @test _gradient(f2, [2,2,3,3])[1] == [1,0,0,0]

        m4 = reshape(shuffle(1:3*4*5*2), 3,4,5,2);
        m2 = reshape(shuffle(1:16), 4,4);
        v2 = shuffle(1:4)

        f3(x) = @tullio (max) y[i,k,l] := x[i,j,k,l]

        @test all(==(1), sum(_gradient(sum∘f3, m4)[1], dims=2))
        @test _gradient(sum∘f3, m4)[1] ≈ ForwardDiff.gradient(sum∘f3, m4)

        f4(x) = @tullio (min) y[j] := x[i,j,k,l]

        @test all(==(1), sum(_gradient(sum∘f4, m4)[1], dims=(1,3,4)))
        @test _gradient(sum∘f4, m4)[1] ≈ ForwardDiff.gradient(sum∘f4, m4)

        f5(x,y) = @tullio (max) z[i] := x[i,j] + 0.01*y[i]

        dm = ForwardDiff.gradient(m -> sum(f5(m,v2)), m2)
        @test dm ≈_gradient(sum∘f5, m2, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f5(m2,v)), v2)
        @test dv ≈_gradient(sum∘f5, m2, v2)[2]

        f6(x,y) = @tullio (max) z[i] := x[i,j] + 0.01*y[j] # max is now along y, not perp

        dm = ForwardDiff.gradient(m -> sum(f6(m,v2)), m2)
        @test dm ≈ _gradient(sum∘f6, m2, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f6(m2,v)), v2)
        @test dv ≈ _gradient(sum∘f6, m2, v2)[2]

        f7(x,y) = @tullio (max) z[i] := x[i,j]^2 / sqrt(y[i]) + exp(y[j])  avx=false

        dm = ForwardDiff.gradient(m -> sum(f7(m,v2)), m2)
        @test dm ≈ _gradient(sum∘f7, m2, v2)[1]  # avx: broken in tests, Julia 1.4
        dm .- _gradient(sum∘f7, m2, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f7(m2,v)), v2)
        @test dv ≈ _gradient(sum∘f7, m2, v2)[2]

        f8(x,y) = @tullio (max) z[i,l] := log(x[i,j,k,l]) / y[j]^1/3  avx=false
        f9(x,y) = @tullio (min) z[i,j] := log(x[i,j,k,l]) / y[j]^1/3  avx=false
        @tullio z89[i,j,k,l] := log(m4[i,j,k,l]) / v2[j]^1/3
        length(z89), length(unique(z89))

        dm = ForwardDiff.gradient(m -> sum(f8(m,v2)), m4)
        @test dm ≈ _gradient(sum∘f8, m4, v2)[1]  # avx: OK with 0.8, broken with 0.9
        dm .- _gradient(sum∘f8, m4, v2)[1]       # at exactly one element
        dv = ForwardDiff.gradient(v -> sum(f8(m4,v)), v2)
        @test dv ≈ _gradient(sum∘f8, m4, v2)[2]

        dm = ForwardDiff.gradient(m -> sum(f9(m,v2)), m4)
        @test dm ≈_gradient(sum∘f9, m4, v2)[1]  # avx: broken with 0.8 and 0.9
        dm .- _gradient(sum∘f9, m4, v2)[1]
        dv = ForwardDiff.gradient(v -> sum(f9(m4,v)), v2)
        @test dv ≈ _gradient(sum∘f9, m4, v2)[2]  # avx: broken with 0.8 and 0.9
        dv .- _gradient(sum∘f9, m4, v2)[2]       # but broken in different elements
        # I suspect that @avx is re-ordering loops, which makes onlyone() incorrect.

    end

    @printline

    @testset "finalisers" begin

        norm2(m) = @tullio n[i] := m[i,j]^2 |> sqrt

        gradtest(norm2, (3,4))
        mat = rand(3,3)
        @test _gradient(sum∘norm2, mat)[1] ≈ ForwardDiff.gradient(sum∘norm2, mat)
        @test gradtest(norm2, (3,4))

        layer(x) = @tullio y[i,k] := mat[i,j] * x[j,k] |> tanh  avx=false # this takes 15 mins +?
        @test gradtest(layer, (3,4))

        @printline

        lse1(mat) = @tullio lse[j] := log <| exp(mat[i,j])
        @test gradtest(lse1, (3,4))

        # relu(x) = max(x, zero(x))
        # lay2(x) = @tullio y[i,k] := mat[i,j] * x[j,k] |> relu

        @printline

        mx3(x) = @tullio (max) r[i] := x[i,j]^3 |> cbrt  avx=false # sometimes gets stuck here?
        mx3(mat) # hmmm what is this?
        _gradient(sum∘mx3, mat)[1] # zero

    end
end

@printline

if GRAD == :Zygote
    @testset "nograd keyword" begin

        f2(x,y) = @tullio out[i,j] := x[i] + y[j]  nograd=y threads=false
        @test _gradient(sum∘f2, rand(2), rand(2)) == ([2,2], nothing)

        f3(x,y,z) = @tullio out[i,j] := x[i] + y[j] * z[k]  nograd=(x,z) threads=false
        @test _gradient(sum∘f3, rand(2), rand(2), ones(2)) == (nothing, [4,4], nothing)

        f0(x,y) = @tullio out[i,j] := x[i]/y[j]  nograd=(y,x) threads=false
        @test _gradient(sum∘f0, rand(2), rand(2)) == (nothing, nothing)

    end
end

@printline
````

## File: test/group-1.jl
````
#===== stuff =====#

t2 = time()

@testset "parsing all the things" begin include("parsing.jl") end

@testset "tests from Einsum.jl" begin include("einsum.jl") end

@info @sprintf("Basic tests took %.1f seconds", time()-t2)

@testset "internal pieces" begin include("utils.jl") end

@testset "matrix multiplication" begin
    # size 200 is big enough to test block_halves even with MINIBLOCK = 64^3
    @testset "size $N, elements $T" for N in [2, 20, 200], T in [1:99, Float32, Float64, ComplexF64]
        for f in [identity, adjoint]
            A = f(rand(T, N,N));
            B = f(rand(T, N,N));
            @test A * B ≈ @tullio C[i,k] := A[i,j] * B[j,k]
        end
        if N < 200
            X = rand(T, N,N+1);
            Y = rand(T, N+1,N+2);
            Z = rand(T, N+2,N+1);
            @test X * Y * Z ≈ @tullio C[a,d] := X[a,b] * Y[b,c] * Z[c,d]
        end
    end
    @testset "@allocated" begin
        m!(C,A,B) = @tullio C[i,k] = A[i,j] * B[j,k] threads=false
        C1, A1, B1 = rand(4,4), rand(4,4), rand(4,4)
        @allocated m!(C1, A1, B1)
        @test 0 == @allocated m!(C1, A1, B1)
    end
end

#===== Tracker =====#

t3 = time()
using Tracker

GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@tullio grad=Base
@testset "gradients: Tracker + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Tracker + ForwardDiff" begin include("gradients.jl") end

@info @sprintf("Tracker tests took %.1f seconds", time()-t3)
````

## File: test/group-2.jl
````
#===== KernelAbstractions =====#

t4 = time()
using KernelAbstractions

using Tracker

GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@testset "KernelAbstractions + gradients" begin
    A = (rand(3,4));
    B = (rand(4,5));
    @tullio C[i,k] := A[i,j] * B[j,k]  threads=false  # verbose=2
    @test C ≈ A * B

    @tullio threads=false # else KernelAbstractions CPU kernels not used
    include("gradients.jl")
    @tullio threads=true

    for sy in Tullio.SYMBOLS
        @test !isdefined(@__MODULE__, sy)
    end
end

using CUDA

if is_buildkite
    # If we are on Buildkite, we should assert that we have a CUDA GPU available
    @test CUDA.has_cuda_gpu()
end

if CUDA.has_cuda_gpu()
    @info "===== found a GPU, starting CUDA tests ====="
    @testset "===== CUDA tests on GPU =====" begin
        include("cuda.jl")
    end
end

@info @sprintf("KernelAbstractions tests took %.1f seconds", time()-t4)

@tullio cuda=false
````

## File: test/group-3.jl
````
#===== Zygote =====#

t5 = time()
using Zygote
# patch for https://github.com/FluxML/Zygote.jl/issues/897
@eval Zygote begin
   function _pullback(cx::AContext, ::typeof(sum), f, xs::AbstractArray)
      y, back = pullback(((f, xs) -> sum(f.(xs))), cx, f, xs)
      y, ȳ -> (nothing, back(ȳ)...)
   end
end


GRAD = :Zygote
_gradient(x...) = Zygote.gradient(x...)

@tullio grad=Base
@testset "gradients: Zygote + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Zygote + ForwardDiff" begin include("gradients.jl") end

@tullio grad=Base
@testset "complex gradients with Zygote" begin

    x0 = [1,2,3] .+ [5im, 0, -11im]
    # y0 = rand(Int8,3) .+ im .* rand(Int8,3) .+ 0.0
    @testset "analytic" begin

        g1 = _gradient(x -> real(sum(x)), x0)[1]
        g1i = _gradient(x -> imag(sum(x)), x0)[1]
        @test g1 ≈ _gradient(x -> real(@tullio y := x[i]), x0)[1]
        @test g1i ≈ _gradient(x -> imag(@tullio y := x[i]), x0)[1]

        g2 = _gradient(x -> real(sum(exp, x)), x0)[1]
        g2i = _gradient(x -> imag(sum(exp, x)), x0)[1]
        @test g2 ≈ _gradient(x -> real(@tullio y := exp(x[i])), x0)[1]
        @test_broken g2i ≈ _gradient(x -> imag(@tullio y := exp(x[i])), x0)[1]

        g3 = _gradient(x -> real(sum(1 ./ (x.+im).^2)), x0)[1]
        g3i = _gradient(x -> imag(sum(1 ./ (x.+im).^2)), x0)[1]
        @test g3 ≈ _gradient(x -> real(@tullio y := 1/(x[i] + im)^2), x0)[1]
        @test g3 ≈ _gradient(x -> real(@tullio y := inv(x[i] + im)^2), x0)[1]
        @test g3i ≈ _gradient(x -> imag(@tullio y := 1/(x[i] + im)^2), x0)[1]
        @test g3i ≈ _gradient(x -> imag(@tullio y := inv(x[i] + im)^2), x0)[1]

        # with finaliser
        g7 = _gradient(x -> real(sum(sqrt.(sum(exp.(x), dims=2)))), x0 .+ x0')[1]
        g7i = _gradient(x -> imag(sum(sqrt.(sum(exp.(x), dims=2)))), x0 .+ x0')[1]
        @test_skip g7 ≈ _gradient(x -> real(sum(@tullio y[i] := sqrt <| exp(x[i,j]) )), x0 .+ 
x0')[1]
        @test_skip g7i ≈ _gradient(x -> imag(sum(@tullio y[i] := sqrt <| exp(x[i,j]) )), x0 
.+ x0')[1]

    end
    @testset "non-analytic" begin

        g4 = _gradient(x -> real(sum(x * x')), x0)[1]
        g4i = _gradient(x -> imag(sum(x * x')), x0)[1] # zero!
        @test_broken g4 ≈ _gradient(x -> real(@tullio y := x[i] * conj(x[j])), x0)[1]
        @test_broken g4i ≈ _gradient(x -> imag(@tullio y := x[i] * conj(x[j])), x0)[1]
        @test_broken g4 ≈ _gradient(x -> real(@tullio y := x[i] * adjoint(x[j])), x0)[1]
        @test_broken g4i ≈ _gradient(x -> imag(@tullio y := x[i] * adjoint(x[j])), x0)[1]

        g5 = _gradient(x -> real(sum(abs2.(x .+ 2 .+ im))), x0)[1]
        g5i = _gradient(x -> imag(sum(abs2.(x .+ 2 .+ im))), x0)[1] # zero!
        @test_broken g5 ≈ _gradient(x -> real(@tullio y := abs2(x[i] + 2 + im)), x0)[1]
        @test_broken g5i ≈ _gradient(x -> real(@tullio y := abs2(x[i] + 2 + im)), x0)[1]

        g6 = _gradient(x -> real(sum(abs.(x.^3))), x0)[1]
        g6i = _gradient(x -> imag(sum(abs.(x.^3))), x0)[1] # zero!
        @test_broken g6 ≈ _gradient(x -> real(@tullio y := abs(x[i]^3)), x0)[1]
        @test_broken g6i ≈ _gradient(x -> real(@tullio y := abs(x[i]^3)), x0)[1]

    end
end

@info @sprintf("Zygote tests took %.1f seconds", time()-t5)

#===== ReverseDiff =====#
#=
t6 = time()
using ReverseDiff

GRAD = :ReverseDiff
_gradient(x...) = ReverseDiff.gradient(x...) # ??

@tullio grad=Base
@testset "gradients: ReverseDiff + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: ReverseDiff + ForwardDiff" begin include("gradients.jl") end

@info @sprintf("ReverseDiff tests took %.1f seconds", time()-t6)
=#

#===== Yota =====#
#=
t7 = time()
using Yota

GRAD = :Yota
_gradient(x...) = Yota.grad(x...)[2]

@tullio grad=Base
@testset "gradients: Yota + DiffRules" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Yota + ForwardDiff" begin include("gradients.jl") end

@info @sprintf("Yota tests took %.1f seconds", time()-t7)
=#

#===== LoopVectorization =====#

if VERSION < v"1.11-"  # LV does not support 1.11

t8 = time()
using LoopVectorization
using VectorizationBase

#=
if isdefined(VectorizationBase, :SVec) # LV version 0.8, for Julia <= 1.5
    using VectorizationBase: SVec, Mask
else # LV version 0.9, supports Julia >= 1.5
    using VectorizationBase: Vec, Mask
    SVec{N,T} = Vec{N,T}
end

@testset "LoopVectorization onlyone" begin
    ms = Mask{4,UInt8}(0x03) # Mask{4,Bool}<1, 1, 0, 0>
    sv = SVec{4,Int}(1,2,3,4) # SVec{4,Int64}<1, 2, 3, 4>

    # preliminaries:
    @test Tullio.allzero(sv) === false
    @test Tullio.allzero(zero(sv)) === true

    @test Tullio.anyone(ms) === true

    # the main function:
    @test Tullio.onlyone(false, 0) === false
    @test Tullio.onlyone(true, 0) === true
    @test Tullio.onlyone(true, 1) === false

    @test Tullio.onlyone(ms, 0) === Mask{4}(0x02)
    # @test Tullio.onlyone(ms, 0).u == 0x02
    @test Tullio.onlyone(ms, sv) === Mask{4}(0x00)
    # @test Tullio.onlyone(ms, sv).u == 0x00
    @test Tullio.onlyone(ms, zero(sv)) === Mask{4}(0x02)
    # @test Tullio.onlyone(ms, zero(sv)).u == 0x02
end
=#

@testset "parsing + LoopVectorization" begin include("parsing.jl") end

using Tracker
GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@tullio grad=Base
@testset "gradients: Tracker + DiffRules + LoopVectorization" begin include("gradients.jl") end

@tullio grad=Dual
@testset "gradients: Tracker + ForwardDiff + LoopVectorization" begin include("gradients.jl") end

@info @sprintf("LoopVectorization tests took %.1f seconds", time()-t8)

@tullio avx=false

end  # if VERSION...

#===== TensorOperations =====#

t9 = time()
using TensorOperations

using Tracker
GRAD = :Tracker
_gradient(x...) = Tracker.gradient(x...)

@testset "gradients: Tracker + TensorOperations" begin include("tensorgrad.jl") end

using Zygote
GRAD = :Zygote
_gradient(x...) = Zygote.gradient(x...)

@testset "gradients: Zygote + TensorOperations" begin include("tensorgrad.jl") end

@testset "complex gradients with TensorOperations" begin

    x0 = [1 2; 3 4] .+ [5im 0; 7im -8im]

    @testset "analytic" begin

        g1 = _gradient(x -> real(sum(x * x)), x0)[1]
        g1i = _gradient(x -> imag(sum(x * x)), x0)[1]
        @test g1 ≈ _gradient(x -> real(sum(Tullio.@tensor y[i,j] := x[i,k] * x[k,j])), x0)[1]
        @test g1i ≈ _gradient(x -> imag(sum(Tullio.@tensor y[i,j] := x[i,k] * x[k,j])), x0)[1]

    end
    #=  # conj isn't handled by gradient code for @tensor here
    @testset "non-analytic" begin

        g2 = _gradient(x -> real(sum(x * x')), x0)[1]
        g2i = _gradient(x -> imag(sum(x * x')), x0)[1] # zero
        @test_broken g2 ≈ _gradient(x -> real(sum(Tullio.@tensor y[i,j] := x[i,k] * conj(x[j,k]))), x0)[1]
        @test_broken g2i ≈ _gradient(x -> imag(sum(Tullio.@tensor y[i,j] := x[i,k] * conj(x[j,k]))), x0)[1]

    end
    =#
end

@info @sprintf("TensorOperations tests took %.1f seconds", time()-t9)

#===== done! =====#
````

## File: test/parsing.jl
````
using Tullio, Test, LinearAlgebra

@testset "new arrays" begin

    # functions
    @tullio A[i] := (1:10)[i]^2
    @test A == [i^2 for i in 1:10]

    @tullio A[i] := (1:10)[i] * i
    @test A == [i^2 for i in 1:10]

    # diagonals
    @tullio D[i,i] := trunc(Int, sqrt(A[i]))
    @test D == Diagonal(sqrt.(A))

    # arrays of arrays
    C = [fill(i,3) for i=1:5]
    @tullio M[i,j] := C[i][j]
    @test M == (1:5) .* [1 1 1]

    # fields
    E = [(a=i, b=i^2, c=[i,2i,3i]) for i in 1:10]
    @tullio O[i] := A[i]//E[i].b # avx disabled by try/catch
    @test O == ones(10)

    @tullio F[i,j] := E[i].c[j]
    @test F == (1:10) .* [1 2 3]

    # arrays of tuples
    Y = [(i,i^2,i^3) for i in 1:10]
    @tullio W[i,j] := Y[i][j]
    @test W[9,3] == 9^3

    # linear indexing
    @tullio V[i] := W[i]^2
    @test V == vec(W).^2

    # scalar
    @tullio S := A[i]/2
    @tullio S′ = A[i]/2 # here = is equivalent
    @test S ≈ S′ ≈ sum(A)/2

    # almost scalar
    @tullio Z[] := A[i] + A[j]  avx=false
    @test Z isa Array{Int,0}
    @tullio Z′[1,1] := A[i] + A[j]  avx=false
    @test size(Z′) == (1,1)
    @tullio Z′′[_] := A[i] + A[j]  avx=false
    @test size(Z′′) == (1,)
    @test Z[] == Z′[1,1] == Z′′[1] == sum(A .+ A')

    # scalar update
    @tullio S += A[i]/2
    @test S ≈ sum(A)

    # fixed
    @tullio F[i] := D[i,5]
    @test F[5] == 5

    j = 6
    @tullio G[i] := D[i,$j]
    @test G[6] == 6

    @test_throws LoadError @eval @tullio D[i,$j] := A[i]

    @tullio H[i] := D[i,:]
    @test H[5] == F

    # trivial dimensions
    @tullio J[1,1,i] := A[i]
    @test size(J) == (1,1,10)

    @tullio J[_,i] := A[i]
    @test J == A'

    # non-unique arrays
    @tullio A2[i] := A[i] + A[i]
    @test A2 == 2 .* A

    # broadcasting
    @tullio S[i] := sqrt.(M[:,i]) 

    # scope
    f(x,k) = @tullio y[i] := x[i] + i + $k
    @test f(ones(3),j) == 1 .+ (1:3) .+ j

    g(x) = @tullio y := sqrt(x[i])  avx=false
    @test g(fill(4,5)) == 10

    # ranges
    @tullio K[i] := i^2  (i ∈ 1:3)
    @test K == (1:3).^2
    @test axes(K,1) === Base.OneTo(3) # literal 1:3

    @tullio N[i,j] := A[i]/j  (j in axes(K,1))  (i in axes(A,1)) # K not an argument
    @test N ≈ A ./ (1:3)'

    @test_throws String @tullio A[i] := i^2 (i in 1+10) # not a range

    # repeated scalar arg
    tri = Base.OneTo(3) # with 1:3, this fails without OffsetArrays,
    # as it only converts shifted indices to OneTo
    @tullio M[i,j] := (r=i, c=j)  (i in tri, j in tri)
    @test M[3,3] == (r=3, c=3)

    # indexing by an array, "gather"...
    J = repeat(1:3, 4);
    @tullio G[i,k] := M[i,J[k]]
    @test G[3,1] == G[3,4] == G[3,7]

    inds = vcat(1:3, 1:3)
    @tullio AI[i] := A[inds[i]]
    @test AI == A[inds]
    jnds = -5:5
    @test_throws String @tullio AJ[j] := A[jnds[j]]
    @test_throws BoundsError A[jnds]
    knds = 1:3.0
    @test_throws String @tullio AK[j] := A[knds[j]]
    @test_throws ArgumentError A[knds]

    # ... and "scatter"
    M = rand(1:99, 4,5)
    J = [3,1,2,3]
    @tullio H[J[i],k] := M[i,k] # i is not marked unsafe, may be threaded
    @test size(H) == (3,5)
    @test H[1,:] == M[2,:] # but H[3,:] gets written into twice.

    J′ = [1,2,10]
    @tullio H′[J′[i'],k] := A[k]  avx=false # new failure LoopVectorization v0.12.13? only on CI?
    @test size(H′) == (10, length(A))
    @test H′[2,:] == A
    @test H′[3,4] == 0 # zeroed before being written into

    inds = vcat(1:3, 1:3)
    @test_throws String @tullio H[inds[i],k] := M[i,k] # range of index i

    # masking
    @tullio M[i,j] := A[i] * A[j] * (i<=j)
    @test M == UpperTriangular(A .* A')

    # primes
    @test A == @tullio P[i′] := A[i']
    @test A == @tullio P[i'] := A[i′]
    @test [1,4,9] == @tullio Q[i'] := (i′)^2  (i' in 1:3)

    # non-numeric array
    @tullio Y[i] := (ind=i, val=A[i])
    @test Y[2] === (ind = 2, val = 4)

    # no name given
    Z = @tullio _[i] := A[i] + 1
    @test Z == A .+ 1

    # multi-line
    @tullio B[i,j] := begin
        x = (1:10)[i] + 3
        y = (1:3)[j]
        x // y
    end
    @test B == (4:13) .// (1:3)'

    # internal name leaks
    for sy in Tullio.SYMBOLS
        @test !isdefined(@__MODULE__, sy)
    end

end

@printline

@testset "in-place" begin

    A = [i^2 for i in 1:10]
    D = similar(A, 10, 10)

    @tullio D[i,j] = A[i] + 100
    @test D[3,7] == A[3] + 100

    # sum and +=
    B = copy(A);
    D .= 3;
    @tullio B[i] += D[i,j]
    @test B[1] == A[1] + 30

    # writing back into same
    B = copy(A)
    @tullio B[i] += B[i] + 10^3
    @test B[6] == 2 * A[6] + 10^3

    @tullio A[i] = A[i] + 100
    @test A[1] == 101

    # indices in expression
    @tullio A[i] = 100*i
    @test A[7] == 700

    # fixed on left
    j = 3
    D .= 3;
    @tullio D[$j,i] = 99
    @test D[j,j] == 99
    @test D[1,1] != 0
    @tullio D[i,end] = 100*A[i]  avx=false
    @test D[2,end] == 100*A[2]
    @tullio D[i,end-3] = 1000*A[i]  avx=false
    @test D[2,end-3] == 1000*A[2]

    # diagonal & ==, from https://github.com/ahwillia/Einsum.jl/pull/14
    B = [1 2 3; 4 5 6; 7 8 9]
    @tullio W[i, j, i, n] := B[n, j]  i in 1:2
    @test size(W) == (2,3,2,3)
    @test W[1,2,1,3] == B[3,2]
    @test W[1,1,2,2] == 0

    W2 = zero(W);
    @tullio W2[i, j, m, n] = (i == m) * B[n, j]
    @test W2 == W

    @test_throws LoadError @eval @tullio [i,j] = A[i] + 100
    @test_throws LoadError @eval @tullio _[i,j] = A[i] + 100

    # zero off-diagonal? no.
    @tullio D[i,i] = A[i]
    @test D[1,3] != 0

    # scatter operation
    D = similar(A, 10, 10) .= 999
    inds = [2,3,5,2]
    @tullio D[inds[i],j] = A[j]
    @test D[2,:] == A
    @test D[4,4] != 0 # not zeroed before writing.

    @tullio D[inds[i],j] += A[j]
    @test D[2,:] == 3 .* A # was not re-zeroed for +=

    kinds = [1,2,13,4]
    @test_throws String @tullio D[kinds[i],j] = A[j] # BoundsError needs to know which array

    # assignment: no loop over j
    B = zero(A);
    @tullio B[i] = begin
        j = mod(i^4, 1:10)
        A[j]
    end
    @test_skip B == A[[mod(i^4, 1:10) for i in 1:10]]
    # on travis 1.3 multi-threaded, B == [500, 600, 100, 600, 500, 600, 100, 600, 100, 1000]
    # and on 1.4 multi-threaded,    B == [100, 600, 100, 600, 100, 600, 100, 600, 100, 1000]

    # wrong ndims
    @test ndims(B)==1 && ndims(D)==2
    @test_throws Any @tullio B[i] = D[i]^2
    @test_throws Any @tullio D[i] = B[i]+2
    @test_throws Any @tullio B[i,j] = D[i,j]

    # internal name leaks
    for sy in Tullio.SYMBOLS
        @test !isdefined(@__MODULE__, sy)
    end

end

@printline

if !@isdefined OffsetArray
    @testset "without packages" begin

        A = [i^2 for i in 1:10]

        # without OffsetArrays
        @test axes(@tullio B[i] := A[2i+1] + A[i]) === (Base.OneTo(4),)
        @test_throws String @tullio C[i] := A[2i+5]

        J = [3,5,7] # doesn't start at 1
        @test_throws String @tullio G[J[i],k] := A[k]

        # without NamedDims
        @test_throws UndefVarError @tullio M[row=i, col=j, i=1] := (1:3)[i] // (1:7)[j]

    end
end

using OffsetArrays

@testset "index shifts" begin

    A = [i^2 for i in 1:10]

    @tullio L[i,j] := A[i]//j  (j ∈ 2:3, i in 1:10) # no shift, just needs OffsetArrays
    @test axes(L) == (1:10, 2:3)

    # shifts
    @tullio B[i] := A[2i+1] + A[i]
    @test axes(B,1) == 1:4 # would be OneTo(4) without OffsetArrays

    @tullio C[i] := A[2i+5]
    @test axes(C,1) == -2:2 # error without OffsetArrays

    j = 7 # interpolation
    @tullio C[i] := A[2i+$j]
    @test axes(C,1) == -3:1

    # end can appear in range inference
    @tullio C[i] := A[end-2i]  avx=false
    @test axes(C,1) == 0:4

    @tullio C[i] := A[end-2begin-i]  avx=false
    @test parent(C) == [A[end-2begin-i] for i in -2:7]

    cee(A) = @tullio C[i] := A[2i+$j] # closure over j
    @test axes(cee(A),1) == -3:1

    @test_throws String @tullio D[i] := A[i] + B[i]
    @tullio D[i] := A[i] + B[i+0] # switches to intersection
    @test axes(D,1) == 1:4

    @test_throws String @tullio M[i,j] := A[i+0]/A[j]  (i ∈ 2:5, j ∈ 2:5) # intersection for i but not j

    @tullio L[i] := A[i+j+1]  (j ∈ -1:1)
    @test axes(L,1) == 1:8

    # negative
    @test eachindex(@tullio F[i] := A[-1i]) == -10:-1
    @test eachindex(@tullio F[i] := A[-i]) == -10:-1
    @test eachindex(@tullio F[i] := A[-i+0]) == -10:-1
    @test eachindex(@tullio F[i] := A[0-i]) == -10:-1

    # non-constant
    @test axes(@tullio I[i,j] := A[i+j] + 0 * B[j]) == (0:6, 1:4)
    @test axes(@tullio I[i,j] := A[j+i+0] + 0 * B[j]) == (0:6, 1:4)
    @test axes(@tullio I[i,j] := A[(j+i)*1] + 0 * B[j]) == (0:6, 1:4)
    @test axes(@tullio I[i,j] := A[2i+j] + 0 * B[j]) == (0:3, 1:4)
    @test axes(@tullio I[i,j] := A[1j+2i] + 0 * B[j]) == (0:3, 1:4)
    @test axes(@tullio I[i,j] := A[i+2j] + 0 * B[j]) == (-1:2, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2j] + 0 * B[j]) == (0:1, 1:4)
    @test axes(@tullio I[i,j] := A[2(i+j)] + 0 * B[j]) == (0:1, 1:4)
    @test axes(@tullio I[i,j] := A[2i-1+2j] + 0 * B[j]) == (0:1, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2j+5] + 0 * B[j]) == (-3:-2, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2j-5] + 0 * B[j]) == (2:3, 1:4)
    @test axes(@tullio I[i,j] := A[2i+2(j-2)-1] + 0 * B[j]) == (2:3, 1:4)
    @test axes(@tullio I[i,j] := A[2(0+i)+(2j-4)-1] + 0 * B[j]) == (2:3, 1:4)

    @test axes(@tullio J[i,j] := A[i-j] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-(-i+j)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-(j-i)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-1*(j-i)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[i+(-j)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-j+i] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-1j+i] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[-1j-(-i)] + 0 * B[j]) == (5:11, 1:4)
    @test axes(@tullio J[i,j] := A[i-2j] + 0 * B[j]) == (9:12, 1:4)
    @test axes(@tullio J[i,j] := A[-2j+i] + 0 * B[j]) == (9:12, 1:4)
    @test axes(@tullio J[i,j] := A[2i-2j] + 0 * B[j]) == (5:6, 1:4)

    @test_throws LoadError @eval @tullio I[i,j] := A[i+j] # under-specified

    # in-place
    E = zero(A)
    @tullio E[i] = A[i+5] + 100
    @test E == vcat(A[6:end] .+ 100, zeros(Int,5))

    M = fill(pi/2, 10, 10)
    @tullio M[i,i] = A[i-2]
    @test M[3,3] == A[1]
    @test M[1,1] == pi/2 # was not set to zero

    # shifts on left
    E = zero(A)
    @tullio E[2i+1] = A[i]  avx=false  # new failure LoopVectorization v0.12.14? only on CI?
    @test E[2+1] == A[1]
    @test E[2*4+1] == A[4]

    # non-constant
    @tullio I[i,j] := 0 * A[i+j] + 0 * B[j]
    @test axes(@tullio I[i,j] = A[i+j] + B[j]) == (0:6, 1:4) # over-specified
    @test axes(@tullio I[i,j] = A[i+j]) == (0:6, 1:4) # needs range from LHS

    # linear indexing
    @tullio L[i] := I[i] + 1 # I is an offset matrix
    @test L == vec(I) .+ 1

    V = OffsetArray([1,10,100,1000],2) # offset vector
    @test axes(@tullio _[i] := log10(V[i]) avx=false) == (3:6,) # https://github.com/JuliaSIMD/LoopVectorization.jl/issues/249

    # indexing by an array
    @tullio W[i] := I[end-i+1]  avx=false # does not use lastindex(I,1)
    @test W == reverse(vec(I))

    # indexing by an array: gather
    inds = [-1,0,0,0,1]
    @tullio K[i,j] := A[inds[i]+j]
    @test K[2,3] == K[3,3] == K[4,3]
    @test axes(K) == (1:5, 2:9)

    @tullio K2[i,j] := A[j+2inds[i]]
    @test axes(K2) == (1:5, 3:8)

    j = 7
    @test_skip @tullio K3[i,j] := A[j+2inds[i]+$j]
    @test_broken vec(K2) == vec(K3)

    # scatter with shift not allowed
    @test_throws LoadError @eval @tullio G[inds[i]+1, j] := A[j]
    @test_throws LoadError @eval @tullio G[2inds[i], j] := A[j]

    # multiplication not implemented
    @test_throws LoadError @eval @tullio C[i] = A[i*j] + A[i]
    @test_throws LoadError @eval @tullio C[i] = A[i⊗j] + A[i]
    @test_throws LoadError @eval @tullio C[i] = A[(i,j)] + A[i]

    # magic shift
    @test axes(@tullio Z[i+_] := A[2i+10]) === (Base.OneTo(5),)

    @test_throws LoadError @eval @tullio Z[_+i] := A[2i+10] # wrong notation
    @test_throws LoadError @eval @tullio Z[J[i]+_] := A[2i+10] # with scatter
    @test_throws LoadError @eval @tullio Z[i+_] = A[2i+10] # in-place
end

@printline

@testset "modulo, clamped & padded" begin

    A = [i^2 for i in 1:10]
    B = 1:5

    @test vcat(B,B) == @tullio C[i] := B[mod(i)]  i in 1:10  avx=false
    @test vcat(B, fill(B[end],5)) == @tullio D[i] := min(A[i], B[clamp(i)])  avx=false  # UndefVarError: `#f###9###` not defined
    @test [4,16,36,64,100,4] == @tullio E[i] := A[mod(2i)]  i in 1:6  avx=false

    @test vcat(zeros(5), B, zeros(5)) == @tullio C[i] := B[pad(i-5,5)]  avx=false # no method matching _vload(::VectorizationBase.FastRange{Int64,
    @test vcat(zeros(2), A, zeros(3)) == @tullio D[i+_] := A[pad(i,2,3)]
    @test vcat(A, zeros(10)) == @tullio E[i] := A[pad(i)]  i in 1:20

    # pairconstraints
    @tullio F[i] := A[mod(i+k)] * ones(3)[k]  (i in axes(A,1))  avx=false
    @test F[end] == 1 + 2^2 + 3^2
    @tullio F[i] = A[clamp(i+k)] * ones(7)[k]  avx=false

    @tullio G[i] := A[pad(i+k, 4)] * ones(3)[k]  pad=100  avx=false # no method matching _vload(::VectorizationBase.StridedPointer{Int64, 1, 1, 0, ...
    @test axes(G,1) == -4:11
    @test G[-4] == G[11] == 300

    # matrix
    M = rand(Int8, 3,4)
    @tullio H[i+_,j+_] := M[pad(i,2), pad(j,3)]  pad=1  avx=false
    @test H == [trues(2,10); trues(3,3) M trues(3,3); trues(2,10)]

    # pad keyword
    @tullio J[i,i] := sqrt(A[i])  pad=-1
    @test J[3,4] == -1
    @test diag(J) == 1:10

    # unable to infer range
    @test_throws LoadError @eval @tullio F[i] := A[mod(i+1)]
    @test_throws LoadError @eval @tullio F[i] := A[pad(2i)]
    # can't use index mod(i) on LHS
    @test_throws LoadError @eval @tullio G[mod(i)] := A[i]
    # not sure what to do with clamp(i), sorry
    @test_throws LoadError @eval @tullio F[i] := A[clamp(i)+1]
    # eltype of pad doesn't fit
    @test_throws InexactError @tullio H[i] := A[pad(i,3)]  pad=im
    @test_throws InexactError @tullio J[i,i] := A[i]  pad=im
end

@printline

@testset "other reductions" begin

    A = [i^2 for i in 1:10]

    # basics
    @test [prod(A)] == @tullio (*) P[_] := float(A[i])
    @test maximum(A) == @tullio (max) m := float(A[i])
    @test minimum(A) == @tullio (min) m := float(A[i])

    @test true == @tullio (&) p := A[i] > 0
    @test true === @tullio (&) p := A[i] > 0
    @test true == @tullio (|) q := A[i] > 50  avx=false # zero_mask not defined

    # in-place
    C = copy(A)
    @test cumprod(A) == @tullio (*) C[k] = ifelse(i<=k, A[i], 1)
    @test cumprod(A).^2 == @tullio (*) C[k] *= i<=k ? A[i] : 1

    M = rand(1:9, 4,5)
    @test vec(prod(M,dims=2)) == @tullio (*) B[i] := M[i,j]

    # ^= generalises +=, *=
    C = copy(A)
    @tullio (max) C[i] ^= 5i
    @test C == max.(5:5:50, A)
    @test_throws LoadError @eval @tullio A[i] ^= A[i]
    @test_throws LoadError @eval @tullio (*) A[i] ^= A[i]

    # initialisation
    @test 200 == @tullio (max) m := A[i] init=200
    @tullio (max) C[i] := i^2   (i in 1:10, j in 1:1)  init=33.3 avx=false # widens type
    @test C == max.(33.3, A)
    @tullio C[i] := 0   (i in 1:10, j in 1:1)  init=randn()
    @test C == fill(C[1], 10)

    # more dimensions
    Q = rand(1:10^3, 4,5,6)
    @test vec(maximum(Q,dims=(2,3))) == @tullio (max) R[i] := Q[i,j,k]
    @test vec(minimum(Q,dims=(1,3))).+2 == @tullio (min) P[j] := Q[i,j,k]+2
    @test dropdims(maximum(Q, dims=2), dims=2) == @tullio (max) S[i,k] := Q[i,j,k]

    # indexing
    ind = vcat(1:3, 1:3)
    V = 1:6
    @tullio (*) Z[j] := M[ind[k],j] * exp(-V[k]) # product over k
    @test Z ≈ vec(prod(M[ind,:] .* exp.(.-V), dims=1))

    # scalar update ("plusequals" internally)
    s = 1.0
    @tullio (*) s *= float(A[i])
    @test s == prod(A)
    @tullio s *= float(A[i]) # works without specifying (*), is this a good idea?
    @test s == float(prod(A))^2

    @test_throws LoadError @eval @tullio s += (*) A[i] # should be *=
    @test_throws LoadError @eval @tullio s *= (max) A[i] # should be ^=

    # scalar + threading
    L = randn(100 * Tullio.TILE[]);
    @tullio (max) m := L[i]
    @test m == maximum(L)

    # ... with a weird init, result would be unpredictable, hence an error:
    @test_throws String @tullio s2 := A[i]^2 init=2 # at runtime
    @test sum(A.^2)+2 == @tullio s2 := A[i]^2 init=2 threads=false # is OK

    # promotion of init & += cases:
    B = rand(10)
    @test sum(B.^2)+2 ≈ @tullio s2 := B[i]^2 init=2 threads=false
    s3 = 3
    @test sum(B.^2)+3 ≈ @tullio s3 += B[i]^2
    s4 = 4im
    @test sum(B.^2)+4im ≈ @tullio s4 += B[i]^2

    # no reduction means no redfun, and no init:
    @test_throws LoadError @eval @tullio (max) A2[i] := A[i]^2
    @test_throws LoadError @eval @tullio A2[i] := A[i]^2 init=0.0

end

@printline

@testset "finalisers" begin

    A = [i^2 for i in 1:10]

    @tullio B[i,j] := A[i] + A[k] // A[j]

    @tullio B2[_,j] := (B[i,j] + B[j,i])^2 |> sqrt  avx=false # new failure LoopVectorization v0.12.14? only on CI?
    @test B2 ≈ mapslices(norm, B .+ B', dims=1)

    # trivial use, scalar output -- now forbidden
    @test_throws LoadError @eval @tullio n2 = A[i]^2 |> sqrt

    # trivial use, no reduction -- now forbidden
    @test_throws LoadError @eval @tullio A2[i] := A[i]^2 |> sqrt
    @test_throws LoadError @eval @tullio (*) A2[i] := A[i]^2 |> sqrt

    # larger size, to trigger threads & tiles
    C = randn(10^6) # > Tullio.BLOCK[]
    @tullio n2[_] := C[i]^2 |> sqrt  avx=false
    @test n2[1] ≈ norm(C,2)

    D = rand(1000, 1000) # > Tullio.TILE[]
    @tullio D2[_,j] := D[i,j]^2 |> sqrt  avx=false
    @test D2 ≈ mapslices(norm, D, dims=1)

    # functions with underscores
    @tullio n2′[] := A[i]^2 |> (_)^0.5
    @test n2′[] ≈ norm(A,2)

    @tullio (max) E[i] := float(B[i,j]) |> atan(_, A[i]) # i is not reduced over
    @test E ≈ vec(atan.(maximum(B, dims=2), A))

    j = 2
    @tullio G[i'] := float(B[i',j]) |> atan(_, B[i',$j])
    @test G ≈ vec(atan.(sum(B, dims=2), B[:,j]))

    @test_throws LoadError @eval @tullio F[i] := B[i,j] |> (_ / A[j]) # wrong index
    C = randn(10^6)
    @test_throws String @tullio F[i] := B[i,j] |> (_ / C[i]) # wrong length

end

@testset "named dimensions" begin

    using NamedDims

    # reading
    N = NamedDimsArray(rand(Int8,3,10), (:r, :c))

    @tullio A[i,j] := N[i, j] + 100 * (1:10)[j] avx=false # conversion to pointer not defined for NamedDimsArray
    @test A == N .+ 100 .* (1:10)'

    @tullio B[i] := N[r=i, c=1] avx=false
    @test B == N[:,1]

    @tullio C[j,i] := N[c=j, r=i] + 100 * (1:10)[j] avx=false
    @test_broken A == C'
    @test_broken dimnames(C) == (:_, :_) # bug in similar, upstream. Work-around removed in https://github.com/mcabbott/Tullio.jl/pull/159

    # writing
    @tullio M[row=i, col=j, i=1] := (1:3)[i] // (1:7)[j] avx=false
    @test dimnames(M) == (:row, :col, :i)

end

@printline

@testset "options" begin

    # keyword threads accepts false or a positive integer
    @tullio A[i] := (1:10)[i]^2  threads=false
    @tullio A[i] := (1:10)[i]^2  threads=2^2
    # when using KernelAbstractions, something leaks from the 1st leading 2nd to error
    block = 64
    @tullio A[i] := (1:10)[i]^2  threads=block # Symbol
    @test_throws LoadError @eval @tullio A[i] := (1:10)[i]^2  threads=:maybe

    # keyword verbose accepts values [true, false, 2, 3]
    @tullio A[i] := (1:10)[i]^2  verbose=1
    @tullio A[i] := (1:10)[i]^2  verbose=false
    @test_throws LoadError @eval @tullio A[i] := (1:10)[i]^2  verbose=4

    # keyword grad accepts values [false, Base, Dual]
    @tullio A[i] := (1:10)[i]^2  grad=false
    @tullio A[i] := (1:10)[i]^2  grad=Base
    @test_throws LoadError @eval @tullio A[i] := (1:10)[i]^2  grad=true

    # recognised keywords are [:threads, :verbose, :avx, :cuda, :grad]
    @test_throws LoadError @eval @tullio A[i] := (1:10)[i]^2  key=nothing

end

@testset "bugs" begin

    # https://github.com/mcabbott/Tullio.jl/issues/10
    arr = [1 2; 3 4]
    function f10(arr)
        @tullio res1 = arr[i, k] - arr[i - 1, k]
        @tullio res2 = arr[i, k] - arr[i, k + 1]
        return res1 + res2
    end
    @test f10(arr) == 2

    let
        B = rand(3,3)
        @tullio tot = B[i, k] - B[i - 1, k]
        @test_throws UndefVarError 𝒜𝒸𝓉! isa Function
    end

    # https://github.com/mcabbott/Tullio.jl/issues/35
    a = [1 2 3; 4 5 6];
    b = [10,20,30];
    @test sum(b) == @tullio s := b[a[1,i]]

    # https://github.com/mcabbott/Tullio.jl/issues/36
    # final type real, intermediate complex... not fixed yet!
    xs = randn(1000)
    @test_throws InexactError @tullio z[i] := exp(im * xs[i] - xs[j]) |> abs2  avx=false # TypeError with LV

    # https://github.com/mcabbott/Tullio.jl/issues/43
    P = rand(2,2,3); Diff = rand(3,3); n=4
    @test axes(
        @tullio dP[x,y,z] := Diff[a+2, b+2] * Diff[c+2, d+2] *
            P[mod(x+a+c), mod(y+b+d), z] * P[mod(x+a),mod(y+b),z] (a in -1:1,
            b in -1:1, c in -1:1, d in -1:1, z in 1:3, x in 1:n, y in 1:n)  avx=false  # UndefVarError: `#f###46###` not defined
        ) == (1:4, 1:4, 1:3)
    @test axes(
        @tullio out[x,y] := Diff[mod(x+(y+2z)),x] * Diff[y,clamp(x+1+2x+y)] z in 1:3  avx=false
        ) == (1:3, 1:3)
    # unable to infer range of index z
    @test_throws LoadError @eval @tullio out[x,y] := Diff[mod(x+(y+2z)),x] * Diff[x,y]

    # https://discourse.julialang.org/t/unexpected-tullio-behavior/49371/6
    x = [rand(3), rand(2)]
    @test_throws String @tullio y[i,j] := x[i][j]
    n = 1; a = ones(4) # should now throw "elements x[\$n] must be of uniform size"
    @test_throws String @tullio a[j] = x[$n][j+0]  j in 1:3

    xt = [(a=1, b=rand(3)), (a=1, b=rand(2))] # version with field access
    @test_throws String @tullio y[i,j] := xt[i].b[j]

    @tullio a[j] = $(x[n])[j+0]  j in 1:3 # interpolation of an expression
    @test a == vcat(x[n],1) # requires postwalk -> prewalk

    # https://github.com/mcabbott/Tullio.jl/issues/46
    @test 1:4 == let s = [1,2,3,4], ndims=maximum(s), axes=size, eachindex=lastindex
       @tullio x[i] := s[i]
    end
    @test_broken [3,7] == let s = [1 2;3 4], zero=one # left because of #50
       @tullio x[i] := s[i,j]  avx=false # Unexpected Pass with LV
    end

    # https://github.com/mcabbott/Tullio.jl/issues/119
    struct X{T} y::T end
    CNT = Ref(0)
    Base.getproperty(x::X, s::Symbol) = s === :y ? begin CNT[] += 1; getfield(x, :y) end : error("nope")
    x = X(rand(100))
    @test sum(x.y) ≈ @tullio _ := x.y[i]
    @test CNT[] == 2  # getproperty is done outside of loop

end

@printline
````

## File: test/runtests.jl
````
using Test, Printf
import Pkg

t1 = @elapsed using Tullio
@info @sprintf("Loading Tullio took %.1f seconds", t1)

is_buildkite = parse(Bool, get(ENV, "BUILDKITE", "false"))
if is_buildkite
    test_group = "2" # only run group 2 on the GPU servers
else
    test_group = get(ENV, "TULLIO_TEST_GROUP", "all")
end

@info "Testing flags" Threads.nthreads() test_group is_buildkite

if Threads.nthreads() > 1 # use threading even on small arrays
    Tullio.BLOCK[] = 32
    Tullio.TILE[] = 32
end

macro printline()  # useful in hunting for where tests get stuck
    file = split(string(__source__.file), "/")[end]
    printstyled("  ", file, ":", __source__.line, "\n", color=:light_black)
end

if test_group in ["all", "1"]
    @info "starting test group 1 (basics)"
    include("group-1.jl")
end

if test_group in ["all", "2"]
    @info "starting test group 2 (KernelAbstractions etc.)"
    include("group-2.jl")
end

if test_group in ["all", "3"]
    @info "starting test group 3 (Zygote, LV)"
    include("group-3.jl")
end
````

## File: test/tensorgrad.jl
````
using Tullio, Test, ForwardDiff
# using Tracker; _gradient(x...) = Tracker.gradient(x...); GRAD = :Tracker

function gradtest(f, dims)
    x = randn(dims...)
    grad = ForwardDiff.gradient(x -> sum(sin, f(x)), x)
    grad ≈ _gradient(x -> sum(sin, f(x)), x)[1]
end

@testset "from TensorTrace" begin
    # These can all be handled using TensorOperations

    triv1(x) = Tullio.@tensor A[i,j] := 2 * x[i,j]
    @test gradtest(triv1, (2,3))

    r32 = randn(3,2);
    r312 = randn(3,1,2);

    ## trace!
    tr1(x) = Tullio.@tensor T[k] := 22 * x[i,i,k]
    @test gradtest(tr1, (3,3,4))

    tr2(x) = Tullio.@tensor T[k] := 22 * x[i,i,k,j,j]
    @test gradtest(tr2, (3,3,4,7,7))

    ## contract! A
    con1(x) = Tullio.@tensor C[i,j] := 5 * x[i,k] * r32[k,j]
    @test gradtest(con1, (2,3))

    r22 = rand(2,2);

    con3(x) = Tullio.@tensor C[i,j,m,n] := x[i,j,k] * r312[k,m,n]
    @test gradtest(con3, (1,2,3))

    con4(x) = Tullio.@tensor C[i,m] := x[i,kk,k] * r312[k,m,kk]
    @test gradtest(con4, (1,2,3))

    con5(x) = Tullio.@tensor C[j,i,n,m] := 44 * x[i,j,k] * r312[k,m,n]
    @test gradtest(con5, (1,2,3))

    r392 = randn(3,9,2);
    con6(x) = Tullio.@tensor C[n,i,m,j] := x[i,j,k] * r392[k,m,n]
    @test gradtest(con6, (9,2,3))

    con7(x) = Tullio.@tensor C[m,n,j,i] := 44 * x[i,j,k] * r392[k,m,n]
    @test gradtest(con7, (9,2,3))

    ## contract! B
    con8b(x) = Tullio.@tensor K[i,j] := 5 * r32[i,k] * x[k,j]
    @test gradtest(con8b, (2,3))

    con9b(x) = Tullio.@tensor K[i,j,m,n] := r312[i,j,k] * x[m,k,n]
    @test gradtest(con9b, (1,2,3))

    con10b(x) = Tullio.@tensor K[n,j,m,i] := r392[i,j,k] * x[m,k,n]
    @test gradtest(con10b, (9,2,3))

    r3399 = randn(3,3,9,9);

    con13(x) = Tullio.@tensor K[i,j] := r3399[s,s,j,k] * x[t,t,k,i]
    @test gradtest(con13, (3,3,9,9))

    r33 = rand(3,3);
    con14(x) = Tullio.@tensor K[i,j] := r3399[a,b,j,k] * x[b,c,k,i] * r33[a,c]
    @test gradtest(con14, (3,3,9,9))

    ## scalar -- one with :=, one without
    sc1(x) = Tullio.@tensor s = r22[b,β] * x[a,b,c] * r312[c,a,β]
    @test gradtest(sc1, (1,2,3))

    sc2(x) = Tullio.@tensor s := x[γ,c] * r3399[c,γ,i,i]
    @test gradtest(sc2, (3,3))

end

@testset "errors" begin
    @test_throws LoadError @eval Tullio.@tensor C[k] := A[i,i,k] + B[k]  # two terms
    @test_throws LoadError @eval Tullio.@tensor B[k] := conj(A[k])  # functions
    @test_throws LoadError @eval Tullio.@tensor C[k] := A[i, i+k]  # not a contraction
end
````

## File: test/utils.jl
````
using Test

using Tullio: storage_type, promote_storage
using ForwardDiff, FillArrays

@testset "storage_type" begin

    @test storage_type(rand(2), rand(2,3)) == Array{Float64,N} where N
    @test storage_type(rand(2), rand(Float32, 2)) == Vector{Float64}
    @test storage_type(rand(2), rand(Float32, 2,2)) == Array{Float64,N} where N

    Base.promote_type(Matrix{Int}, Vector{Int}) == Array{Int64,N} where N
    Base.promote_type(Matrix{Int}, Matrix{Int32}) == Matrix{Int64}
    Base.promote_type(Matrix{Int}, Vector{Int32}) == Array # != Array{Int64,N} where N
    promote_storage(Matrix{Int}, Vector{Int32}) == Array{Int64,N} where N

    @test storage_type(rand(2), 1:3) == Vector{Float64}
    @test storage_type(rand(Int,2), 1:3) == Vector{Int}
    @test storage_type(1:3.0, 1:3) <: AbstractRange{Float64}

    @test storage_type(rand(2), fill(ForwardDiff.Dual(1,0),2)) == Vector{ForwardDiff.Dual{Nothing,Float64,1}}
    @test storage_type(rand(2), fill(ForwardDiff.Dual(1,0),2,3)) == Array{ForwardDiff.Dual{Nothing,Float64,1}}

    # special case, but is this a good idea?
    @test storage_type(rand(2), FillArrays.Fill(1.0, 2,2)) == Vector{Float64}
    @test storage_type(rand(2), FillArrays.Fill(true, 2,2)) == Vector{Float64}

end

using Tullio: range_expr_walk, divrange, minusrange, subranges, addranges

@testset "range_expr_walk" begin

    for r in [Base.OneTo(10), 0:10, 0:11, 0:12, -1:13]
        for (f, ex) in [
            # +
            (i -> i+1, :(i+1)),
            (i -> i+2, :(i+2)),
            (i -> 3+i, :(3+i)),
            # -
            (i -> -i, :(-i)),
            (i -> i-1, :(i-1)),
            (i -> 1-i, :(1-i)),
            (i -> 2-i, :(2-i)),
            (i -> 1+(-i), :(1+(-i))),
            (i -> -i+1, :(-i+1)),
            (i -> -i-1, :(-i-1)),
            (i -> 1-(2-i), :(1-(2-i))),
            (i -> 1-(-i+2), :(1-(-i+2))),
            # *
            (i -> 2i, :(2i)),
            (i -> 2i+1, :(2i+1)),
            (i -> -1+2i, :(-1+2i)),
            (i -> 1-3i, :(1-3i)),
            (i -> 1-3(i+4), :(1-3(i+4))),
            # triple...
            (i -> i+1+2, :(i+1+2)),
            (i -> 1+2+i, :(1+2+i)),
            (i -> 2i+3+4, :(2i+3+4)),
            (i -> 1+2+3i+4, :(1+2+3i+4)),
            (i -> 1+2+3+4(-i), :(1+2+3+4(-i))),
            # evil
            (i -> (2i+1)*3+4, :((2i+1)*3+4)),
            ]
            rex, i = range_expr_walk(:($r .+ 0), ex)
            @test issubset(sort(f.(eval(rex))), r)
        end

        rex, _ = range_expr_walk(:($r .+ 0), :(pad(i,2)))
        @test extrema(eval(rex)) == (first(r)-2, last(r)+2)
        rex, _ = range_expr_walk(:($r .+ 0), :(pad(i+1,2,5)))
        @test extrema(eval(rex)) == (first(r)-1-2, last(r)-1+5)

        @test range_expr_walk(:($r .+ 0), :(i+j))[2] == (:i, :j) # weak test!

        # range adjusting functions
        @test minusrange(r) == divrange(r, -1)

        @test issubset(subranges(r, 1:3) .+ 1, r)
        @test issubset(subranges(r, 1:3) .+ 3, r)
        @test union(subranges(r, 1:3) .+ 1, subranges(r, 1:3) .+ 3) == r

        @test issubset(addranges(r, 1:3) .- 1, r)
        @test issubset(addranges(r, 1:3) .- 3, r)
        @test sort(union(addranges(r, 1:3) .- 1, addranges(r, 1:3) .- 3)) == r
    end
end

using Tullio: cleave, trisect, productlength

@testset "threading" begin
    @test cleave((1:10, 1:4, 7:8)) == ((1:4, 1:4, 7:8), (5:10, 1:4, 7:8))
    @test cleave((7:8, 9:9)) == ((7:7, 9:9), (8:8, 9:9))
    @test cleave((1:4,)) == ((1:2,), (3:4,))
    @test cleave(()) == ((), ())

    @test trisect((1:9, 11:12)) == ((1:3, 11:12), (4:6, 11:12), (7:9, 11:12))
    @test trisect((1:9,)) == ((1:3,), (4:6,), (7:9,))
    @test trisect(()) == ((), (), ())

    @test sum(productlength, trisect((1:10, 11:20))) == 100

    for r1 in [1:10, 3:4, 5:5], r2 in [11:21, -3:-2, 0:0], r3 in [2:7, 8:9, 0:0]
        tup = (r1,r2,r3)
        len = productlength(tup)
        @test len == sum(productlength, cleave(tup))
        @test len == sum(productlength, trisect(tup))
    end
end

using Tullio: @capture_

@testset "capture_ macro" begin
    EXS  = [:(A[i,j,k]),  :(B{i,2,:}),  :(C.dee), :(fun(5)),   :(g := h+i),        :(k[3] += l[4]), :([m,n,0]) ]
    PATS = [:(A_[ijk__]), :(B_{ind__}), :(C_.d_), :(f_(arg_)), :(left_ := right_), :(a_ += b_),     :([emm__]) ]
    # @test length(EXS) == length(PATS)
    @testset "ex = $(EXS[i])" for i in eachindex(EXS)
        for j in eachindex(PATS)
        @eval res = @capture_($EXS[$i], $(PATS[j]))
        if i != j
            @test res == false
        else
            @test res == true
            if i==1
                @test A == :A
                @test ijk == [:i, :j, :k]
            elseif i==3
                @test C == :C
                @test d == :dee
            elseif i==5
                @test left == :g
                @test right == :(h+i)
            elseif i==7
                @test emm == [:m, :n, 0]
            end
        end
        end
    end

end

using Tullio: leibnitz
using ForwardDiff

@testset "symbolic gradients" begin

    @testset "ex = $ex" for ex in [
        :(x*y + 1/z),
        :(x*y + z*x*z^2),
        :((x/y)^2 + inv(z)),
        :((x+2y)^z),
        :(1/(x+1) + 33/(22y) -4/(z/4)),
        :(inv(x+y) + z^(-2)),

        :(sqrt(x) + 1/sqrt(y+2z)),
        :(inv(sqrt(x)*sqrt(y)) + sqrt(2*inv(z))),
        :(x * z / sqrt(y * z)),

        :(log(x/y) - log(z+2)),
        :(log(x*y*z) - 33y),

        :(2exp(x*y*z)),
        :(exp((x-y)^2/2)/z),
    ]

        dfdx = leibnitz(ex, :x)
        dfdy = leibnitz(ex, :y)
        dfdz = leibnitz(ex, :z)

        @eval f_(x,y,z) = $ex
        @eval f_x(x,y,z) = $dfdx
        @eval f_y(x,y,z) = $dfdy
        @eval f_z(x,y,z) = $dfdz

        xyz = rand(Float32, 3)

        # check correctness
        gx, gy, gz = ForwardDiff.gradient(xyz -> f_(xyz...), xyz)
        @test f_x(xyz...) ≈ gx
        @test f_y(xyz...) ≈ gy
        @test f_z(xyz...) ≈ gz

        # don't accidentally make Float64
        @test 0f0 + f_x(xyz...) isa Float32
        @test 0f0 + f_y(xyz...) isa Float32
        @test 0f0 + f_z(xyz...) isa Float32

    end

end

macro cse(ex)
    esc(Tullio.commonsubex(ex))
end

@testset "common subexpressionism" begin

    x,y,z = 1.2, 3.4, 5.6

    @test (x+y)*z/(x+y) ≈
        @cse (x+y)*z/(x+y)

    @test x*y*z + 2*x*y*z ≈
        @cse x*y*z + 2*x*y*z

    @test (sqrt(inv(x)) * inv(sqrt(y)) + inv(x)/inv(z)) ≈
        @cse (sqrt(inv(x)) * inv(sqrt(y)) + inv(x)/inv(z))

    @test (a1 = inv(x); b1 = inv(x); c1 = inv(x)*inv(y)) ≈
        @cse (a = inv(x); b = inv(x); c = inv(x)*inv(y))

    # setting a, b (outside @test)
    (a1 = inv(x); b1 = inv(x); c1 = inv(x)*inv(y))
    @cse (a = inv(x); b = inv(x); c = inv(x)*inv(y))

    @test a1 ≈ a
    @test b1 ≈ b
    @test c1 ≈ c

    # updating a, b, etc
    a = 1
    @cse a = a + x*y/z
    @test a ≈ 1 + x*y/z

    a = 1
    b = 1
    @cse begin
        a = a + x*y/z
        b += x*y + z
    end
    @test a ≈ 1 + x*y/z
    @test b ≈ 1 + x*y + z

    a = a1 = 1
    b = b1 = 1
    begin
        θ = inv(x)*inv(y+a)
        a1 += θ^2
        b1 = θ^3 + inv(x)
    end
    @cse begin
        θ = inv(x)*inv(y+a)
        a += θ^2
        b = θ^3 + inv(x)
    end
    @test a ≈ a1
    @test b ≈ b1

end
````

## File: .gitignore
````
Manifest.toml
````

## File: LICENSE
````
MIT License

Copyright (c) 2019-2020 Michael Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````

## File: Project.toml
````toml
name = "Tullio"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
authors = ["Michael Abbott"]
version = "0.3.8"

[deps]
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Requires = "ae029012-a4dd-5104-9daa-d747884805df"

[weakdeps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[extensions]
TullioCUDAExt = "CUDA"
TullioFillArraysExt = "FillArrays"
TullioTrackerExt = "Tracker"
TullioChainRulesCoreExt = "ChainRulesCore"

[compat]
CUDA = "4, 5"
ChainRulesCore = "1"
DiffRules = "1"
FillArrays = "0.11, 0.12, 0.13, 1"
ForwardDiff = "0.10, 1.0"
KernelAbstractions = "0.9"
LoopVectorization = "0.12.101"
NamedDims = "0.2, 1"
OffsetArrays = "1"
Requires = "1"
TensorOperations = "4, 5"
Tracker = "0.2"
VectorizationBase = "0.21.23"
Zygote = "0.6.33, 0.7"
julia = "1.10"  # note that this is the minimum Julia version, 1.11 & 1.12 etc. are allowed & should work.

[extras]
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
NamedDims = "356022a1-0364-5f58-8944-0da4b18d706f"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
TensorOperations = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
VectorizationBase = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[targets]
test = ["Test", "CUDA", "FillArrays", "ForwardDiff", "KernelAbstractions", "LinearAlgebra", "LoopVectorization", "NamedDims", "OffsetArrays", "Pkg", "Printf", "Random", "TensorOperations", "Tracker", "VectorizationBase", "Zygote"]
````

## File: README.md
````markdown
<div align="center">
<h1>Tullio.jl</h1>


[![GitHub 
CI](https://img.shields.io/github/actions/workflow/status/mcabbott/Tullio.jl/ci.yml?logo=github)](https://github.com/mcabbott/Tullio.jl/actions?query=workflow%3ACI)
[![Buildkite GPU CI](https://img.shields.io/buildkite/7f7fec35c774174a59cf616fc6e1711c70e94c088248088758?color=eee&label=gpu&logo=nvidia)](https://buildkite.com/julialang/tullio-dot-jl)
[![Tag Version](https://img.shields.io/github/v/tag/mcabbott/Tullio.jl?color=red&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMzI1cHQiIGhlaWdodD0iMzAwcHQiIHZpZXdCb3g9IjAgMCAzMjUgMzAwIiB2ZXJzaW9uPSIxLjEiPgo8ZyBpZD0ic3VyZmFjZTkxIj4KPHBhdGggc3R5bGU9IiBzdHJva2U6bm9uZTtmaWxsLXJ1bGU6bm9uemVybztmaWxsOnJnYig3OS42JSwyMy41JSwyMCUpO2ZpbGwtb3BhY2l0eToxOyIgZD0iTSAxNTAuODk4NDM4IDIyNSBDIDE1MC44OTg0MzggMjY2LjQyMTg3NSAxMTcuMzIwMzEyIDMwMCA3NS44OTg0MzggMzAwIEMgMzQuNDc2NTYyIDMwMCAwLjg5ODQzOCAyNjYuNDIxODc1IDAuODk4NDM4IDIyNSBDIDAuODk4NDM4IDE4My41NzgxMjUgMzQuNDc2NTYyIDE1MCA3NS44OTg0MzggMTUwIEMgMTE3LjMyMDMxMiAxNTAgMTUwLjg5ODQzOCAxODMuNTc4MTI1IDE1MC44OTg0MzggMjI1ICIvPgo8cGF0aCBzdHlsZT0iIHN0cm9rZTpub25lO2ZpbGwtcnVsZTpub256ZXJvO2ZpbGw6cmdiKDIyJSw1OS42JSwxNC45JSk7ZmlsbC1vcGFjaXR5OjE7IiBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSAiLz4KPHBhdGggc3R5bGU9IiBzdHJva2U6bm9uZTtmaWxsLXJ1bGU6bm9uemVybztmaWxsOnJnYig1OC40JSwzNC41JSw2OS44JSk7ZmlsbC1vcGFjaXR5OjE7IiBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1ICIvPgo8L2c+Cjwvc3ZnPgo=)](https://github.com/mcabbott/Tullio.jl/releases)
</div>

Tullio is a very flexible einsum macro. It understands many array operations written in index notation -- not just matrix multiplication and permutations, but also convolutions, stencils, scatter/gather, and broadcasting. For example:

```julia
@tullio M[x,y,c] := N[x+i, y+j,c] * K[i,j]     # sum over i,j, and create M

@tullio S[x] = P[x,y] * log(Q[x,y] / R[y])     # sum over y, and write into S

@tullio A[i,j] += B[i,k,l] * C[l,j] * D[k,j]   # sum over k,l, and add to values in A

@tullio (*) Z[j] := X[ind[k],j] * exp(-Y[k])   # product over k
```

Used by itself the macro writes ordinary nested loops much like [`Einsum.@einsum`](https://github.com/ahwillia/Einsum.jl).
One difference is that it can parse more expressions, and infer ranges for their indices.
Another is that it will use multi-threading (via [`Threads.@spawn`](https://julialang.org/blog/2019/07/multithreading/)) and recursive tiling, on large enough arrays.
But it also co-operates with various other packages, provided they are loaded before the macro is called:

* It uses [`LoopVectorization.@avx`](https://github.com/chriselrod/LoopVectorization.jl) to speed many things up. (Disable with keyword `avx=false`.) On a good day this will match the speed of OpenBLAS for matrix multiplication.

* It uses [`KernelAbstractions.@kernel`](https://github.com/JuliaGPU/KernelAbstractions.jl) to make a GPU version. (Disable with `cuda=false`.) This is somewhat experimental, and may not be fast.

The macro also tries to provide a gradient for use with [Tracker](https://github.com/FluxML/Tracker.jl) or (via  [ChainRules](https://github.com/JuliaDiff/ChainRules.jl)) for [Zygote](https://github.com/FluxML/Zygote.jl), [Yota](https://github.com/dfdx/Yota.jl), etc. <!-- or [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl). -->
(Disable with `grad=false`, or `nograd=A`.) This is done in one of two ways:

* By default it takes a symbolic derivative of the right hand side expression. This works for reductions over `+` or `min`/`max`. The functions as typed must be known, mostly from [DiffRules](https://github.com/JuliaDiff/DiffRules.jl). 

* The option `grad=Dual` uses instead [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) to differentiate the right hand side (only for reductions over `+`). This allows for more complicated expressions.

The entire right hand side is summed over the full possible range of any indices not appearing on the left.
Pipe operators `|>` or `<|` indicate functions to be performed *outside* the sum, for example:

```julia
@tullio lse[j] := log <| exp(mat[i,j])   # vec(log.(sum(exp.(mat), dims=1)))
```

The option `@tullio verbose=true` will cause it to print index ranges, symbolic derivatives,
and notices when it is unable to use the packages mentioned above.
And `verbose=2` will print everything.

If it's useful in academic work, you can cite it with this DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5047409.svg)](https://doi.org/10.5281/zenodo.5047409)

## Notation
<details>

Index notation for some simple functions:

```julia
using Pkg; Pkg.add("Tullio")
using Tullio, Test
M = rand(1:20, 3, 7)

@tullio S[1,c] := M[r,c]  # sum over r ∈ 1:3, for each c ∈ 1:7
@test S == sum(M, dims=1) 

@tullio Q[ρ,c] := M[ρ,c] + sqrt(S[1,c])  # loop over ρ & c, no sum -- broadcasting
@test Q ≈ M .+ sqrt.(S)

mult(M,Q) = @tullio P[x,y] := M[x,c] * Q[y,c]  # sum over c ∈ 1:7 -- matrix multiplication
@test mult(M,Q) ≈ M * transpose(Q)

R = [rand(Int8, 3, 4) for δ in 1:5]

@tullio T[j,i,δ] := R[δ][i,j] + 10im  # three nested loops -- concatenation
@test T == permutedims(cat(R...; dims=3), (2,1,3)) .+ 10im

@tullio (max) X[i] := abs2(T[j,i,δ])  # reduce using max, over j and δ
@test X == dropdims(maximum(abs2, T, dims=(1,3)), dims=(1,3))

dbl!(M, S) = @tullio M[r,c] = 2 * S[1,c]  # write into existing matrix, M .= 2 .* S
dbl!(M, S)
@test all(M[r,c] == 2*S[1,c] for r ∈ 1:3, c ∈ 1:7)
```

More complicated examples:

```julia
using Tullio
A = [abs2(i - 11) for i in 1:21]

# Downsample -- range of i is that allowed by both terms:
@tullio B[i] := (A[2i] + A[2i+1])/2  # 1:10 == intersect(1:10, 0:10)

# Shifts -- range of i calculated in terms of that given for j:
@tullio M[i,j] := A[i+j-1]  (j in 1:15)  # i in 1:7
@tullio M[i+_,j] := A[i+j]  (j in 1:15)  # i in 0:6, automatic shift "i+_"

using OffsetArrays # Convolve a filter:
K = OffsetArray([1,-1,2,-1,1], -2:2)
@tullio C[i] := A[i+j] * K[j]    # j ∈ -2:2 implies i ∈ 3:19

# Index by the values in K
@tullio D[i,j] := A[2K[j]+i] ÷ K[j] # extrema(K)==(-1,2) implies i ∈ 3:17

# Wrapped & padded:
@tullio M[i,j] := A[mod(i+j)]  (j in 1:15, i in 1:15)   # wraps around, mod(i+j, axes(A,1))
@tullio M[i,j] := A[clamp(i+j)]  (j in 1:15, i in 1:15) # instead repeats "100"
@tullio M[i+_,j] := A[pad(i+j, 3)]  (j in 1:15)         # fills with zeros

using FFTW # Functions of the indices are OK:
S = [0,1,0,0, 0,0,0,0]
fft(S) ≈ @tullio F[k] := S[x] * exp(-im*pi/8 * (k-1) * x)  (k ∈ axes(S,1))

# Finalisers <| or |> are applied after sum (the two are equivalent):
@tullio N2[j] := sqrt <| M[i,j]^2     # N2 ≈ map(norm, eachcol(M))
@tullio n3[_] := A[i]^3  |> (_)^(1/3) # n3[1] ≈ norm(A,3), with _ anon. func.

# Reduction over any function:
@tullio (*) P[i] := A[i+k]  (k in 0:2) # product
@tullio (max) X[i,_] := D[i,j]         # maximum(D, dims=2), almost

min1(x,y) = ifelse(first(x) < first(y), x, y); # findmin(D, dims=1), almost:
@tullio (min1) Ts[j+_] := (D[i,j], (i,j))  init=(typemax(Int), (0,0))

# Access to fields & arrays -- this uses j ∈ eachindex(first(N).c)
N = [(a=i, b=i^2, c=fill(i^3,3)) for i in 1:10]
@tullio T[i,j] := (N[i].a // 1, N[i].c[j])

# Functions which create arrays are evaluated once:
@tullio R[i,j] := abs.((rand(Int8, 5)[i], rand(Int8, 5)[j]))

using NamedDims, AxisKeys # Dimension names, plus pretty printing:
@tullio M[row=i, col=j, z=k] := A[i+j-1]  (j in 1:15, k in 1:2)
@tullio S[i] := M[col=j-i, z=k, row=i+1] # sum over j,k
```

</details>

## Fast & Slow
<details>

When used with LoopVectorization, on straightforward matrix multiplication of real numbers, 
`@tullio` tends to be about as fast as OpenBLAS. Depending on the size, and on your computer. 
Here's a speed comparison on mine: [v2.5](https://github.com/mcabbott/Tullio.jl/blob/master/benchmarks/02/matmul-0.2.5-Float64-1.5.0.png).

This race is a useful diagnostic, but isn't really the goal. There is little point in avoiding 
using BLAS libraries, if you want precisely what they are optimised to give you.
One of the things `@tullio` is often very fast at is weird tensor contractions, 
for which you would otherwise need `permutedims`:

```julia
using Tullio, LoopVectorization, NNlib, BenchmarkTools

# Batched matmul, with batch index first in B:
bmm_rev(A, B) = @tullio C[i,k,b] := A[i,j,b] * B[b,k,j]  # (sum over j)

A = randn(20,30,500); B = randn(500,40,30);
bmm_rev(A, B) ≈ NNlib.batched_mul(A, permutedims(B, (3,2,1)))  # true

@btime bmm_rev($A, $B);  # 317.526 μs μs, same speed as un-permuted
@btime NNlib.batched_mul($A, permutedims($B, (3,2,1)));  # 1.478 ms, with MKL
```

Complex numbers aren't handled by LoopVectorization, so will be much slower.

Chained multiplication is also very slow, because it doesn't know there's a better
algorithm. Here it just makes 4 loops, instead of multiplying sequentially, 
`30^4` instead of `2 * 30^3` operations:

```julia
M1, M2, M3 = randn(30,30), randn(30,30), randn(30,30);
@btime $M1 * $M2 * $M3;                                   #  3.525 μs
@btime @tullio M4[i,l] := $M1[i,j] * $M2[j,k] * $M3[k,l]; # 30.401 μs
```

Or slightly less obviously:

```julia
M, Σ = randn(100,100), randn(100,100);
@tullio R4[i, j] := (M[μ, i] - M[μ,j])' * Σ[μ,ν] * (M[ν, i] - M[ν, j]);
begin
  S = M' * Σ * M  # two N^3 operations, instead of one N^4
  @tullio R3[i,j] := S[i,i] + S[j,j] - S[i,j] - S[j,i]
end;
R3 ≈ R4
```

Another thing Tullio can be very fast at is broadcast reductions, where it can avoid large allocations. Here LoopVectorization is speeding up `log`, and Tullio is handling tiled memory access and multi-threading:

```julia
sum_opp(X, Y=X) = @tullio s := X[i,j] * log(Y[j,i])
sum_part(X, Y=X) = @tullio S[i] := X[i,j] * log(Y[j,i])

X = rand(1000,1000);
@btime sum_opp($X)                    #   499.814 μs (93 allocations: 3.97 KiB)
@btime sum($X .* log.(transpose($X))) # 8.759 ms (2 allocations: 7.63 MiB)

@btime sum_part($X)'                           #  1.599 ms (not the same computer!)
@btime sum($X .* log.(transpose($X)), dims=2)  # 13.292 ms
```

At present indices using `pad`, `clamp` or `mod` are also slow. These result in extra
checks or operations at every iteration, not just around the edges:

```julia
conv1(x,k) = @tullio y[i+_, j+_] := x[i+a, j+b] * k[a,b]
conv2(x,k) = @tullio y[i+_, j+_] := x[2i-a, 2j-b] * k[a,b]
conv3(x,k) = @tullio y[i+_, j+_] := x[pad(i-a,3), pad(j-b,3)] * k[a,b]

x100 = rand(100,100); k7 = randn(7,7);
@btime conv1($x100, $k7); #  25.574 μs
@btime conv2($x100, $k7); #  44.590 μs
@btime conv3($x100, $k7); #  86.228 μs

using Flux
x104 = reshape(x100,(100,100,1,1)); k74 = reshape(k7,(7,7,1,1)); 
conv1(x100, k7) ≈ @btime CrossCor($k74, false)($x104)       # 586.694 μs
conv2(x100, k7) ≈ @btime Conv($k74, false, stride=2)($x104) # 901.573 μs
conv3(x100, k7) ≈ @btime Conv($k74, false, pad=3)($x104)    # 932.658 μs

using DSP
@btime DSP.conv($x100, $k7); # 198.331 μs
```

</details>

## Gradients & GPU
<details><summary><b>Derivatives & GPU</b></summary>

```julia
using Tullio
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

A = rand(3,40); B = rand(40,500);
A * B ≈ mul(A, B) # true

using Tracker # or Zygote
ΔA = Tracker.gradient((A,B) -> sum(mul(A, B)), A, B)[1]
ΔA ≈ ones(3,500) * B' # true

using CUDA, KernelAbstractions # Now defined with a GPU version:
mul(A, B) = @tullio C[i,k] := A[i,j] * B[j,k]

cu(A * B) ≈ mul(cu(A), cu(B)) # true

cu(ΔA) ≈ Tracker.gradient((A,B) -> sum(mul(A, B)), cu(A), cu(B))[1] # true

# Reduction over min/max:
Tracker.gradient(x -> (@tullio (max) res := x[i]^3), [1,2,3,-2,-1,3])[1]
```

Some warnings are in order:
* Complete reductions to a number will not work on the GPU at present.
  They were extremely slow, and a re-organisation of multi-threading for the CPU case killed them, sorry.
* Gradients are not calculated for scalars, only arrays.
  Thus for example `gradient(a -> (@tullio _ := $a * A[i]), 3.14)` will be zero.
* When using `grad=Dual`, the right hand side is evaluated a second time during the backward pass.
  This avoids needing memory to store partials, but if the function is expensive, it may be slow.


</details>

## Larger Expressions
<details>

The expression need not be just one line, for example:

```julia
@tullio out[x, y] := @inbounds(begin  # sum over k
        a,b = off[k]
        mat[mod(x+a), mod(y+b)]
    end) (x in axes(mat,1), y in axes(mat,2)) grad=Dual nograd=off
```
Here the macro cannot infer the range of the output's indices `x,y`, so they must be provided explicitly.
(If writing into an existing array, with `out[x,y] = begin ...` or `+=`, then ranges would be taken from there.)
Because it sees assignment being made, it does not attempt to sum over `a,b`, and it assumes that indices could go out of bounds so does not add `@inbounds` for you.
(Although in fact `mod(x+a) == mod(x+a, axes(mat,1))` is safe.)
It will also not be able to take a symbolic derivative, but dual numbers will work fine.

More examples:

```julia
using Tullio, OffsetArrays

# A convolution with cyclic indices
mat = zeros(10,10,1); mat[2,2] = 101; mat[10,10] = 1;
@tullio kern[i,j] := 1/(1+i^2+j^2)  (i in -3:3, j in -3:3)

@tullio out[x,y,c] := begin
    xi = mod(x+i, axes(mat,1)) # xi = ... means that it won't be summed,
    # yj = mod(y+j, axes(mat,2))
    @inbounds trunc(Int, mat[xi, mod(y+j), c] * kern[i,j]) # and disables automatic @inbounds,
end (x in 1:10, y in 1:10) # and prevents range of x from being inferred.

# A stencil?
offsets = [(a,b) for a in -2:2 for b in -2:2 if a>=b] # vector of tuples

@tullio out[x,y,1] = begin
        a,b = offsets[k]
        i = clamp(x+a, extrema(axes(mat,1))...)
        # j = clamp(y+b, extrema(axes(mat,2))...) # can be written clamp(y+b)
        @inbounds mat[i, clamp(y+b), 1] * 10
    end # ranges of x,y read from out[x,y,1]

# Applying a vector of functions
fs = [sin, cos, tan]
xs = randn(3,100)
@tullio ys[r,c] := (fs[r])(xs[r,c])

using Zygote, ForwardDiff
rowmap(fs, xs) = @tullio ys[r,c] := (fs[r])(xs[r,c]) grad=Dual nograd=fs
Zygote.gradient(sum∘rowmap, fs, ones(3,2))
[f'(1) for f in fs] # agrees
```

</details>

## Keyword Options
<details>

The default setting is:
```@tullio threads=true fastmath=true avx=true tensor=true cuda=256 grad=Base verbose=false A[i,j] := ...```
* `threads=false` turns off threading, while `threads=64^3` sets a threshold size at which to divide the work (replacing the macro's best guess).
* `avx=false` turns off the use of `LoopVectorization`, while `avx=4` inserts `@avx unroll=4 for i in ...`.
* `grad=false` turns off gradient calculation, and `grad=Dual` switches it to use `ForwardDiff` (which must be loaded).
* `nograd=A` turns of the gradient calculation just for `A`, and `nograd=(A,B,C)` does this for several arrays.
* `tensor=false` turns off the use of `TensorOperations`.
* Assignment `xi = ...` removes `xi` from the list of indices: its range is note calculated, and it will not be summed over. It also disables `@inbounds` since this is now up to you.
* `verbose=true` prints things like the index ranges inferred, and gradient calculations. `verbose=2` prints absolutely everything.
* `A[i,j] := ...` makes a new array, while `A[i,j] = ...` and `A[i,j] += ...` write into an existing one. `A[row=i, col=j] := ...` makes a new `NamedDimsArray`.
* `@tullio (*) A[i,j] := ...` is a product, as is `@tullio A[i,j] *= ...`. For other reductions, `@tullio (f) A[i,j] ^= ...` is an in-place update.
* `init=0.0` gives the initial value for reductions. For `+`, `*`, `min`, `min`, `&`, `|` it has sensible defaults, for other reductions uses zero.

Implicit:
* Indices without shifts must have the same range everywhere they appear, but those with shifts (even `A[i+0]`) run over the intersection of possible ranges.
* Shifted output indices must start at 1, unless `OffsetArrays` is visible in the calling module.
* The use of `@avx`, and the calculation of gradients, are switched off by sufficiently complex syntax (such as arrays of arrays).
* Gradient hooks are attached for any or all of `ReverseDiff`, `Tracker` & `Zygote`. These packages need not be loaded when the macro is run.
* Gradients are only defined for reductions over `(+)` (default) and `min`, `max`.
* GPU kernels are only constructed when both `KernelAbstractions` and `CUDA` are visible. The default `cuda=256` is passed to `kernel(CUDA(), 256)`.
* The CPU kernels from `KernelAbstractions` are called only when `threads=false`; they are not at present very fast, but perhaps useful for testing.

Extras:
* `A[i] := i^2  (i in 1:10)` is how you specify a range for indices when this can't be inferred.
* `A[i] := B[i, $col] - C[i, 2]` is how you fix one index to a constant (to prevent `col` being summed over).
* `A[i] := $d * B[i]` is the preferred way to include other constants. Note that no gradient is calculated for `d`.
* Within indexing, `A[mod(i), clamp(j)]` both maps `i` & `j` to lie within `axes(A)`, and disables inference of their ranges from `A`.
* Similarly, `A[pad(i,3)]` extends the range of `i`, inserting zeros outside of `A`. Instead of zero, `pad=NaN` uses this value as padding. The implementation of this (and `mod`, `clamp`) is not very fast at present.
* On the left, when making a new array, an underscore like `A[i+_] :=` inserts whatever shift is needed to make `A` one-based.
* `Tullio.@printgrad (x+y)*log(x/z)   x y z` prints out how symbolic derivatives will be done.

Macros:
* `Tullio.@tensor` is a macro which uses TensorOperations to evaluate expressions, but provides gradient definitions. (Previously this was automatic behaviour, when TensorOperations.jl was loaded & the expression was suitable.)
* `Tullio.@einsum` is a variant with a few changes, to allow the running of Einsum.jl's tests.

</details>

## How it Works
<details>

The following three macros all end up calling the same functions as does `C = A * B`:

```julia
@tensor C[i,j] := A[i,k] * B[k,j]         # TensorOperations.jl
@ein C[i,j] := A[i,k] * B[k,j]            # OMEinsum.jl
@matmul C[i,j] := sum(k) A[i,k] * B[k,j]  # TensorCast.jl
```

But this one writes its own for-loops:

```julia
@einsum C[i,j] := A[i,k] * B[k,j]         # Einsum.jl
```

expanding out to roughly this:

```julia
T = promote_type(eltype(A), eltype(B))
C = Array{T}(undef, size(A,1), size(B,2))
@inbounds for j in 1:size(B,2)
    for i in 1:size(A,1)
        acc = zero(T)
        for k in 1:size(A,2)
            acc += A[i,k] * B[k,j]
        end
        C[i,j] = acc
    end
end
```

Tullio does something similar, but working through a few functions. Taking a slightly more complicated example, this:

```julia
@tullio C[i,j] := tanh <| A[i,k] * B[k,j]
```

expands to roughly this:

```julia
function act!(::Type, C::AbstractArray{T}, A, B, ax_i, ax_j, ax_k, keep=nothing, final=true) where T
    @inbounds @fastmath for i in ax_i
        for j in ax_j
            acc = isnothing(keep) ? zero(T) : C[i,j]
            for k in ax_k
                acc += A[i,k] * B[k,j]
            end
            C[i,j] = isnothing(final) ? acc : tanh(acc)
        end
    end
end

function make(A, B)
    ax_i = axes(A,1)
    ax_j = axes(B,2)
    ax_k = axes(A,2) # and check this is == axes(B,1)
    rhs(A,B,i,j,k) = tanh(A[i,k] * B[k,j])
    T = Core.Compiler.return_type(rhs, eltype.((A,B,1,1,1))) # plus a fallback
    C = similar(A, T, (ax_i, ax_j))
    Tullio.threader(act!, Array{T}, C, (A,B), (ax_i,ax_j), (ax_k,), +, 64^3)
    return C
end

C = Tullio.Eval(make, ∇make)(A, B)
```

This division allows it to dispatch to other methods of `act!`: one generated with `@avx` if LoopVectorization is loaded, and one for `::CuArray` if KernelAbstractions is loaded.

It also allows `threader` to divide the work, calling `act!` many times, from different threads, on small tiles made by dividing the longest axis (say `ax_i`) in half, repeatedly. If it divides up `ax_k`, these are done sequentially, with `keep=true` on all ranges except the first, and `final=nothing` on all except the last. But `ax_i` and `ax_j` are safe to do in parallel.

Finally, `Eval` exists to give Zygote and friends somewhere to attach themselves. The gradient calculation looks roughly like this:

```julia
@adjoint function (e::Eval)(AB...)
    C = e.fwd(AB...)
    C, ΔC -> e.rev(ΔC, C, AB...)
end

function ∇act!(::Type, ΔC, ΔA, ΔB, C, A, B, ax_i, ax_j, ax_k, keep)
    for k in ax_k, i in ax_i, j in ax_j
        ex = ΔC[i,j] * (1-C[i,j])^2
        ΔA[i,k] += ex * B[k,j]
        ΔB[k,j] += A[i,k] * ex
    end
end

function ∇make(ΔC, C, A, B)
    ΔA = similar(A) .= 0
    ΔB = similar(B) .= 0
    ax_i, ax_k = axes(A); ax_j = axes(B,2)
    Tullio.∇threader(∇act!, Array{T}, (ax_k,), (ax_i, ax_j), nothing)
    return (ΔA, ΔB)
end
```

In this case, it is the loop over `k` which can be safely broken among different threads, since both `ΔA` and `ΔB` have this index. Both `ΔA` and `ΔB` are filled in at once.

Notice that the derivative of `y = tanh(z)` has been written in terms of `y` (the final result of the forward pass) but free of `z` (the result of the sum, which was not saved). If this is not possible, it will fail.

If using `grad=Dual`, the gradient kernel looks different. This method cannot handle finalisers like `tanh` above, but for plain `@tullio C[i,j] := A[i,k] * B[k,j]` it would read:

```julia
function ∇act!(::Type, ΔC, ΔA, ΔB, C, A, B, ax_i, ax_j, ax_k, keep)
    eps1 = ForwardDiff.Dual(0, (1,0))
    eps2 = ForwardDiff.Dual(0, (0,1))
    for k in ax_k, i in ax_i, j in ax_j
        res = (A[i,k] + eps1) * (B[k,j] + eps2)
        ΔA[i,k] += ForwardDiff.partials(res, 1) * ΔC[i,j]
        ΔB[k,j] += ForwardDiff.partials(res, 2) * ΔC[i,j]
    end
end
```

Writing `@tullio verbose=2` will print all of these functions out.

Scalar reductions, such as `@tullio s := A[i,j] * log(B[j,i])`, are slightly different in that the `act!` function simply returns the sum, i.e. the variable `acc` above.

</details>

## Elsewhere

Back-end friends & relatives:

* [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl) is used here, if available.

* [Gaius.jl](https://github.com/MasonProtter/Gaius.jl) and [PaddedMatrices.jl](https://github.com/chriselrod/PaddedMatrices.jl) build on that.

* [GPUifyLoops.jl](https://github.com/vchuravy/GPUifyLoops.jl) and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) generate GPU-compatible kernels.

* [ThreadsX.jl](https://github.com/tkf/ThreadsX.jl) does threaded reductions, and much else.

* [Strided.jl](https://github.com/Jutho/Strided.jl) does multi-threaded broadcasting.

Front-end near-lookalikes:

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl) makes simple loops. See [tests/einsum.jl](https://github.com/mcabbott/Tullio.jl/blob/master/test/einsum.jl) where `using Tullio: @einsum` is an almost-seamless replacement.

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) identify patterns on which they can call various basic operations. [TensorRules.jl](https://github.com/ho-oto/TensorRules.jl) makes `@tensor` differentiable; see also [TensorGrad.jl](https://github.com/mcabbott/TensorGrad.jl) and [TensorTrack.jl](https://github.com/mcabbott/TensorTrack.jl) for earlier attempts.

* [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl) expresses everything as Julia array operations, broadcasting and reduction. (OMEinsum.jl also treats some cases as a special lazy broadcast-reduction.)

Things you can't run:

* [Tortilla.jl](https://www.youtube.com/watch?v=Rp7sTl9oPNI) seems to exist, publicly, only in this very nice talk.

* [ArrayMeta.jl](https://github.com/shashi/ArrayMeta.jl) was a Julia 0.5 take on some of this.

* [Tokamak.jl](https://github.com/MikeInnes/Tokamak) was another, see [readme here](https://github.com/tkelman/Tokamak.jl).
````
