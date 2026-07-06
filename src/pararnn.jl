# ParaRNN: sequence-parallel nonlinear RNNs via Newton iterations.
#
# Based on "ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large
# Language Models" (Danieli et al., 2025).  This is the small, from-scratch
# version of the paper's math: no CUDA kernels, only Julia arrays and the
# diagonal / 2×2 block-diagonal Jacobian structure from the ParaGRU and
# ParaLSTM cells.

sigmoid_scalar(x) = inv(one(x) + exp(-x))
sigmoid_derivative_from_value(y) = y * (one(y) - y)
tanh_derivative_from_value(y) = one(y) - y^2

"One Hillis-Steele style parallel prefix solve for diagonal affine recurrences."
function pararnn_scan_diag(A, b; threaded = Threads.nthreads() > 1)
    threaded || return _pararnn_scan_diag_ad(A, b)

    L = size(b, 2)
    δ = copy(b)
    Ã = copy(A)
    offset = 1
    while offset < L
        oldδ = δ
        oldÃ = Ã
        δ = similar(oldδ)
        Ã = similar(oldÃ)
        δ[:, 1:offset] .= oldδ[:, 1:offset]
        Ã[:, 1:offset] .= oldÃ[:, 1:offset]
        Threads.@threads for l in (offset + 1):L
            left = l - offset
            @views begin
                δ[:, l] .= oldδ[:, l] .+ oldÃ[:, l] .* oldδ[:, left]
                Ã[:, l] .= oldÃ[:, l] .* oldÃ[:, left]
            end
        end
        offset *= 2
    end
    δ
end

"AD-friendly scan path: pure array expressions, no threaded mutation."
function _pararnn_scan_diag_ad(A, b)
    L = size(b, 2)
    δ = b
    Ã = A
    offset = 1
    while offset < L
        left = 1:(L - offset)
        right = (offset + 1):L
        δ = hcat(δ[:, 1:offset], δ[:, right] .+ Ã[:, right] .* δ[:, left])
        Ã = hcat(Ã[:, 1:offset], Ã[:, right] .* Ã[:, left])
        offset *= 2
    end
    δ
end

"Parallel prefix solve for independent 2×2 block-diagonal affine recurrences."
function pararnn_scan_block2(Jcc, Jch, Jhc, Jhh, b; threaded = Threads.nthreads() > 1)
    threaded || return _pararnn_scan_block2_ad(Jcc, Jch, Jhc, Jhh, b)

    L = size(b, 2)
    H = size(Jcc, 1)
    bc = copy(b[1:H, :])
    bh = copy(b[H+1:2H, :])
    Acc, Ach, Ahc, Ahh = copy(Jcc), copy(Jch), copy(Jhc), copy(Jhh)

    offset = 1
    while offset < L
        oldbc, oldbh = bc, bh
        oldAcc, oldAch, oldAhc, oldAhh = Acc, Ach, Ahc, Ahh
        bc, bh = similar(oldbc), similar(oldbh)
        Acc, Ach, Ahc, Ahh = similar(oldAcc), similar(oldAch), similar(oldAhc), similar(oldAhh)
        for M in ((bc, oldbc), (bh, oldbh), (Acc, oldAcc), (Ach, oldAch), (Ahc, oldAhc), (Ahh, oldAhh))
            M[1][:, 1:offset] .= M[2][:, 1:offset]
        end

        Threads.@threads for l in (offset + 1):L
            left = l - offset
            @views begin
                bc[:, l] .= oldbc[:, l] .+ oldAcc[:, l] .* oldbc[:, left] .+ oldAch[:, l] .* oldbh[:, left]
                bh[:, l] .= oldbh[:, l] .+ oldAhc[:, l] .* oldbc[:, left] .+ oldAhh[:, l] .* oldbh[:, left]

                Acc[:, l] .= oldAcc[:, l] .* oldAcc[:, left] .+ oldAch[:, l] .* oldAhc[:, left]
                Ach[:, l] .= oldAcc[:, l] .* oldAch[:, left] .+ oldAch[:, l] .* oldAhh[:, left]
                Ahc[:, l] .= oldAhc[:, l] .* oldAcc[:, left] .+ oldAhh[:, l] .* oldAhc[:, left]
                Ahh[:, l] .= oldAhc[:, l] .* oldAch[:, left] .+ oldAhh[:, l] .* oldAhh[:, left]
            end
        end
        offset *= 2
    end

    vcat(bc, bh)
end

"AD-friendly 2×2 block scan path: pure array expressions, no threaded mutation."
function _pararnn_scan_block2_ad(Jcc, Jch, Jhc, Jhh, b)
    L = size(b, 2)
    H = size(Jcc, 1)
    bc = b[1:H, :]
    bh = b[H+1:2H, :]
    Acc, Ach, Ahc, Ahh = Jcc, Jch, Jhc, Jhh

    offset = 1
    while offset < L
        left = 1:(L - offset)
        right = (offset + 1):L

        bc_right = bc[:, right] .+ Acc[:, right] .* bc[:, left] .+ Ach[:, right] .* bh[:, left]
        bh_right = bh[:, right] .+ Ahc[:, right] .* bc[:, left] .+ Ahh[:, right] .* bh[:, left]

        Acc_right = Acc[:, right] .* Acc[:, left] .+ Ach[:, right] .* Ahc[:, left]
        Ach_right = Acc[:, right] .* Ach[:, left] .+ Ach[:, right] .* Ahh[:, left]
        Ahc_right = Ahc[:, right] .* Acc[:, left] .+ Ahh[:, right] .* Ahc[:, left]
        Ahh_right = Ahc[:, right] .* Ach[:, left] .+ Ahh[:, right] .* Ahh[:, left]

        bc = hcat(bc[:, 1:offset], bc_right)
        bh = hcat(bh[:, 1:offset], bh_right)
        Acc = hcat(Acc[:, 1:offset], Acc_right)
        Ach = hcat(Ach[:, 1:offset], Ach_right)
        Ahc = hcat(Ahc[:, 1:offset], Ahc_right)
        Ahh = hcat(Ahh[:, 1:offset], Ahh_right)
        offset *= 2
    end

    vcat(bc, bh)
end

@kwdef struct ParaGRU
    input_dim = 8
    hidden_dim = 8
    a_z = 0.1f0 .* randn(Float32, hidden_dim)
    a_r = 0.1f0 .* randn(Float32, hidden_dim)
    a_c = 0.1f0 .* randn(Float32, hidden_dim)
    B_z = glorot(hidden_dim, input_dim; gain = sqrt(2f0))
    B_r = glorot(hidden_dim, input_dim; gain = sqrt(2f0))
    B_c = glorot(hidden_dim, input_dim; gain = sqrt(2f0))
    b_z = zeros(Float32, hidden_dim)
    b_r = zeros(Float32, hidden_dim)
    b_c = zeros(Float32, hidden_dim)
end

"ParaGRU recurrence, Eq. (3.1a), with diagonal recurrent matrices."
function paragru_step(cell::ParaGRU, h, x)
    z = sigmoid_scalar.(cell.a_z .* h .+ cell.B_z * x .+ cell.b_z)
    r = sigmoid_scalar.(cell.a_r .* h .+ cell.B_r * x .+ cell.b_r)
    c = tanh.(cell.a_c .* h .* r .+ cell.B_c * x .+ cell.b_c)
    (one(eltype(h)) .- z) .* h .+ z .* c
end

function _paragru_system(cell::ParaGRU, H, X)
    Hdim = size(H, 1)
    z0 = zeros(eltype(H), Hdim)

    terms = map(axes(X, 2)) do l
        hm1 = l == 1 ? z0 : H[:, l - 1]
        hcur = H[:, l]
        x = X[:, l]
        ẑ = cell.a_z .* hm1 .+ cell.B_z * x .+ cell.b_z
        r̂ = cell.a_r .* hm1 .+ cell.B_r * x .+ cell.b_r
        z = sigmoid_scalar.(ẑ)
        r = sigmoid_scalar.(r̂)
        ĉ = cell.a_c .* hm1 .* r .+ cell.B_c * x .+ cell.b_c
        c = tanh.(ĉ)

        dz = cell.a_z .* sigmoid_derivative_from_value.(z)
        dr = cell.a_r .* sigmoid_derivative_from_value.(r)
        dc = cell.a_c .* tanh_derivative_from_value.(c) .* (r .+ hm1 .* dr)
        J = (one(eltype(H)) .- z) .+ (c .- hm1) .* dz .+ z .* dc
        residual = (one(eltype(H)) .- z) .* hm1 .+ z .* c .- hcur
        (J, residual)
    end

    hcat(first.(terms)...), hcat(last.(terms)...)
end

"Sequential ParaGRU application, useful as the reference solution."
function forward_sequential(cell::ParaGRU, X)
    H = size(cell.a_z, 1)
    h = zeros(eltype(X), H)
    states = Vector{typeof(h)}(undef, size(X, 2))
    for l in axes(X, 2)
        h = paragru_step(cell, h, X[:, l])
        states[l] = h
    end
    hcat(states...)
end

"Sequence-parallel ParaGRU application using Newton + parallel reduction."
function forward(cell::ParaGRU, X; newton_iters = 3, threaded = Threads.nthreads() > 1)
    Hdim = size(cell.a_z, 1)
    z0 = zeros(eltype(X), Hdim)
    H = reduce(hcat, map(l -> paragru_step(cell, z0, X[:, l]), axes(X, 2)))
    for _ in 1:newton_iters
        J, r = _paragru_system(cell, H, X)
        H = H .+ pararnn_scan_diag(J, r; threaded)
    end
    H
end

@kwdef struct ParaLSTM
    input_dim = 8
    hidden_dim = 8
    a_f = 0.1f0 .* randn(Float32, hidden_dim)
    a_z = 0.1f0 .* randn(Float32, hidden_dim)
    a_o = 0.1f0 .* randn(Float32, hidden_dim)
    c_f = 0.1f0 .* randn(Float32, hidden_dim)
    c_o = 0.1f0 .* randn(Float32, hidden_dim)
    B_f = glorot(hidden_dim, input_dim; gain = sqrt(2f0))
    B_z = glorot(hidden_dim, input_dim; gain = sqrt(2f0))
    B_o = glorot(hidden_dim, input_dim; gain = sqrt(2f0))
    b_f = zeros(Float32, hidden_dim)
    b_z = zeros(Float32, hidden_dim)
    b_o = zeros(Float32, hidden_dim)
end

"ParaLSTM recurrence, Eq. (3.1b), with diagonal state/peephole matrices."
function paralstm_step(cell::ParaLSTM, state, x)
    H = length(cell.a_f)
    cprev = state[1:H]
    hprev = state[H+1:2H]
    f = sigmoid_scalar.(cell.a_f .* hprev .+ cell.B_f * x .+ cell.c_f .* cprev .+ cell.b_f)
    z = tanh.(cell.a_z .* hprev .+ cell.B_z * x .+ cell.b_z)
    c = f .* cprev .+ (one(eltype(state)) .- f) .* z
    o = sigmoid_scalar.(cell.a_o .* hprev .+ cell.B_o * x .+ cell.c_o .* c .+ cell.b_o)
    h = o .* tanh.(c)
    vcat(c, h)
end

function _paralstm_system(cell::ParaLSTM, S, X)
    H = length(cell.a_f)
    z0 = zeros(eltype(S), 2H)

    terms = map(axes(X, 2)) do l
        sprev = l == 1 ? z0 : S[:, l - 1]
        scur = S[:, l]
        cprev = sprev[1:H]
        hprev = sprev[H+1:2H]
        x = X[:, l]

        f̂ = cell.a_f .* hprev .+ cell.B_f * x .+ cell.c_f .* cprev .+ cell.b_f
        ẑ = cell.a_z .* hprev .+ cell.B_z * x .+ cell.b_z
        f = sigmoid_scalar.(f̂)
        z = tanh.(ẑ)
        c = f .* cprev .+ (one(eltype(S)) .- f) .* z
        ô = cell.a_o .* hprev .+ cell.B_o * x .+ cell.c_o .* c .+ cell.b_o
        o = sigmoid_scalar.(ô)
        h = o .* tanh.(c)

        df = sigmoid_derivative_from_value.(f)
        dz = tanh_derivative_from_value.(z)
        do_ = sigmoid_derivative_from_value.(o)
        tanhc = tanh.(c)
        dtanhc = tanh_derivative_from_value.(tanhc)

        Jcc = f .+ (cprev .- z) .* df .* cell.c_f
        Jch = (cprev .- z) .* df .* cell.a_f .+ (one(eltype(S)) .- f) .* dz .* cell.a_z
        Jhc = (tanhc .* do_ .* cell.c_o .+ o .* dtanhc) .* Jcc
        Jhh = tanhc .* do_ .* (cell.a_o .+ cell.c_o .* Jch) .+ o .* dtanhc .* Jch
        residual = vcat(c, h) .- scur
        (Jcc, Jch, Jhc, Jhh, residual)
    end

    hcat(getindex.(terms, 1)...),
    hcat(getindex.(terms, 2)...),
    hcat(getindex.(terms, 3)...),
    hcat(getindex.(terms, 4)...),
    hcat(getindex.(terms, 5)...)
end

"Sequential ParaLSTM application, useful as the reference solution."
function forward_sequential(cell::ParaLSTM, X)
    H = length(cell.a_f)
    s = zeros(eltype(X), 2H)
    states = Vector{typeof(s)}(undef, size(X, 2))
    for l in axes(X, 2)
        s = paralstm_step(cell, s, X[:, l])
        states[l] = s
    end
    hcat(states...)
end

"Sequence-parallel ParaLSTM application using Newton + 2×2 block reduction."
function forward(cell::ParaLSTM, X; newton_iters = 3, threaded = Threads.nthreads() > 1)
    H = length(cell.a_f)
    z0 = zeros(eltype(X), 2H)
    S = reduce(hcat, map(l -> paralstm_step(cell, z0, X[:, l]), axes(X, 2)))
    for _ in 1:newton_iters
        Jcc, Jch, Jhc, Jhh, r = _paralstm_system(cell, S, X)
        S = S .+ pararnn_scan_block2(Jcc, Jch, Jhc, Jhh, r; threaded)
    end
    S
end

_pararnn_output(cell::ParaGRU, states) = states
_pararnn_output(cell::ParaLSTM, states) = states[(length(cell.a_f) + 1):end, :]

@kwdef struct ParaRNNLanguageModel
    E = glorot(8, 29; gain = inv(sqrt(3f0)))
    cell = ParaGRU(input_dim = 8, hidden_dim = 8)
    W_out = glorot(29, 8; gain = inv(sqrt(3f0)))
    b_out = zeros(Float32, 29, 1)
end

"Character-level language model using a ParaRNN cell as the sequence mixer."
function forward(x, θ::ParaRNNLanguageModel; newton_iters = 3, threaded = false)
    X = θ.E[:, x]
    states = forward(θ.cell, X; newton_iters, threaded)
    θ.W_out * _pararnn_output(θ.cell, states) .+ θ.b_out
end

function loss(θ::ParaRNNLanguageModel, x, y; newton_iters = 3, threaded = false)
    ŷ = forward(x, θ; newton_iters, threaded)
    max_ŷ = maximum(ŷ; dims = 1)
    log_probs = ŷ .- max_ŷ .- log.(sum(exp.(ŷ .- max_ŷ); dims = 1))
    correct_log_probs = log_probs[CartesianIndex.(y, eachindex(y))]
    -mean(correct_log_probs)
end

function step!(model::ParaRNNLanguageModel, x, y, η; newton_iters = 3)
    (∇,) = gradient(m -> loss(m, x, y; newton_iters, threaded = false), model)
    sgd!(model, ∇, η)
end

function train!(model::ParaRNNLanguageModel, x, y, η, epochs; newton_iters = 3)
    for _ in 1:epochs
        step!(model, x, y, η; newton_iters)
    end
    model
end

function generate(model::ParaRNNLanguageModel, vocab, seed; n::Int = 20, choose = choose, newton_iters = 3, threaded = Threads.nthreads() > 1)
    idx = encode(string(seed), vocab)
    for _ in 1:n
        logits = forward(idx, model; newton_iters, threaded)
        p = softmax(logits[:, end])
        push!(idx, choose(p))
    end
    join(vocab[i] for i in idx)
end
