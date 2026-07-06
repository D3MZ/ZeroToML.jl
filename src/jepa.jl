# I-JEPA: Image-based Joint-Embedding Predictive Architecture
# https://arxiv.org/abs/2301.08243
#
# From a single context block, predict the representations of various target
# blocks in the same image.  Predictions and targets live in *representation*
# space (not pixel space), which pushes the model to learn semantic features.
#
# Key design choices from the paper:
#   (a) Target blocks are sufficiently large (semantic scale 0.15–0.20 of image)
#   (b) Context block is informative yet sparse (scale 0.85–1.0, holes where targets sit)
#   (c) Target encoder is an EMA of the context encoder (prevents collapse)

using Base: @kwdef
using Random, Statistics, Zygote, Dates
using NNlib: conv
using .ZeroToML: glorot, relu, sgd!

# ── Multi-block mask ────────────────────────────────────────────────────────
# Each mask is a BitMatrix (num_blocks × num_blocks).  `true` means *visible*.

"Sample M random rectangular target blocks; return (target_masks, context_mask).

Each target block has a random scale in `target_scale` and random aspect ratio
in `[1/aspect_max, aspect_max]`.  The context mask covers a large region of the
image (scale in `context_scale`) with the target regions removed so the task is
non-trivial.

Returns
  `target_masks::Vector{BitMatrix}`  – one mask per target block (1 = in target)
  `context_mask::BitMatrix`          – 1 = in context (used by encoder)
"
function sample_multiblock_mask(
    grid_h::Int,
    grid_w::Int;
    num_targets::Int = 4,
    target_scale = (0.15f0, 0.20f0),
    context_scale = (0.85f0, 0.95f0),
    aspect_max = 1.5f0,
    rng = Random.default_rng(),
)
    target_masks = Vector{BitMatrix}(undef, num_targets)
    occupied = falses(grid_h, grid_w)  # cells already claimed by a target

    for i in 1:num_targets
        # Random area for this target block (fraction of total grid cells)
        area_frac = rand(rng, Float32) * (target_scale[2] - target_scale[1]) + target_scale[1]
        target_area = max(1, round(Int, area_frac * grid_h * grid_w))

        # Random aspect ratio
        ar = rand(rng, Float32) * (aspect_max - 1f0 / aspect_max) + 1f0 / aspect_max
        h = round(Int, sqrt(target_area / ar))
        w = round(Int, target_area ÷ h)
        h = clamp(h, 1, grid_h)
        w = clamp(w, 1, grid_w)

        # Random top-left position
        y0 = rand(rng, 1:grid_h - h + 1)
        x0 = rand(rng, 1:grid_w - w + 1)

        mask = falses(grid_h, grid_w)
        mask[y0:y0+h-1, x0:x0+w-1] .= true
        target_masks[i] = mask
        occupied .|= mask
    end

    # Context: a random visible block, with target regions removed.  In I-JEPA
    # the context block can be large; tests can also set a small context_scale
    # for a MAE-like/high-masking visualization.
    area_frac = rand(rng, Float32) * (context_scale[2] - context_scale[1]) + context_scale[1]
    context_area = max(1, round(Int, area_frac * grid_h * grid_w))
    ar = rand(rng, Float32) * (aspect_max - 1f0 / aspect_max) + 1f0 / aspect_max
    h = round(Int, sqrt(context_area / ar))
    w = round(Int, context_area ÷ h)
    h = clamp(h, 1, grid_h)
    w = clamp(w, 1, grid_w)
    y0 = rand(rng, 1:grid_h - h + 1)
    x0 = rand(rng, 1:grid_w - w + 1)

    context_mask = falses(grid_h, grid_w)
    context_mask[y0:y0+h-1, x0:x0+w-1] .= true
    context_mask .&= .!occupied

    # Extremely small context blocks can be swallowed by targets; keep at least
    # one visible cell so the context encoder always receives a signal.
    if !any(context_mask)
        candidates = findall(.!occupied)
        context_mask[rand(rng, candidates)] = true
    end

    return target_masks, context_mask
end

# ── Encoder (context-encoder / target-encoder share architecture) ────────────
# Processes *only* the visible (context) pixels and outputs per-pixel
# embeddings of dimension `embed_dim`.  The same architecture is used for the
# target encoder (whose weights are an EMA of the context encoder).

@kwdef struct ConvJEPAEncoder
    embed_dim = 8
    W₁ = glorot(3, 3, 1, 16)
    b₁ = zeros(Float32, 1, 1, 16, 1)
    W₂ = glorot(3, 3, 16, 32)
    b₂ = zeros(Float32, 1, 1, 32, 1)
    W₃ = glorot(3, 3, 32, 8)
    b₃ = zeros(Float32, 1, 1, 8, 1)
end

"Encode an image with the original lightweight conv encoder."
function forward(enc::ConvJEPAEncoder, x::Matrix{Float32})
    H, W = size(x)
    h = reshape(x, H, W, 1, 1)
    p = 1  # padding for 3×3 conv

    h = conv(h, enc.W₁; pad=p) .+ enc.b₁
    h = relu(h)
    h = conv(h, enc.W₂; pad=p) .+ enc.b₂
    h = relu(h)
    h = conv(h, enc.W₃; pad=p) .+ enc.b₃

    reshape(h, H, W, enc.embed_dim)
end

@kwdef struct JEPAEncoder
    embed_dim = 8
    patch_size = 1
    W_patch = glorot(8, 1; gain = inv(sqrt(3f0)))
    b_patch = zeros(Float32, 8, 1)
    transformer = TransformerEncoder(
        embed_dim = 8,
        hidden_dim = 16,
        max_len = 144,
        P = positional_encoding(144, 8),
        Wq = glorot(8, 8; gain = inv(sqrt(3f0))),
        Wk = glorot(8, 8; gain = inv(sqrt(3f0))),
        Wv = glorot(8, 8; gain = inv(sqrt(3f0))),
        Wo = glorot(8, 8; gain = inv(sqrt(3f0))),
        ln₁_γ = ones(Float32, 8, 1),
        ln₁_β = zeros(Float32, 8, 1),
        ln₂_γ = ones(Float32, 8, 1),
        ln₂_β = zeros(Float32, 8, 1),
        W₁ = glorot(16, 8; gain = inv(sqrt(3f0))),
        b₁ = zeros(Float32, 16, 1),
        W₂ = glorot(8, 16; gain = inv(sqrt(3f0))),
        b₂ = zeros(Float32, 8, 1),
    )
end

"Patchify a 2D image into columns of flattened patch tokens."
function image_patches(x::Matrix{Float32}, patch_size::Int)
    H, W = size(x)
    @assert H % patch_size == 0 && W % patch_size == 0
    ys = 1:patch_size:H
    xs = 1:patch_size:W
    hcat([vec(x[y:y+patch_size-1, x₀:x₀+patch_size-1]) for y in ys for x₀ in xs]...)
end

"Repeat patch tokens back to per-pixel embeddings for compatibility with JEPA loss."
function tokens_to_grid(tokens, H, W, patch_size)
    D = size(tokens, 1)
    patches_h = H ÷ patch_size
    patches_w = W ÷ patch_size
    patch_grid = permutedims(reshape(tokens, D, patches_w, patches_h), (3, 2, 1))
    repeat(patch_grid; inner = (patch_size, patch_size, 1))
end

"Encode an image with ViT-style patch projection + bidirectional self-attention."
function forward(enc::JEPAEncoder, x::Matrix{Float32})
    H, W = size(x)
    patches = Zygote.ignore() do
        image_patches(x, enc.patch_size)
    end
    tokens = enc.W_patch * patches .+ enc.b_patch
    tokens = forward(enc.transformer, tokens, 1:size(tokens, 2))
    tokens_to_grid(tokens, H, W, enc.patch_size)
end

# ── Predictor ───────────────────────────────────────────────────────────────
# Takes context encoder output (only context pixels) and predicts target-block
# representations.  Uses learnable mask tokens + positional encoding for
# target locations, processed by conv layers, and projected back to encoder
# embed_dim.
#
# Following the paper: for each target block, the predictor takes the context
# encoder output and a mask token for each patch to predict, and outputs a
# patch-level prediction.  The mask tokens are a shared learnable vector with
# added positional embedding.

@kwdef struct JEPAPredictor
    embed_dim = 8          # encoder output dim
    pred_dim = 16           # internal predictor dim
    # 1×1 convs for channel projection — scaled to match encoder output magnitude
    W_embed = glorot(1, 1, 8, 16) * 0.3f0   # project embed_dim → pred_dim
    W_proj  = glorot(1, 1, 16, 8) * 0.3f0   # project pred_dim → embed_dim
    # Learnable mask tokens (one per target block, shared across positions)
    mask_tokens = zeros(Float32, 1, 1, 4, 16)  # (1,1,num_targets,pred_dim)
    # Conv layers for prediction
    W₁ = glorot(3, 3, 16, 32)
    b₁ = zeros(Float32, 1, 1, 32, 1)
    W₂ = glorot(3, 3, 32, 16)
    b₂ = zeros(Float32, 1, 1, 16, 1)
end

"Sin/cos positional encoding for a single (cy, cx) in [0,1]²."
function positional_token(cy, cx, dim)
    half = div(dim, 2)
    twoπ = oftype(cy, 2π)
    sines = map(d -> sin(cy * twoπ * d), 1:half)
    cosines = map(d -> cos(cx * twoπ * d), 1:half)
    dim % 2 == 0 ? vcat(sines, cosines) : vcat(sines, cosines, 0f0)
end

"Predict a single target block representation from context encoder output.

Returns shape (H, W, embed_dim), non-zero only at target positions.
"
function _predict_block(pred::JEPAPredictor, ctxt, tgt_mask, token_idx)
    H, W, D = size(ctxt)
    p = 1

    # Project context to pred_dim via 1×1 conv: (H,W,D,1) → (H,W,pred_dim,1)
    ctxt_4d = reshape(ctxt, H, W, pred.embed_dim, 1)
    ctxt_proj = reshape(conv(ctxt_4d, pred.W_embed; pad=0), H, W, pred.pred_dim)

    # Centroid of target block as positional signal
    tgt_cells = findall(tgt_mask)
    isempty(tgt_cells) && return zeros(Float32, H, W, pred.embed_dim)
    cy = Float32(mean(c -> c[1], tgt_cells) / H)
    cx = Float32(mean(c -> c[2], tgt_cells) / W)

    # Mask token with positional encoding, broadcast over target positions
    # positional_token depends only on mask geometry, not parameters → ignore for grad
    pos_tok = Zygote.ignore() do
        positional_token(cy, cx, pred.pred_dim)
    end
    mask_tok = pred.mask_tokens[1, 1, token_idx, :] .+ pos_tok
    mask_tok_3d = reshape(mask_tok, 1, 1, pred.pred_dim)

    # Combine: context_proj where context, mask_token where target
    ctx_flag = Float32.(.!(tgt_mask))
    tgt_flag = Float32.(tgt_mask)
    h = ctxt_proj .* ctx_flag .+ mask_tok_3d .* tgt_flag

    # Conv layers
    h = reshape(h, H, W, pred.pred_dim, 1)
    h = conv(h, pred.W₁; pad=p) .+ pred.b₁
    h = relu(h)
    h = conv(h, pred.W₂; pad=p) .+ pred.b₂

    # Project back to embed_dim via 1×1 conv
    out = reshape(conv(reshape(h, H, W, pred.pred_dim, 1), pred.W_proj; pad=0),
                  H, W, pred.embed_dim)

    # Keep only target positions
    out .* tgt_flag
end

"Predict target block representations from context encoder output.

Arguments:
  `pred` – the predictor
  `ctxt` – context encoder output, shape (H, W, embed_dim), zeroed outside context
  `target_masks` – list of BitMatrix masks, one per target block

Returns a vector of predictions, one per target block, each shape (H, W, embed_dim).
"
forward(pred::JEPAPredictor, ctxt, target_masks) =
    [_predict_block(pred, ctxt, m, i) for (i, m) in enumerate(target_masks)]

# ── I-JEPA model ────────────────────────────────────────────────────────────
# The full model bundles context encoder, predictor, and target encoder.
# Target encoder weights are maintained as an EMA of context encoder weights.

@kwdef struct JEPA
    context_encoder = JEPAEncoder()
    predictor = JEPAPredictor()
    target_encoder = deepcopy(context_encoder)  # EMA copy of context_encoder
    ema_rate = 0.993f0                         # momentum coefficient for EMA
end

"Original conv-based JEPA baseline, kept for comparisons/tests."
ConvJEPA(; predictor = JEPAPredictor(), ema_rate = 0.993f0) = begin
    context_encoder = ConvJEPAEncoder()
    JEPA(; context_encoder, predictor, target_encoder = deepcopy(context_encoder), ema_rate)
end

# ── Helpers ─────────────────────────────────────────────────────────────────

"Apply context mask: zero out pixels not in the context region."
apply_context_mask(x::Matrix{Float32}, ctx_mask::BitMatrix) = x .* Float32.(ctx_mask)

"Layer-normalize each spatial embedding over its feature dimension."
function feature_layernorm(h)
    μ = mean(h; dims = 3)
    σ² = mean(abs2, h .- μ; dims = 3)
    (h .- μ) ./ sqrt.(σ² .+ 1f-4)
end

"Extract target block representations from target encoder output."
extract_targets(target_repr::Array{Float32,3}, target_masks::Vector{BitMatrix}) =
    [target_repr .* Float32.(mask) for mask in target_masks]

# ── Loss ────────────────────────────────────────────────────────────────────
# L2 distance between predicted and target representations, averaged over
# target blocks and over pixels within each target block (Eq. from paper).

"Per-block L2 loss: mean ‖ŷ_j − y_j‖₂² over pixels j in the target block."
function block_loss(pred_block, target_block)
    # Both are (H, W, D); target_block is zeroed outside the mask
    diff = pred_block .- target_block
    in_target = any(target_block .!= 0f0; dims = 3)
    n_pixels = count(in_target)
    n_pixels > 0 || return 0f0
    return sum(abs2, diff) / Float32(n_pixels * size(diff, 3))
end

"Full I-JEPA loss: average L2 across all target blocks."
function loss(m::JEPA, x, target_masks, context_mask)
    # Forward context encoder on masked image
    x_ctx = apply_context_mask(x, context_mask)
    ctxt_repr = forward(m.context_encoder, x_ctx)

    # Forward target encoder on full image and normalize features, as in I-JEPA.
    target_repr = feature_layernorm(forward(m.target_encoder, x))

    # Extract target block representations
    targets = extract_targets(target_repr, target_masks)

    # Predict
    predictions = forward(m.predictor, ctxt_repr, target_masks)

    # Average L2 loss
    ℓ = 0f0
    for (p, t) in zip(predictions, targets)
        ℓ += block_loss(p, t)
    end
    return ℓ / Float32(length(target_masks))
end

# ── EMA update ──────────────────────────────────────────────────────────────
"Update target encoder as exponential moving average of context encoder."
function update_ema!(m::JEPA, rate::Float32 = 0.993f0)
    for field in propertynames(m.context_encoder)
        src = getproperty(m.context_encoder, field)
        dst = getproperty(m.target_encoder, field)
        (src isa AbstractArray) || continue
        dst .*= rate
        dst .+= (1f0 - rate) .* src
    end
    m
end

# ── Training ────────────────────────────────────────────────────────────────

"One stochastic gradient step for I-JEPA."
function step!(m::JEPA, x;
    num_targets = 4,
    target_scale = (0.15f0, 0.20f0),
    context_scale = (0.85f0, 0.95f0),
    aspect_max = 1.5f0,
    η = 1f-3,
    rng = Random.default_rng(),
    update_ema = true)

    H, W = size(x)
    # Treat each pixel as a "block" for fine-grained masking
    grid_h, grid_w = H, W

    target_masks, context_mask = sample_multiblock_mask(
        grid_h, grid_w;
        num_targets = num_targets,
        target_scale = target_scale,
        context_scale = context_scale,
        aspect_max = aspect_max,
        rng = rng,
    )

    (∇enc, ∇pred) = gradient(m.context_encoder, m.predictor) do ce, pr
        local m′ = JEPA(context_encoder = ce, predictor = pr, target_encoder = m.target_encoder)
        loss(m′, x, target_masks, context_mask)
    end

    # Only update context_encoder and predictor (not target_encoder)
    sgd!(m.context_encoder, ∇enc, η)
    sgd!(m.predictor, ∇pred, η)

    # EMA update for target encoder
    update_ema && update_ema!(m)

    return m
end

"Train I-JEPA for N epochs over a dataset."
function train!(model::JEPA, η, dataset, epochs::Int = 1;
    num_targets = 4,
    target_scale = (0.15f0, 0.20f0),
    context_scale = (0.85f0, 0.95f0),
    aspect_max = 1.5f0,
    rng = Random.default_rng())

    foldl(1:epochs; init = model) do m, _
        foldl(dataset; init = m) do θ, x₀
            step!(θ, x₀;
                num_targets = num_targets,
                target_scale = target_scale,
                context_scale = context_scale,
                aspect_max = aspect_max,
                η = η,
                rng = rng)
        end
    end
end

"Train I-JEPA for a time budget, completing full dataset passes."
function train!(model::JEPA, η, dataset, duration::Dates.Period;
    num_targets = 4,
    target_scale = (0.15f0, 0.20f0),
    context_scale = (0.85f0, 0.95f0),
    aspect_max = 1.5f0,
    rng = Random.default_rng())

    target_s = seconds(duration)
    t₀ = time()
    while true
        time() - t₀ >= target_s && break
        model = foldl(dataset; init = model) do θ, x₀
            step!(θ, x₀;
                num_targets = num_targets,
                target_scale = target_scale,
                context_scale = context_scale,
                aspect_max = aspect_max,
                η = η,
                rng = rng)
        end
    end
    model
end

# ── Evaluation helpers ──────────────────────────────────────────────────────

"Run full forward pass: predict target representations from context."
function predict(m::JEPA, x, target_masks, context_mask)
    x_ctx = apply_context_mask(x, context_mask)
    ctxt_repr = forward(m.context_encoder, x_ctx)
    forward(m.predictor, ctxt_repr, target_masks)
end

"Get target representations (ground truth) for the given masks."
function get_targets(m::JEPA, x, target_masks)
    target_repr = feature_layernorm(forward(m.target_encoder, x))
    extract_targets(target_repr, target_masks)
end
