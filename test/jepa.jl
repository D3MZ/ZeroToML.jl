# This is testing I-JEPA: https://arxiv.org/abs/2301.08243
#
# I-JEPA learns semantic image representations by predicting target block
# representations from a context block.  Unlike MAE (pixel reconstruction),
# I-JEPA predicts in *representation* space, which pushes the model to learn
# higher-level features without hand-crafted data augmentations.
#
# Key differences from the official FAIR implementation (facebookresearch/jepa):
#   - We use a simple conv encoder/predictor instead of Vision Transformers
#   - Grid-level masking (each pixel = one "block") instead of patch-level
#   - No distributed training, no mixed precision, no AdamW
#   - Target scale and context scale adapted for small 12×12 images

ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random, Dates
using Statistics
using Plots

@testset "I-JEPA" begin
    "Generate all possible h×w boxes (filled with +1f0s) in a H×W grid of -1f0s."
    boxes(H = 12, W = 12, h = 3, w = 3) = [
        (g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1f0; g)
        for i in 1:H-h+1 for j in 1:W-w+1
    ]

    "Zero-centered Pearson correlation between two arrays."
    center(x) = x .- mean(x)
    correlate(x, y) = sum(center(x) .* center(y)) /
        (sqrt(sum(abs2, center(x)) * sum(abs2, center(y))) + eps())

    "Nearest-neighbor correlation: best match against the dataset."
    nearest_correlation(sample, dataset) = maximum(correlate(sample, x) for x in dataset)

    "Visualize a sample as a grayscale heatmap."
    panel(title, sample) = heatmap(
        sample;
        title = title,
        color = :grays,
        clims = (-1, 1),
        axis = false,
        colorbar = false,
        aspect_ratio = :equal,
    )

    Random.seed!(1)

    H, W = 12, 12
    h, w = 3, 3
    η = 5f-2
    num_targets = 2  # fewer targets for small images
    # Small target blocks (2-4% of image = 3-6 cells) so they fit the 3×3 box.
    # High masking for the visual example: only about 10% of pixels are visible.
    target_scale = (0.02f0, 0.04f0)
    context_scale = (0.10f0, 0.12f0)
    rng = MersenneTwister(1)
    dataset = shuffle(rng, boxes(H, W, h, w))

    # ── Sample a visual example ─────────────────────────────────────────
    x₀ = rand(rng, dataset)

    # Use content-agnostic random targets and a 90%-masked random-pixel context.
    # This is intentionally *not* chosen to line up with the object.
    target_masks, _ = sample_multiblock_mask(
        H, W;
        num_targets = num_targets,
        target_scale = target_scale,
        context_scale = context_scale,
        rng = rng,
    )
    combined_target_mask = reduce(.|, target_masks)
    context_mask = (rand(rng, Float32, H, W) .< 0.10f0) .& .!combined_target_mask
    if !any(context_mask)
        candidates = findall(.!combined_target_mask)
        context_mask[rand(rng, candidates)] = true
    end

    x_ctx = apply_context_mask(x₀, context_mask)

    # ── Training ────────────────────────────────────────────────────────

    Random.seed!(42)
    model = JEPA()

    # Use a separate random-block mask for quantitative checks so the visual
    # 90%-random-pixel mask stays purely illustrative.
    eval_target_masks, eval_context_mask = sample_multiblock_mask(
        H, W;
        num_targets = num_targets,
        target_scale = target_scale,
        context_scale = (0.85f0, 0.95f0),
        rng = MersenneTwister(11),
    )

    # Untrained loss
    untrained_loss = loss(model, x₀, eval_target_masks, eval_context_mask)

    # Train with the same masking distribution used for evaluation.
    model = train!(model, η, dataset, 3;
        num_targets = num_targets,
        target_scale = target_scale,
        context_scale = (0.85f0, 0.95f0),
        rng = MersenneTwister(2))

    # Trained loss (same sample & masks)
    trained_loss = loss(model, x₀, eval_target_masks, eval_context_mask)

    # ── Evaluate predictions ────────────────────────────────────────────
    predictions = predict(model, x₀, eval_target_masks, eval_context_mask)
    targets = get_targets(model, x₀, eval_target_masks)

    # Compute per-block correlations (prediction vs target representation)
    block_correlations = [correlate(p[:], t[:]) for (p, t) in zip(predictions, targets)]
    mean_block_corr = mean(block_correlations)

    # Also check untrained correlations for comparison
    Random.seed!(42)
    untrained_model = JEPA()
    untrained_predictions = predict(untrained_model, x₀, eval_target_masks, eval_context_mask)
    untrained_block_corrs = [
        correlate(p[:], t[:]) for (p, t) in zip(untrained_predictions, targets)
    ]
    mean_untrained_corr = mean(untrained_block_corrs)

    # Compare against the previous conv-only setup at its original LR.
    Random.seed!(42)
    conv_model = train!(ConvJEPA(), 1f-2, dataset, 3;
        num_targets = num_targets,
        target_scale = target_scale,
        context_scale = (0.85f0, 0.95f0),
        rng = MersenneTwister(2))
    conv_loss = loss(conv_model, x₀, eval_target_masks, eval_context_mask)
    conv_predictions = predict(conv_model, x₀, eval_target_masks, eval_context_mask)
    conv_targets = get_targets(conv_model, x₀, eval_target_masks)
    conv_corr = mean(correlate(p[:], t[:]) for (p, t) in zip(conv_predictions, conv_targets))

    # ── Visualize ───────────────────────────────────────────────────────
    # I-JEPA predicts in representation space, not pixel space.
    # We visualize: masks, encoder representations, and prediction accuracy.

    # Combine target masks for visualization
    combined_target_mask = falses(H, W)
    for m in target_masks
        combined_target_mask .|= m
    end

    # Context encoder representation (scalar magnitude for visualization)
    x_ctx = apply_context_mask(x₀, context_mask)
    ctxt_repr = forward(model.context_encoder, x_ctx)
    ctxt_mag = sqrt.(sum(abs2, ctxt_repr; dims=3)[:, :, 1])

    # Target encoder representation
    target_repr = forward(model.target_encoder, x₀)
    target_mag = sqrt.(sum(abs2, target_repr; dims=3)[:, :, 1])

    # Per-pixel prediction error for target 1 in representation space.
    # Use cosine distance for display: I-JEPA cares about matching embeddings,
    # and the target branch is feature-normalized like the reference code.
    visual_predictions = predict(model, x₀, target_masks, context_mask)
    visual_targets = get_targets(model, x₀, target_masks)
    pred1 = visual_predictions[1]
    tgt1 = visual_targets[1]
    target1_mask = Float32.(target_masks[1])
    pred1_mag = sqrt.(sum(abs2, pred1; dims=3)[:, :, 1])
    tgt1_mag = sqrt.(sum(abs2, tgt1; dims=3)[:, :, 1])
    similarity1 = sum(pred1 .* tgt1; dims=3)[:, :, 1] ./
        (pred1_mag .* tgt1_mag .+ eps(Float32))
    error1 = (1f0 .- similarity1) .* target1_mask

    # Show what JEPA is trying to match: predicted vs target embedding
    # magnitude only at the hidden target cells. Normalize each panel to its
    # own target-cell maximum so matched structure appears with matched color;
    # absolute magnitude differences are already captured by the cosine-error
    # and loss tests rather than this qualitative panel.
    pred1_visible = pred1_mag .* target1_mask
    tgt1_visible = tgt1_mag .* target1_mask
    pred1_visible ./= maximum(pred1_visible) + eps(Float32)
    tgt1_visible ./= maximum(tgt1_visible) + eps(Float32)

    # Normalize each panel for display
    scale(v) = (v .- minimum(v)) ./ (maximum(v) - minimum(v) + eps(Float32))

    # Panels: pixel images use clims=(-1,1), representation panels don't
    panel_repr(title, v; clims = nothing) = heatmap(
        v; title = title, color = :viridis, axis = false,
        colorbar = false, aspect_ratio = :equal, clims = clims,
    )

    figure = plot(
        panel("original", x₀),
        panel("context (masked)", x_ctx),
        panel("target mask", Float32.(combined_target_mask)),
        panel("pred box (tgt 1)", 2f0 .* pred1_visible .- 1f0),
        panel_repr("context repr", scale(ctxt_mag)),
        panel_repr("target repr", scale(target_mag)),
        panel_repr("cos error (tgt 1)", error1; clims = (0, 1)),
        panel("true box (tgt 1)", 2f0 .* tgt1_visible .- 1f0),
        layout = (2, 4),
        size = (800, 400),
    )

    output_dir = joinpath(@__DIR__, "outputs")
    mkpath(output_dir)
    path = joinpath(output_dir, "jepa_samples.png")
    savefig(figure, path)

    @test trained_loss < untrained_loss
    @test mean_block_corr > mean_untrained_corr
    @test trained_loss < conv_loss
    @test mean_block_corr > conv_corr
    @debug "I-JEPA" untrained_loss = untrained_loss trained_loss = trained_loss conv_loss = conv_loss
    @debug "I-JEPA correlations" untrained = mean_untrained_corr trained = mean_block_corr conv = conv_corr
end
