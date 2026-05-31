# Conditional Flow Matching on daily OHLC bars for toy time-series forecasting.
ENV["GKSwstype"] = "100"

using ZeroToML
using Test
using Random
using Statistics
using Zygote
using Plots

struct DailyBar
    date::String
    open::Float32
    high::Float32
    low::Float32
    close::Float32
end

struct TimeSeriesAsset
    symbol::String
    sector::String
    name::String
end

const TIMESERIES_ASSETS = [
    TimeSeriesAsset("NVDA", "Technology", "NVIDIA"),
    TimeSeriesAsset("LLY", "Health Care", "Eli Lilly"),
    TimeSeriesAsset("WMT", "Consumer Staples", "Walmart"),
    TimeSeriesAsset("XOM", "Energy", "Exxon Mobil"),
    TimeSeriesAsset("NEE", "Utilities", "NextEra Energy"),
]

asset_path(asset::TimeSeriesAsset) = joinpath(@__DIR__, "data", "$(lowercase(asset.symbol))_daily.csv")

function read_daily_bars(path)
    rows = split.(readlines(path)[2:end], ',')
    [DailyBar(row[1], parse.(Float32, row[2:5])...) for row in rows]
end

prices(bar::DailyBar) = Float32[bar.open, bar.high, bar.low, bar.close]

function ohlc_returns(bars)
    values = reduce(hcat, prices.(bars))'
    Float32.(diff(log.(values); dims=1))
end

close_returns(asset::TimeSeriesAsset) = ohlc_returns(read_daily_bars(asset_path(asset)))[:, 4]

@kwdef struct TimeSeriesFlow
    W₁ = glorot(128, 139)
    b₁ = zeros(Float32, 128)
    W₂ = glorot(128, 128)
    b₂ = zeros(Float32, 128)
    W₃ = glorot(4, 128)
    b₃ = zeros(Float32, 4)
end

function predict_day(m::TimeSeriesFlow, context_features, xt_day, t, day, horizon)
    τ = Float32(day / horizon)
    h = relu(m.W₁ * vcat(context_features, xt_day, Float32[t, τ, τ^2]) .+ m.b₁)
    h = relu(m.W₂ * h .+ m.b₂)
    m.W₃ * h .+ m.b₃
end

function velocity(m::TimeSeriesFlow, context, xt, t)
    summary = vcat(vec(mean(context; dims=1)), vec(std(context; dims=1)), vec(context[end, :]))
    context_features = vcat(vec(context), summary)
    rows = [predict_day(m, context_features, vec(xt[day, :]), t, day, size(xt, 1)) for day in 1:size(xt, 1)]
    reduce(hcat, rows)'
end

function flow_loss(m::TimeSeriesFlow, path::OTFlowPath, context, x₀, x₁, t)
    xt = flow_sample(path, x₀, x₁, t)
    ut = flow_target(path, x₀, x₁)
    mean((velocity(m, context, xt, t) .- ut).^2)
end

function train_timeseries_flow!(model, path, windows; steps=1_500, η=3f-3, rng=MersenneTwister(1))
    for _ in 1:steps
        context, future = rand(rng, windows)
        x₀ = randn(rng, Float32, size(future))
        t = clamp(rand(rng, Float32), 1f-3, 1f0)
        (∇,) = gradient(θ -> flow_loss(θ, path, context, x₀, future, t), model)
        ZeroToML.sgd!(model, ∇, η)
    end
    model
end

function forecast_sample(model, context; horizon=10, features=4, steps=100, rng=MersenneTwister(2))
    sample = randn(rng, Float32, horizon, features)
    Δt = 1f0 / steps
    foldl(1:steps; init=sample) do xt, step
        t = Float32((step - 1) * Δt)
        xt .+ Δt .* velocity(model, context, xt, t)
    end
end

function forecast_mean(model, context; samples=64, horizon=10, features=4, steps=100, rng=MersenneTwister(2))
    forecasts = [forecast_sample(model, context; horizon, features, steps, rng) for _ in 1:samples]
    stacked = cat(forecasts...; dims=3)
    Float32.(dropdims(median(stacked; dims=3); dims=3))
end

function returns_to_bars(last_prices, returns)
    values = reshape(last_prices, 1, :) .* exp.(cumsum(returns; dims=1))
    opens = values[:, 1]
    highs = max.(values[:, 2], values[:, 1], values[:, 4])
    lows = min.(values[:, 3], values[:, 1], values[:, 4])
    closes = values[:, 4]
    (; opens, highs, lows, closes)
end

bars_to_ohlc(bars) = (;
    opens=[bar.open for bar in bars],
    highs=[bar.high for bar in bars],
    lows=[bar.low for bar in bars],
    closes=[bar.close for bar in bars],
)

function add_candles!(p, bars; color, xs=eachindex(bars.closes))
    for (x, i) in zip(xs, eachindex(bars.closes))
        plot!(p, [x, x], [bars.lows[i], bars.highs[i]]; color, linewidth=1, label=false)
        top = max(bars.opens[i], bars.closes[i])
        bottom = min(bars.opens[i], bars.closes[i])
        body = Shape([x - 0.15, x + 0.15, x + 0.15, x - 0.15], [bottom, bottom, top, top])
        plot!(p, body; color, opacity=0.35, linecolor=color, label=false)
    end
    p
end

function candle_panel(title, history, actual, forecasted)
    history_xs = -length(history.closes)+1:0
    forecast_xs = 1:length(actual.closes)
    p = plot(title=title, xlabel="days from forecast start", ylabel="price", legend=:outertopright, right_margin=8Plots.mm)
    add_candles!(p, history; color=:gray, xs=history_xs)
    add_candles!(p, actual; color=:black, xs=forecast_xs)
    add_candles!(p, forecasted; color=:blue, xs=forecast_xs)
    plot!(p, history_xs, history.closes; color=:gray, linewidth=2, label="prior actual close")
    plot!(p, forecast_xs, actual.closes; color=:black, linewidth=2, label="actual close")
    plot!(p, forecast_xs, forecasted.closes; color=:blue, linewidth=2, label="forecast close")
    vline!(p, [0]; color=:gray, linestyle=:dash, label=false)
    p
end

function run_flow_matching_timeseries(asset=TIMESERIES_ASSETS[1]; image_label=get(ENV, "FM_TS_LABEL", "latest"), train_steps=5_000, η=1f-3, forecast_samples=64)
    Random.seed!(1)
    context_len = 30
    horizon = 10
    feature_count = 4
    bars = read_daily_bars(asset_path(asset))
    returns = ohlc_returns(bars)
    μ = mean(returns; dims=1)
    σ = std(returns; dims=1) .+ 1f-6
    normalized = Float32.((returns .- μ) ./ σ)

    split = size(normalized, 1) - horizon
    windows = [(normalized[i:i+context_len-1, :], normalized[i+context_len:i+context_len+horizon-1, :]) for i in 1:split-context_len-horizon]
    context = normalized[split-context_len:split-1, :]
    future = normalized[split:split+horizon-1, :]
    path = OTFlowPath(σmin=1f-4)
    model = TimeSeriesFlow()
    x₀ = randn(MersenneTwister(3), Float32, horizon, feature_count)
    untrained_loss = flow_loss(model, path, context, x₀, future, 0.5f0)
    model = train_timeseries_flow!(model, path, windows; steps=train_steps, η=η)
    trained_loss = flow_loss(model, path, context, x₀, future, 0.5f0)
    predicted = forecast_mean(model, context; samples=forecast_samples, horizon, features=feature_count)

    last_prices = prices(bars[split])
    actual_returns = Float32.(future .* σ .+ μ)
    predicted_returns = Float32.(predicted .* σ .+ μ)
    history = bars_to_ohlc(bars[split-19:split])
    actual = returns_to_bars(last_prices, actual_returns)
    forecasted = returns_to_bars(last_prices, predicted_returns)

    close_mae = mean(abs.(forecasted.closes .- actual.closes))
    close_mape = mean(abs.((forecasted.closes .- actual.closes) ./ actual.closes)) * 100
    direction_accuracy = mean(sign.(diff([last_prices[4]; forecasted.closes])) .== sign.(diff([last_prices[4]; actual.closes]))) * 100

    figure = candle_panel("$(asset.symbol) Flow Matching forecast vs actual", history, actual, forecasted)
    output_dir = joinpath(@__DIR__, "outputs")
    mkpath(output_dir)
    image_path = joinpath(output_dir, "flow_matching_timeseries_$(lowercase(asset.symbol))_$(image_label).png")
    savefig(figure, image_path)
    if asset.symbol == "NVDA"
        savefig(figure, joinpath(output_dir, "flow_matching_timeseries_nvda.png"))
    end

    (; asset, untrained_loss, trained_loss, close_mae, close_mape, direction_accuracy, image_path)
end

@testset "Flow Matching Time Series" begin
    close_return_matrix = reduce(hcat, close_returns.(TIMESERIES_ASSETS))
    pairwise_correlations = [cor(close_return_matrix[:, i], close_return_matrix[:, j]) for i in 1:length(TIMESERIES_ASSETS) for j in i+1:length(TIMESERIES_ASSETS)]
    @test maximum(abs.(pairwise_correlations)) < 0.4

    metrics_by_asset = [run_flow_matching_timeseries(asset) for asset in TIMESERIES_ASSETS]
    for metrics in metrics_by_asset
        @test isfinite(metrics.trained_loss)
        @test isfinite(metrics.close_mape)
        @test metrics.close_mape < 20
    end
    metrics = first(metrics_by_asset)
    if get(ENV, "AUTORESEARCH", "0") == "1"
        println("METRIC close_mape=$(metrics.close_mape)")
        println("METRIC close_mae=$(metrics.close_mae)")
        println("METRIC direction_accuracy=$(metrics.direction_accuracy)")
        println("METRIC trained_loss=$(metrics.trained_loss)")
        println("METRIC untrained_loss=$(metrics.untrained_loss)")
        println("IMAGE $(metrics.image_path)")
    end
end
