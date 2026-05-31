# Conditional Flow Matching on NVDA daily OHLC bars for toy time-series forecasting.
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

function read_nvda_bars(path=joinpath(@__DIR__, "data", "nvda_daily.csv"))
    rows = split.(readlines(path)[2:end], ',')
    [DailyBar(row[1], parse.(Float32, row[2:5])...) for row in rows]
end

prices(bar::DailyBar) = Float32[bar.open, bar.high, bar.low, bar.close]

function ohlc_returns(bars)
    values = reduce(hcat, prices.(bars))'
    Float32.(diff(log.(values); dims=1))
end

@kwdef struct TimeSeriesFlow
    W₁ = glorot(128, 173)
    b₁ = zeros(Float32, 128)
    W₂ = glorot(128, 128)
    b₂ = zeros(Float32, 128)
    W₃ = glorot(128, 128)
    b₃ = zeros(Float32, 128)
    W₄ = glorot(40, 128)
    b₄ = zeros(Float32, 40)
end

predict(m::TimeSeriesFlow, x) = m.W₄ * relu(m.W₃ * relu(m.W₂ * relu(m.W₁ * x .+ m.b₁) .+ m.b₂) .+ m.b₃) .+ m.b₄

function velocity(m::TimeSeriesFlow, context, xt, t)
    summary = vcat(vec(mean(context; dims=1)), vec(std(context; dims=1)), vec(context[end, :]))
    reshape(predict(m, vcat(vec(context), vec(xt), Float32[t], summary)), size(xt))
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

function add_candles!(p, bars; color)
    for i in eachindex(bars.closes)
        plot!(p, [i, i], [bars.lows[i], bars.highs[i]]; color, linewidth=1, label=false)
        top = max(bars.opens[i], bars.closes[i])
        bottom = min(bars.opens[i], bars.closes[i])
        body = Shape([i - 0.25, i + 0.25, i + 0.25, i - 0.25], [bottom, bottom, top, top])
        plot!(p, body; color, opacity=0.35, linecolor=color, label=false)
    end
    p
end

function candle_panel(title, actual, forecasted)
    p = plot(title=title, xlabel="forecast day", ylabel="price", legend=:topright)
    add_candles!(p, actual; color=:black)
    add_candles!(p, forecasted; color=:blue)
    plot!(p, actual.closes; color=:black, linewidth=2, label="actual close")
    plot!(p, forecasted.closes; color=:blue, linewidth=2, label="forecast close")
    p
end

function run_flow_matching_timeseries(; image_label=get(ENV, "FM_TS_LABEL", "latest"), train_steps=5_000, η=1f-3, forecast_samples=64)
    Random.seed!(1)
    context_len = 30
    horizon = 10
    feature_count = 4
    bars = read_nvda_bars()
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
    actual = returns_to_bars(last_prices, actual_returns)
    forecasted = returns_to_bars(last_prices, predicted_returns)

    close_mae = mean(abs.(forecasted.closes .- actual.closes))
    close_mape = mean(abs.((forecasted.closes .- actual.closes) ./ actual.closes)) * 100
    direction_accuracy = mean(sign.(diff([last_prices[4]; forecasted.closes])) .== sign.(diff([last_prices[4]; actual.closes]))) * 100

    figure = candle_panel("NVDA Flow Matching forecast vs actual", actual, forecasted)
    image_path = joinpath(@__DIR__, "flow_matching_timeseries_nvda_$(image_label).png")
    savefig(figure, image_path)
    savefig(figure, joinpath(@__DIR__, "flow_matching_timeseries_nvda.png"))

    (; untrained_loss, trained_loss, close_mae, close_mape, direction_accuracy, image_path)
end

@testset "Flow Matching Time Series" begin
    metrics = run_flow_matching_timeseries()
    @test isfinite(metrics.trained_loss)
    @test isfinite(metrics.close_mape)
    @test metrics.close_mape < 20
    if get(ENV, "AUTORESEARCH", "0") == "1"
        println("METRIC close_mape=$(metrics.close_mape)")
        println("METRIC close_mae=$(metrics.close_mae)")
        println("METRIC direction_accuracy=$(metrics.direction_accuracy)")
        println("METRIC trained_loss=$(metrics.trained_loss)")
        println("METRIC untrained_loss=$(metrics.untrained_loss)")
        println("IMAGE $(metrics.image_path)")
    end
end
