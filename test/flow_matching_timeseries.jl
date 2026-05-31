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
    W₁ = glorot(64, 161)
    b₁ = zeros(Float32, 64)
    W₂ = glorot(64, 64)
    b₂ = zeros(Float32, 64)
    W₃ = glorot(40, 64)
    b₃ = zeros(Float32, 40)
end

predict(m::TimeSeriesFlow, x) = m.W₃ * relu(m.W₂ * relu(m.W₁ * x .+ m.b₁) .+ m.b₂) .+ m.b₃

function velocity(m::TimeSeriesFlow, context, xt, t)
    reshape(predict(m, vcat(vec(context), vec(xt), Float32[t])), size(xt))
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

function forecast(model, context; horizon=10, features=4, steps=100, rng=MersenneTwister(2))
    sample = randn(rng, Float32, horizon, features)
    Δt = 1f0 / steps
    foldl(1:steps; init=sample) do xt, step
        t = Float32((step - 1) * Δt)
        xt .+ Δt .* velocity(model, context, xt, t)
    end
end

function returns_to_bars(last_close, returns)
    closes = last_close .* exp.(cumsum(returns[:, 4]))
    opens = [last_close; closes[1:end-1]]
    highs = max.(opens, closes) .* exp.(abs.(returns[:, 2]) .* 0.25f0)
    lows = min.(opens, closes) .* exp.(-abs.(returns[:, 3]) .* 0.25f0)
    (; opens, highs, lows, closes)
end

function candle_panel(title, bars; color)
    n = length(bars.closes)
    p = plot(title=title, legend=false, xlabel="day", ylabel="price")
    for i in 1:n
        plot!(p, [i, i], [bars.lows[i], bars.highs[i]]; color, linewidth=1)
        plot!(p, [i - 0.3, i, i + 0.3], [bars.opens[i], bars.closes[i], bars.closes[i]]; color, linewidth=3)
    end
    p
end

@testset "Flow Matching Time Series" begin
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
    model = train_timeseries_flow!(model, path, windows)
    trained_loss = flow_loss(model, path, context, x₀, future, 0.5f0)
    predicted = forecast(model, context; horizon, features=feature_count)

    last_close = bars[split].close
    actual_returns = Float32.(future .* σ .+ μ)
    predicted_returns = Float32.(predicted .* σ .+ μ)
    actual = returns_to_bars(last_close, actual_returns)
    forecasted = returns_to_bars(last_close, predicted_returns)

    figure = plot(
        candle_panel("NVDA actual", actual; color=:black),
        candle_panel("Flow forecast", forecasted; color=:blue);
        layout=(2, 1), size=(700, 700)
    )
    savefig(figure, joinpath(@__DIR__, "flow_matching_timeseries_nvda.png"))

    @test trained_loss < untrained_loss
    @test all(isfinite, predicted)
end
