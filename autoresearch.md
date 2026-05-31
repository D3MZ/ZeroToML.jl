# Autoresearch: Flow Matching Time-Series Forecasting

## Objective
Improve the toy Conditional Flow Matching forecast for NVDA daily OHLC bars in `test/flow_matching_timeseries.jl`. The held-out workload forecasts the last 10 available NVDA daily bars from the preceding 30 bars. The user wants visible, unique candle-chart images for each major idea and asked for best-effort large ideas without benchmark cheating.

## Metrics
- **Primary**: `close_mape` (%, lower is better) — MAPE of predicted close prices over the 10-day held-out forecast.
- **Secondary**: `close_mae` ($, lower), `direction_accuracy` (%, higher), `trained_loss` (lower), `untrained_loss`.

## How to Run
`./autoresearch.sh` — runs `test/flow_matching_timeseries.jl` with `AUTORESEARCH=1`, writes a unique image label, and outputs `METRIC name=value` lines.

## Files in Scope
- `test/flow_matching_timeseries.jl` — time-series Flow Matching model, training, plotting, metrics.
- `test/data/nvda_daily.csv` — frozen local NVDA daily bars through latest available bar; may be refreshed only if explicitly needed, not during optimization.
- `test/runtests.jl` — includes time-series test.
- `autoresearch.sh`, `autoresearch.md`, `autoresearch.ideas.md` — experiment harness and notes.

## Off Limits
- Do not tune directly on the held-out future values by leaking them into inputs or training windows.
- Do not reduce the forecast horizon below 10 or move the split to an easier date.
- Do not change the metric definition to make results look better.
- Do not fetch newer/different data mid-loop to cherry-pick a better target.
- Do not plot smoothed actuals or otherwise hide forecast errors.

## Constraints
- Keep this as a Flow Matching forecast test.
- No benchmark cheating: training windows must end before the held-out 10-day future.
- Prefer five large, interpretable ideas over tiny manual knob twiddling.
- If parameter optimization is used, use BlackBox-style algorithm over several fields rather than manual one-at-a-time tuning.
- Produce unique images per kept/major experiment via `FM_TS_LABEL`.
- Full test suite should remain passable; autoresearch primary command is the focused time-series test.

## Initial Baseline
Current model is a tiny MLP velocity model on normalized OHLC log returns:
- context: 30 days × 4 OHLC returns
- forecast: 10 days × 4 OHLC returns
- path: OTFlowPath, Gaussian prior to future return window
- model: Dense 161→64→64→40 ReLU MLP
- forecast: mean of 64 sampled trajectories
Prior observed approximate metrics: close MAPE ~5%, direction accuracy ~30%, trained loss improves only slightly.

## Five Large Ideas To Try
1. Predict residual future around a drift/baseline instead of raw normalized returns, then add baseline back before plotting.
2. Add a stronger conditional architecture (wider residual MLP / skip features / context summary) while keeping Flow Matching objective.
3. Use BlackBoxOptim to jointly tune context length, learning rate, train steps, sample count, hidden width, and residual scale without touching split/horizon/noise.
4. Add walk-forward validation over several pre-holdout windows and optimize validation average, then report the final held-out image to reduce overfit.
5. Forecast close returns first and derive OHLC around close, or train separate close-focused weighted loss while still plotting full candles.

## What's Been Tried
- Baseline tiny MLP with direct normalized OHLC-return future prediction and 64-sample forecast mean. It trains but visual forecast drifts upward and misses held-out downturn.
