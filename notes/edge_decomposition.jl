using DataFrames, CSV, Statistics, Plots, Printf

# ─────────────────────────────────────────────────────────────────────────────
# Query data: 3 months of SPY 1s candles (Q3 2024)
# ─────────────────────────────────────────────────────────────────────────────
sql = """
WITH base AS (
    SELECT 
        window_start, open, high, low, close, volume,
        avg(volume) OVER w AS avg_vol,
        stddevSamp(volume) OVER w AS std_vol,
        avg((high - low) / NULLIF(open, 0)) OVER w AS avg_range,
        stddevSamp((high - low) / NULLIF(open, 0)) OVER w AS std_range,
        stddevSamp(log(close / open)) OVER w AS std_log_ret,
        lead(close, 1) OVER (ORDER BY window_start) AS c1,
        lead(close, 5) OVER (ORDER BY window_start) AS c5
    FROM massive_candles
    WHERE duration = '1_second' AND ticker = 'SPY'
      AND window_start >= '2024-06-01' AND window_start < '2024-09-01'
      AND volume > 0 AND close > 0
    WINDOW w AS (ORDER BY window_start ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING)
)
SELECT 
    log(close / open) AS log_ret,
    (high - low) / NULLIF(open, 0) AS range_pct,
    volume,
    (volume - avg_vol) / NULLIF(std_vol, 0) AS vol_z,
    ((high - low) / NULLIF(open, 0) - avg_range) / NULLIF(std_range, 0) AS range_z,
    log(close / open) / NULLIF(std_log_ret, 0) AS ret_z,
    CASE WHEN c1 > 0 THEN log(c1 / close) ELSE 0.0 END AS ret_1s,
    CASE WHEN c5 > 0 THEN log(c5 / close) ELSE 0.0 END AS ret_5s
FROM base
WHERE avg_vol > 0 AND std_vol > 0 AND avg_range > 0 AND std_range > 0 AND std_log_ret > 0
  AND c1 > 0 AND c5 > 0
  AND abs(log(close / open)) > 0.00001
ORDER BY window_start
FORMAT CSVWithNames
"""

println("Querying ClickHouse...")
df = CSV.read(pipeline(`clickhouse-client --query $sql`), DataFrame)
println("Loaded $(nrow(df)) rows")

# Clean
df = filter(r -> all(isfinite, [r.log_ret, r.range_pct, r.vol_z, r.range_z, r.ret_z, r.ret_1s, r.ret_5s]), df)
df.dir_same = sign.(df.log_ret) .== sign.(df.ret_1s)
println("Non-finite rows removed, $(nrow(df)) remaining")
println("Overall 1s persistence (non-zero moves): $(round(100 * mean(df.dir_same), digits=1))%")

# ─────────────────────────────────────────────────────────────────────────────
# Bucket helper
# ─────────────────────────────────────────────────────────────────────────────
function bucket_1d(df, col; nbins=20)
    f = filter(r -> abs(r.ret_1s) > 0.00001, df)[!, col]
    lo, hi = quantile(f, [0.02, 0.98])
    edges = range(lo, hi, nbins+1)
    bins = [findlast(e -> v >= e, edges) for v in f]
    
    results = []
    for b in 1:nbins
        valid = filter(r -> abs(r.ret_1s) > 0.00001, df)
        rows = valid[bins .== b, :]
        n = nrow(rows)
        n < 50 && continue
        mid = 0.5*(edges[b]+edges[b+1])
        same = count(rows.dir_same)
        opp = n - same
        push!(results, (mid=mid, n=n, persist=same/n, se=sqrt(same*opp)/n^(3/2),
                        net1=1e4*mean(rows.ret_1s)))
    end
    DataFrame(results)
end

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Direction Persistence vs Volume Z-score
# ─────────────────────────────────────────────────────────────────────────────
b1 = bucket_1d(df, :vol_z)
p1 = plot(b1.mid, b1.persist; seriestype=:scatter, yerror=b1.se,
          color=:blue, ms=5, msw=0, label="persistence (same direction →)",
          xlabel="Vol Z-score (std from rolling mean)",
          ylabel="Persistence (next 1s same direction)",
          title="Edge vs Volume Anomaly",
          ylims=(0.35, 0.55), legend=:bottomleft)
hline!([0.5]; c=:black, ls=:dash, label="50/50 (no edge)")
vline!([0]; c=:gray, ls=:dot, label="mean vol")
annotate!([(3.5, 0.53, text("↑ high volume → reversion", 9)),
           (-0.6, 0.53, text("↓ low volume → reversion", 9, :right))])

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Persistence vs Return Z-score
# ─────────────────────────────────────────────────────────────────────────────
b2 = bucket_1d(df, :ret_z)
p2 = plot(b2.mid, b2.persist; seriestype=:scatter, yerror=b2.se,
          color=:red, ms=5, msw=0, label="persistence",
          xlabel="Return Z-score (σ of this second's move)",
          ylabel="Persistence (next 1s same direction)",
          title="Edge vs Move Size",
          ylims=(0.35, 0.55), legend=:bottomleft)
hline!([0.5]; c=:black, ls=:dash, label="50/50")
vline!([0]; c=:gray, ls=:dot, label="mean move")
annotate!([(3.5, 0.53, text("↑ big move → reversion", 9)),
           (-3.5, 0.53, text("↓ big move → reversion", 9, :right))])

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Persistence vs Range Z-score
# ─────────────────────────────────────────────────────────────────────────────
b3 = bucket_1d(df, :range_z)
p3 = plot(b3.mid, b3.persist; seriestype=:scatter, yerror=b3.se,
          color=:green, ms=5, msw=0, label="persistence",
          xlabel="Range Z-score (σ of high-low width)",
          ylabel="Persistence (next 1s same direction)",
          title="Edge vs Candle Range (high-low)",
          ylims=(0.35, 0.55), legend=:bottomleft)
hline!([0.5]; c=:black, ls=:dash, label="50/50")
vline!([0]; c=:gray, ls=:dot, label="mean range")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Persistence by combined conditions (bar chart)
# ─────────────────────────────────────────────────────────────────────────────
function combo_persist(df, label, cond)
    rows = filter(r -> abs(r.ret_1s) > 0.00001 && cond(r), df)
    n = nrow(rows)
    n < 20 && return nothing
    same = count(rows.dir_same)
    (label=label, n=n, persist=same/n, se=sqrt(same*(n-same))/n^(3/2), 
     net1=1e4*mean(rows.ret_1s))
end

combos = filter(!isnothing, [
    combo_persist(df, "all data", r -> true),
    combo_persist(df, "high vol only", r -> r.vol_z > 1),
    combo_persist(df, "big range only", r -> r.range_z > 1),
    combo_persist(df, "big move only", r -> abs(r.ret_z) > 1),
    combo_persist(df, "high vol + big range", r -> r.vol_z > 1 && r.range_z > 1),
    combo_persist(df, "high vol + big move", r -> r.vol_z > 1 && abs(r.ret_z) > 1),
    combo_persist(df, "big range + big move", r -> r.range_z > 1 && abs(r.ret_z) > 1),
    combo_persist(df, "ALL THREE", r -> r.vol_z > 1 && r.range_z > 1 && abs(r.ret_z) > 1),
    combo_persist(df, "opposite (low vol,\n small range, small move)", 
                  r -> r.vol_z < -0.5 && r.range_z < -0.5 && abs(r.ret_z) < 0.5),
])
combo_df = DataFrame(combos)
sort!(combo_df, :persist)

p4 = bar(1:nrow(combo_df), combo_df.persist; yerror=combo_df.se,
         color=[c == "ALL THREE" ? :red : :steelblue for c in combo_df.label],
         legend=false, xrotation=30,
         xticks=(1:nrow(combo_df), combo_df.label),
         ylabel="Persistence (next 1s)", ylims=(0.3, 0.65),
         title="Persistence by Combined Conditions")
hline!([0.5]; c=:black, ls=:dash, label="50/50")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Net forward return by Vol Z-score
# ─────────────────────────────────────────────────────────────────────────────
p5 = plot(b1.mid, b1.net1; seriestype=:scatter, c=:blue, ms=5, msw=0,
          label="avg next-1s return (bps)",
          xlabel="Volume Z-score", ylabel="Net forward return (bps)",
          title="Forward Return by Volume Anomaly")
hline!([0]; c=:black, ls=:dash, label="zero")
vline!([0]; c=:gray, ls=:dot, label="mean vol")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Persistence vs raw |log return| (log x scale)
# ─────────────────────────────────────────────────────────────────────────────
b6 = bucket_1d(df, :log_ret)
b6.abs_mid = abs.(b6.mid)
sort!(b6, :abs_mid)
p6 = plot(b6.abs_mid, b6.persist; seriestype=:scatter, yerror=b6.se,
          c=:purple, ms=5, msw=0, label="persistence",
          xscale=:log10, ylims=(0.35, 0.55), legend=:bottomleft,
          xlabel="|Log return| this second",
          ylabel="Persistence (next 1s same direction)",
          title="Edge vs Raw Move Size")
hline!([0.5]; c=:black, ls=:dash, label="50/50")

# ─────────────────────────────────────────────────────────────────────────────
# Display all 6 plots
# ─────────────────────────────────────────────────────────────────────────────
p_all = plot(p1, p2, p3, p4, p5, p6; layout=(3, 2), size=(1400, 1200))
savefig(p_all, joinpath(@__DIR__, "edge_decomposition.png"))
println("\nDone — saved to notes/edge_decomposition.png")
