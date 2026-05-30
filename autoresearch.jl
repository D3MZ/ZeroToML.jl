using ZeroToML, Random, Statistics, BlackBoxOptim, Dates

boxes(H=12, W=12, h=3, w=3) = [(g = -ones(Float32, H, W); g[i:i+h-1, j:j+w-1] .= 1f0; g) for i in 1:H-h+1 for j in 1:W-w+1]
center(x) = x .- mean(x)
correlate(x, y) = sum(center(x) .* center(y)) / (sqrt(sum(abs2, center(x)) * sum(abs2, center(y))) + eps())

Random.seed!(1)
H, W = 12, 12
h, w = 3, 3
t = 1f0
rng = MersenneTwister(1)
dataset = shuffle(rng, boxes(H, W, h, w))
x₀ = rand(rng, dataset);
ε = randn(rng, Float32, size(x₀));

function evaluate(x)
    η = exp10(x[1])
    βmax = exp(x[2])
    βmin = exp(x[3])
    steps = exp(x[4])
    βmin = clamp(βmin, 0.01f0, βmax * 0.5f0)
    steps = clamp(round(Int, steps), 50, 200)

    sde = VPSDE(βmin=Float32(βmin), βmax=Float32(βmax))
    input = perturbed_sample(sde, x₀, t, ε)
    model = train!(ScoreSDE(), sde, Float32(η), dataset, Second(10))
    denoised = clamp.(probability_flow_sample(model, sde, input, t; steps=steps), -1f0, 1f0)
    -correlate(x₀, denoised)
end

res = bboptimize(evaluate;
    SearchRange = [(-5.0, 1.0), (-1.0, 3.0), (-4.0, 0.0), (3.0, 5.5)],
    Method = :de_rand_1_bin,
    PopulationSize = 10,
    MaxFuncEvals = 50,
    TraceMode = :silent
)

best_val = -best_fitness(res)
best_x = best_candidate(res)
best_η = exp10(best_x[1])
best_βmax = exp(best_x[2])
best_βmin = exp(best_x[3])
best_steps = round(Int, exp(best_x[4]))
println("METRIC denoised_correlation=$best_val")
println("BEST η=$best_η βmax=$best_βmax βmin=$best_βmin steps=$best_steps")
