using ZeroToML
using Random, Plots

@fastmath function normalize_frame(frame)
    pixels = reshape(frame, 32, 32, 3)
    @views gray = 0.2989f0 .* pixels[:, :, 1] .+ 0.5870f0 .* pixels[:, :, 2] .+ 0.1140f0 .* pixels[:, :, 3]
    2f0 .* gray .- 1f0
end

function load_cifar_batches(dir; limit=512)
    files = sort(filter(name -> startswith(name, "data_batch"), readdir(dir)))
    samples = Matrix{Float32}[]
    record = 3073
    for name in files
        raw = read(joinpath(dir, name))
        count = length(raw) ÷ record
        data = reshape(raw, record, count)
        payload = Float32.(data[2:end, :]) ./ 255f0
        for idx in axes(payload, 2)
            push!(samples, normalize_frame(@view payload[:, idx]))
            if length(samples) == limit
                return samples
            end
        end
    end
    samples
end

dataset_dir = joinpath(@__DIR__, "cifar-10-batches-bin")
dataset = load_cifar_batches(dataset_dir; limit=512*100)
shuffle!(dataset)
@info "Loaded $(length(dataset)) CIFAR-10 samples"
display(heatmap(first(dataset); c=:grays, colorbar=false, frame=false, aspect_ratio=1, title="Dataset sample", xticks=false, yticks=false))

T = 1_000
β = noise_schedule(T)
α = signal_schedule(β)
ᾱ = remaining_signal(α)
time_embedding = ᾱ

model = DDPM()
η = 5f-4
epochs = 3
@info "Training for $(epochs) epochs"
model = train!(model, ᾱ, T, η, dataset, time_embedding, epochs)

image = first(dataset)
d = length(image)

sample = reverse_sample(model, β, α, ᾱ, T, d, time_embedding)
display(heatmap(sample; c=:grays, colorbar=false, frame=false, aspect_ratio=1, xticks=false, yticks=false))
