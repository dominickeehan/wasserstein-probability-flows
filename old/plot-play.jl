using Random, Distributions
using Plots


Random.seed!(42)

shift_distribution = Normal(0, 100)

history_length = 50

distributions = [Normal(0,1) for _ in 1:history_length]

μ = 0
σ = 60

for t in 1:history_length
    distributions[t] = Normal(μ, σ)

    μ = μ + (rand(shift_distribution))
end

X = LinRange(-500, 500, 100)

plt = plot()

for t in 1:history_length; plot!(X, [pdf(distributions[t], x) for x in X].-0.001*t, color=:red, alpha=(t/history_length), label=nothing); end

display(plt)