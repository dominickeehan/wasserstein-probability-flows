using Random, Statistics, StatsBase, Distributions

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
include("weights.jl")

Random.seed!(42)

shift_distribution = Normal(0, 1)

repetitions = 300
history_length = 20

value_sequences = [zeros(history_length+1) for _ in 1:repetitions]
value_distributions = [[Normal(0, 0) for _ in 1:history_length+1] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 100
    σ = 10

    for t in 1:history_length+1
        value_distributions[repetition][t] = Normal(μ, σ)
        value_sequences[repetition][t] = rand(value_distributions[repetition][t])

        μ = μ + rand(shift_distribution)
    end
end

using Plots
using ProgressBars
function parameter_fit(solve_for_weights, weight_parameters)

    sample_weights = [zeros(length(history_length)) for _ in 1:repetitions]

    Threads.@threads for repetition in ProgressBar(1:repetitions)
        local value_samples = value_sequences[repetition][1:history_length]
        sample_weights[repetition] = solve_for_weights(value_samples, weight_parameters[1])

    end

    plt = plot(1:history_length, mean(sample_weights), ribbon=sem.(sample_weights))
    plot!(1:history_length, smoothing_weights(1:history_length, 0.09))

    display(plt)

end


parameter_fit(WPF_weights, [10])