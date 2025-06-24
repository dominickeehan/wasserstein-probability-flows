using Random, Statistics, StatsBase, Distributions

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
include("weights.jl")

Random.seed!(42)

using LinearAlgebra

dims = 10

shift_distribution = MvNormal(zeros(dims), 1*I)

repetitions = 1000
history_length = 20

sample_sequences = [[zeros(dims) for t in 1:history_length+1] for r in 1:repetitions]

for repetition in 1:repetitions
    μ = 100*ones(dims)
    σ = 10

    for t in 1:history_length+1
        sample_sequences[repetition][t] = rand(MvNormal(μ, σ*I))
        μ = μ + rand(shift_distribution)
    end
end

using Plots
using ProgressBars
function parameter_fit(solve_for_weights, weight_parameter)

    sample_weights = [zeros(length(history_length)) for _ in 1:repetitions]

    Threads.@threads for repetition in ProgressBar(1:repetitions)
        local samples = sample_sequences[repetition][1:history_length]
        sample_weights[repetition] = solve_for_weights(samples, weight_parameter)

    end

    plt = plot(1:history_length, mean(sample_weights), ribbon=sem.(sample_weights))
    plot!(1:history_length, smoothing_weights(1:history_length, 0.27))

    display(plt)

end

parameter_fit(WPF_weights, 0.1)



