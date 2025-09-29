using Random, Distributions, Statistics, StatsBase
using IterTools, ProgressBars

include("weights.jl")

Cu = 4  # Underage cost.
Co = 1  # Overage cost.

newsvendor_loss(order, demand) =
    sum(Cu * max(demand[i] - order[i], 0) + Co * max(order[i] - demand[i], 0) for i in eachindex(order))

function newsvendor_order(demand, weights)
    m, T = size(demand)
    q = Cu / (Cu + Co)
    
    return [quantile(demand[i, :], Weights(weights), q) for i in 1:m]
end

Random.seed!(42)

dimension = 2
modes = 3

repetitions = 300
history_length = 50

# Initial demand-distribution parameters. Mixture of axis-aligned normals.
μ = [i*100 for i in 1:modes]
σ = 10

# Demand-mode shift-distribution parameters.
shift_distribution = [MvNormal(zeros(dimension), (10^2) * I) for _ in 1:modes]

demand = [zeros(dimension, history_length) for _ in 1:repetitions]
final_demand = [[Vector{Float64}(undef, dimension) for _ in 1:1000] for _ in 1:repetitions]

for repetition in 1:repetitions

    μs = [ones(dimension) * μ[i] for i in 1:modes]

    for t in 1:history_length
        demand[repetition][:, t] = rand(MixtureModel(MvNormal, [(μs[i], Diagonal(fill((σ)^2, dimension))) for i in 1:modes]))
        
        for i in eachindex(μs); μs[i] += rand(shift_distribution[1]); end

    end

    for j in 1:length(final_demand[1])
        final_demand[repetition][j] = 
            rand(MixtureModel(MvNormal, [(μs[i] + rand(shift_distribution[i]), Diagonal(fill((σ)^2, dimension))) for i in 1:modes]))

    end
end


function parameter_fit(solve_for_weights, weight_parameters, distance_function)
    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        demand_samples = demand[repetition]
        weights = solve_for_weights(eachcol(demand_samples), weight_parameters[weight_parameter_index], distance_function)
        order = newsvendor_order(demand_samples, weights)

        costs[repetition][weight_parameter_index] = mean([
            newsvendor_loss(order, final_demand[repetition][i]) for i in eachindex(final_demand[repetition])])
    
    end

    minimal_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][minimal_index] for repetition in 1:repetitions]

    display([
        round(mean(minimal_costs), digits = 4),
        round(sem(minimal_costs), digits = 4),
        round(weight_parameters[minimal_index], digits = 4)
    ])

    return minimal_costs
end

parameter_fit(windowing_weights, history_length, 0)

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

windowing_costs = parameter_fit(windowing_weights, unique(ceil.(Int, LogRange(1,history_length,30))), 0)
smoothing_costs = parameter_fit(smoothing_weights, [[0]; LogRange(1e-3, 1, 30)], 0)

using LinearAlgebra
d(ξ, ζ) = norm(ξ - ζ, 1)
WPF_costs = parameter_fit(WPF_weights, [[0]; LogRange(1e-3, 1, 30); [Inf]], d)

display(mean(WPF_costs - smoothing_costs) / mean(smoothing_costs)*100)
display(sign(mean(WPF_costs - smoothing_costs)) * mean(WPF_costs - smoothing_costs) / sem(WPF_costs - smoothing_costs))

