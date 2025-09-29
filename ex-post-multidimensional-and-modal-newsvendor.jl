using Random, Statistics, StatsBase, Distributions
using LinearAlgebra
using IterTools
using ProgressBars, Plots

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
repetitions = 150
history_length = 50

μ_1 = 100
σ_1 = 10
μ_2 = 200
σ_2 = 10
μ_3 = 300
σ_3 = 10

shift_distribution_1 = MvNormal(zeros(dimension), (10^2) * I)
shift_distribution_2 = MvNormal(zeros(dimension), (10^2) * I)
shift_distribution_3 = MvNormal(zeros(dimension), (10^2) * I)

demand = [zeros(dimension, history_length) for _ in 1:repetitions]
final_demand = [[Vector{Float64}(undef, dimension) for _ in 1:1000] for _ in 1:repetitions]

for repetition in 1:repetitions
    μs_1 = ones(dimension) * μ_1
    μs_2 = ones(dimension) * μ_2
    μs_3 = ones(dimension) * μ_3

    for t in 1:history_length
        demand[repetition][:, t] = 
            rand(MixtureModel(MvNormal, [(μs_1, Diagonal(fill((σ_1)^2, dimension))), (μs_2, Diagonal(fill((σ_2)^2, dimension))), (μs_3, Diagonal(fill((σ_3)^2, dimension))),]))
        
        μs_1 += rand(shift_distribution_1)
        μs_2 += rand(shift_distribution_2)
        μs_3 += rand(shift_distribution_3)

    end

    for i in 1:length(final_demand[1])
        ν_1 = μs_1 + rand(shift_distribution_1)
        ν_2 = μs_2 + rand(shift_distribution_2)
        ν_3 = μs_3 + rand(shift_distribution_3)

        final_demand[repetition][i] = 
            max.(rand(MixtureModel(MvNormal, [(ν_1, Diagonal(fill((σ_1)^2, dimension))), (ν_2, Diagonal(fill((σ_2)^2, dimension))), (ν_3, Diagonal(fill((σ_3)^2, dimension)))])),0)

    end
end


function parameter_fit(solve_for_weights, weight_parameters)
    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        demand_samples = demand[repetition]
        weights = solve_for_weights(eachcol(demand_samples), weight_parameters[weight_parameter_index])
        order = newsvendor_order(demand_samples, weights)

        costs[repetition][weight_parameter_index] = mean([
            newsvendor_loss(order, final_demand[repetition][i]) for i in eachindex(final_demand[repetition])])
    
    end

    minimal_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][minimal_index] for repetition in 1:repetitions]

    display([
        round(mean(minimal_costs), digits=4),
        round(sem(minimal_costs), digits=4),
        round(weight_parameters[minimal_index], digits=4)
    ])

    return minimal_costs
end


d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)

include("weights.jl")

parameter_fit(windowing_weights, history_length)

smoothing_costs = parameter_fit(smoothing_weights, [
    #LinRange(0.00001, 0.0001, 10);
    LinRange(0.001, 0.01, 9);
    LinRange(0.02, 0.1, 9);
    LinRange(0.2, 1.0, 9)
])


WPF_costs = parameter_fit(WPF_weights, [    
    LinRange(0.002, 0.01, 9);    
    LinRange(0.02, 0.1, 9);
    LinRange(0.2, 1.0, 9);
    #LinRange(2, 10, 9);
    #LinRange(20, 100, 9);
    #LinRange(200, 1000, 9)
])

display(mean(WPF_costs - smoothing_costs) / mean(smoothing_costs)*100)
display(sign(mean(WPF_costs - smoothing_costs)) * mean(WPF_costs - smoothing_costs) / sem(WPF_costs - smoothing_costs))
