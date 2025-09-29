using Random, Statistics, StatsBase, Distributions
using LinearAlgebra
using IterTools
using ProgressBars, Plots

Cu = 4  # Underage cost.
Co = 1  # Overage cost.

newsvendor_loss(orders, demands) =
    sum(Cu * max(demands[i] - orders[i], 0) + Co * max(orders[i] - demands[i], 0) for i in eachindex(orders))

function expected_newsvendor_loss(orders, normal_demand_distributions)
    costs = zeros(length(orders))

    for i in eachindex(orders)
        order = orders[i]
        μ = mean(normal_demand_distributions[i])
        σ = std(normal_demand_distributions[i])

        z = (order - μ) / σ
        ϕ = pdf(Normal(), z)
        Φ = cdf(Normal(), z)
    
        overage = σ * (z * Φ + ϕ)
        underage = σ * (ϕ - z * (1 - Φ))
        costs[i] = Cu * underage + Co * overage
    
    end

    return sum(costs)
end

function newsvendor_orders(demands, weights)
    m, T = size(demands)
    q = Cu / (Cu + Co)
    
    return [quantile(demands[i, :], Weights(weights), q) for i in 1:m]
end

Random.seed!(42)

dimension = 10
repetitions = 100
history_length = 50

μ = 100
σ = 10

shift_distribution = MvNormal(zeros(dimension), (1^2) * I)

demands = [zeros(dimension, history_length) for _ in 1:repetitions]
final_demand_distributions = [[Vector{UnivariateDistribution}(undef, dimension) for _ in 1:10000] for _ in 1:repetitions]

for repetition in 1:repetitions
    μs = ones(dimension) * μ

    for t in 1:history_length
        demands[repetition][:, t] = rand(MvNormal(μs, Diagonal(fill((σ)^2, dimension))))
        μs += rand(shift_distribution)

    end

    for i in 1:length(final_demand_distributions[1])
        ν = μs + rand(shift_distribution)
        final_demand_distributions[repetition][i] = [Normal(ν[j], σ) for j in 1:dimension]

    end
end


function parameter_fit(solve_for_weights, weight_parameters)
    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        demand_samples = demands[repetition]
        weights = solve_for_weights(eachcol(demand_samples), weight_parameters[weight_parameter_index])
        orders = newsvendor_orders(demand_samples, weights)

        costs[repetition][weight_parameter_index] = mean([
            expected_newsvendor_loss(orders, final_demand_distributions[repetition][i]) for i in eachindex(final_demand_distributions[repetition])])
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
    LinRange(2, 10, 9);
    #LinRange(20, 100, 9);
    #LinRange(200, 1000, 9)
])

display(mean(WPF_costs - smoothing_costs) / mean(smoothing_costs)*100)
display(sign(mean(WPF_costs - smoothing_costs)) * mean(WPF_costs - smoothing_costs) / sem(WPF_costs - smoothing_costs))
