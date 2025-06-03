using Random, Statistics, StatsBase, Distributions

Cu = 3 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)

using QuadGK

function distribution_newsvendor_loss(order, demand_distribution)

    overage_integrand(d) = (order-d) * pdf(demand_distribution, d)
    overage_result, error = quadgk(overage_integrand, -Inf, order)

    underage_integrand(d) = (d-order) * pdf(demand_distribution, d)
    underage_result, error = quadgk(underage_integrand, order, Inf)

    return Cu*underage_result + Co*overage_result
end

newsvendor_order(ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

Random.seed!(42)

using LinearAlgebra

shift_distribution = Laplace(0,100) #MixtureModel(Normal[Normal(0, .1), Normal(0, 100)], [.5, .5]) #MixtureModel(Normal[Normal(0, .1), Normal(0, 100)], [.9, .1]) #Normal(0, 10)

repetitions = 100
history_length = 30

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
demand_distributions = [[Normal(0, 0) for _ in 1:history_length+1] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 1000
    σ = 200

    for t in 1:history_length+1
        demand_distributions[repetition][t] = Normal(μ, σ)
        demand_sequences[repetition][t] = rand(demand_distributions[repetition][t])

        μ = μ + rand(shift_distribution)      
    end
end

using Plots
using ProgressBars, IterTools
function parameter_fit(solve_for_weights, weight_parameters)

    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        local demand_samples = demand_sequences[repetition][1:history_length]
        local demand_sample_weights = solve_for_weights(demand_samples, weight_parameters[weight_parameter_index])
        local order = newsvendor_order(demand_samples, demand_sample_weights)
        costs[repetition][weight_parameter_index] = distribution_newsvendor_loss(order, demand_distributions[repetition][history_length+1])

    end

    weight_parameter_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][weight_parameter_index] for repetition in 1:repetitions]

    digits = 4

    display([round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits), round(weight_parameters[weight_parameter_index], digits=digits)])

    #display(plot(weight_parameters, mean(costs), xscale=:log10))

    return minimal_costs
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i[1] - ξ_j[1], 1) #ifelse(i == j, 0, norm(ξ_i[1] - ξ_j[1], 1)+0)
include("weights.jl")


parameter_fit(windowing_weights, history_length)

SES_costs = parameter_fit(SES_weights, [LinRange(0.00001,0.0001,10); LinRange(0.0001,0.001,9); LinRange(0.002,0.01,9); LinRange(0.02,1.0,99)])

WPF_costs = parameter_fit(WPF_weights, [LinRange(.0000001,.000001,10); LinRange(.000001,.00001,10); LinRange(.00001,.0001,10); LinRange(.0001,.001,10); LinRange(.001,.01,10); LinRange(.01,.1,10); LinRange(.1,1,10); LinRange(1,10,10); LinRange(10,100,10); LinRange(100,1000,10);])


display(mean(WPF_costs - SES_costs))
display(sign(mean(WPF_costs - SES_costs))*(mean(WPF_costs - SES_costs))/sem(WPF_costs - SES_costs))



