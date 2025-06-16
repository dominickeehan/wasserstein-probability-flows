using Random, Statistics, StatsBase, Distributions

Cu = 4 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)

using QuadGK

function expected_newsvendor_loss(order, demand_distribution)

    #overage_integrand(d) = (order-d) * pdf(demand_distribution, d)
    #overage_result, error = quadgk(overage_integrand, -Inf, order)

    #underage_integrand(d) = (d-order) * pdf(demand_distribution, d)
    #underage_result, error = quadgk(underage_integrand, order, Inf)

    #return Cu*underage_result + Co*overage_result

    Q = order

    μ = mean(demand_distribution)
    σ = std(demand_distribution)
    z = (Q - μ) / σ
    
    phi = pdf(Normal(), z)  # Standard normal PDF
    Phi = cdf(Normal(), z)  # Standard normal CDF
    
    overage = σ * (z * Phi + phi)
    underage = σ * (phi - z * (1 - Phi))
    
    return Cu * underage + Co * overage

end

newsvendor_order(ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

Random.seed!(42)

using LinearAlgebra

shift_distribution = Normal(0,0)

repetitions = 100
history_length = 10

demand_sequences = [zeros(history_length) for _ in 1:repetitions]
demand_distributions = [[Normal(0, 0) for _ in 1:history_length] for _ in 1:repetitions]
final_demand_distributions = [[Normal(0,0) for _ in 1:10*repetitions] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 100
    σ = 20

    for t in 1:history_length

        demand_distributions[repetition][t] = Normal(μ, σ)
        demand_sequences[repetition][t] = rand(demand_distributions[repetition][t])

        if t < history_length
            μ = μ + rand(shift_distribution)

        else
            for s in eachindex(final_demand_distributions[repetition])
                final_demand_distributions[repetition][s] = Normal(μ + rand(shift_distribution), σ)

            end
        end   
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
        costs[repetition][weight_parameter_index] = 
            #expected_newsvendor_loss(order, demand_distributions[repetition][history_length+1])
            mean([expected_newsvendor_loss(order, final_demand_distributions[repetition][s]) for s in eachindex(final_demand_distributions[repetition])])

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

smoothing_costs = parameter_fit(smoothing_weights, [LinRange(0.00001,0.0001,10); LinRange(0.0001,0.001,9); LinRange(0.002,0.01,9); LinRange(0.02,1.0,99)])

WPF_costs = parameter_fit(WPF_weights, [LinRange(.0000001,.000001,10); LinRange(.000002,.00001,9); LinRange(.00002,.0001,9); LinRange(.0002,.001,9); LinRange(.002,.01,9); LinRange(.02,.1,9); LinRange(.2,1,9); LinRange(2,10,9); LinRange(20,100,9); LinRange(200,1000,9);])


display(mean(WPF_costs - smoothing_costs)/mean(smoothing_costs))
display(sign(mean(WPF_costs - smoothing_costs))*(mean(WPF_costs - smoothing_costs))/sem(WPF_costs - smoothing_costs))



