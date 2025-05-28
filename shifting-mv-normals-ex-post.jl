using Random, Statistics, StatsBase, Distributions

Cu = 4 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = sum(Cu*max(ξ[i]-x[i],0) + Co*max(x[i]-ξ[i],0) for i in eachindex(x))
function newsvendor_order(ξs, weights)

    order = zeros(length(ξs[end]))
    for i in eachindex(ξs[end])

        ξ = [ξs[t][i] for t in eachindex(ξs)]
        order[i] = quantile(ξ, Weights(weights), Cu/(Co+Cu))
    end

    return order

end

Random.seed!(42)

using LinearAlgebra

Dims = 100

matrix = Matrix(1.0*I, Dims, Dims)
for i in 1:Dims; matrix[i,i] = rand()*i; end
μ_shift_distribution = MvNormal(zeros(Dims), 1*matrix)

repetitions = 300
history_length = 30

demand_sequences = [[zeros(Dims) for t in 1:history_length+1] for r in 1:repetitions]

for repetition in 1:repetitions
    μ = 1000*ones(Dims)
    σ = 100

    for t in 1:history_length+1
        demand_sequences[repetition][t] = rand(MvNormal(μ, σ*I))
        μ = μ + rand(μ_shift_distribution)

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
        costs[repetition][weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

    end

    weight_parameter_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][weight_parameter_index] for repetition in 1:repetitions]

    digits = 4

    display([round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits), round(weight_parameters[weight_parameter_index], digits=digits)])

    display(plot(weight_parameters, mean(costs), xscale=:log10))

    return minimal_costs
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1) #ifelse(i == j, 0, norm(ξ_i - ξ_j, 1) + 0)
include("weights.jl")

D = 10

parameter_fit(windowing_weights, history_length)
SES_costs = parameter_fit(SES_weights, [LinRange(0.001,0.01,10); LinRange(0.01,0.1,10); LinRange(0.1,1.0,10)])
WPF_costs = parameter_fit(WPF_weights, [LinRange(.001,.01,D); LinRange(.01,.1,D); LinRange(.1,1,D); LinRange(1,10,D); LinRange(10,100,D)])

display(sem(WPF_costs - SES_costs))

#display(1)