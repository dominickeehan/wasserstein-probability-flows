using Random, Statistics, StatsBase, Distributions

price_revenue(price,value) = price*ifelse(value >= price, 1, 0)

using QuadGK

function expected_price_revenue(price, value_distribution)

    #integrand(p) = p * pdf(value_distribution, p)
    #result, error = quadgk(integrand, price, Inf)

    result = price * (1-cdf(value_distribution, price))

    return result
end

function optimal_price(values, weights)
    
    T = length(values)
    revenues = zeros(T)
    
    for i in 1:T; revenues[i] = sum(weights[j]*price_revenue(values[i],values[j]) for j in 1:T); end

    return values[argmax(revenues)]
end

Random.seed!(42)

using LinearAlgebra

shift_distribution = Normal(0, 10)

repetitions = 300
history_length = 30 #30 #100, 30

value_sequences = [zeros(history_length+1) for _ in 1:repetitions]
value_distributions = [[Normal(0, 1) for _ in 1:history_length] for _ in 1:repetitions]
final_value_distributions = [[Normal(0,1) for _ in 1:10000] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 100
    σ = 30 # 200

    for t in 1:history_length
        value_distributions[repetition][t] = Normal(μ, σ)
        value_sequences[repetition][t] = rand(value_distributions[repetition][t])

        if t < history_length
            μ = μ + rand(shift_distribution)

        else
            for s in eachindex(final_value_distributions[repetition])
                final_value_distributions[repetition][s] = Normal(μ + rand(shift_distribution), σ)

            end
        end        
    end
end

using Plots
using ProgressBars, IterTools
function parameter_fit(solve_for_weights, weight_parameters)

    revenues = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        local value_samples = value_sequences[repetition][1:history_length]
        local value_sample_weights = solve_for_weights(value_samples, weight_parameters[weight_parameter_index])
        local price = optimal_price(value_samples, value_sample_weights)
        
        revenues[repetition][weight_parameter_index] = 
            mean([expected_price_revenue(price, final_value_distributions[repetition][s]) for s in eachindex(final_value_distributions[repetition])])

    end

    weight_parameter_index = argmax(mean(revenues))
    maximal_revenues = [revenues[repetition][weight_parameter_index] for repetition in 1:repetitions]

    digits = 3

    display([round(mean(maximal_revenues), digits=digits), round(sem(maximal_revenues), digits=digits), round(weight_parameters[weight_parameter_index], digits=digits)])

    #display(plot(weight_parameters, mean(revenues), xscale=:log10, label=nothing))

    return maximal_revenues
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1) #ifelse(i==j, 0, norm(ξ_i - ξ_j, 1)+1000)
include("weights.jl")


#parameter_fit(windowing_weights, history_length)
smoothing_revenues = parameter_fit(smoothing_weights, [LinRange(0.001,0.01,10); LinRange(0.01,0.1,10); LinRange(0.1,1.0,10)])
WPF_revenues = parameter_fit(WPF_weights, [LinRange(0.001,0.01,10); LinRange(0.02,0.1,9); LinRange(0.2,1,9); LinRange(2,10,9); LinRange(20,100,9); LinRange(200,1000,9); LinRange(2000,10000,9)])

display(mean(WPF_revenues - smoothing_revenues)/mean(smoothing_revenues))
display(sign(mean(WPF_revenues - smoothing_revenues))*(mean(WPF_revenues - smoothing_revenues))/sem(WPF_revenues - smoothing_revenues))





