
price_revenue(price,value) = price*ifelse(value >= price, price, 0)

function optimal_price(values, weights)
    
    T = length(values)
    revenues = zeros(T)
    
    for i in 1:T; revenues[i] = sum(weights[j]*price_revenue(values[i],values[j]) for j in 1:T); end

    return values[argmax(revenues)]
end

Random.seed!(42)

using LinearAlgebra

shift_distribution = Normal(0, 1)

repetitions = 100
history_length = 30

value_sequences = [zeros(history_length+1) for _ in 1:repetitions]
value_distributions = [[Normal(0,1) for _ in 1:history_length+1] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 1
    σ = 1

    for t in 1:history_length+1
        value_distributions[repetition][t] = Normal(μ, σ)
        value_sequences[repetition][t] = rand(value_distributions[repetition][t])

        μ = μ + rand(shift_distribution)     
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
        
        revenues[repetition][weight_parameter_index] = price_revenue(price, value_sequences[repetition][history_length+1])

    end

    weight_parameter_index = argmax(mean(revenues))
    maximal_revenues = [revenues[repetition][weight_parameter_index] for repetition in 1:repetitions]

    digits = 4

    display([round(mean(maximal_revenues), digits=digits), round(sem(maximal_revenues), digits=digits), round(weight_parameters[weight_parameter_index], digits=digits)])

    display(plot(weight_parameters, mean(revenues), xscale=:log10))

    return maximal_revenues
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
include("weights.jl")


parameter_fit(windowing_weights, history_length)
#parameter_fit(windowing_weights, round.(Int, LinRange(1,history_length,history_length)))
SES_revenues = parameter_fit(SES_weights, [LinRange(0.00001,0.0001,10); LinRange(0.0001,0.001,9); LinRange(0.002,0.01,9); LinRange(0.02,1.0,99)])

WPF_revenues = parameter_fit(WPF_weights, [LinRange(.002,.01,9); LinRange(.02,.1,9); LinRange(.2,1,9); LinRange(2,10,9); LinRange(20,100,9)])

display(sem(WPF_revenues - SES_revenues))






