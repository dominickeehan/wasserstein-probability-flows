using Random, Statistics, StatsBase, Distributions
using QuadGK

price_revenue(price, value) = price*ifelse(value >= price, 1, 0)
function expected_price_revenue(price, value_distribution)

    integrand(p) = p * pdf(value_distribution, p)
    result, error = quadgk(integrand, price, Inf)

    return result
end

function optimal_price(values, weights)
    
    T = length(values)
    revenues = zeros(T)
    
    for i in 1:T; revenues[i] = sum(weights[j]*price_revenue(values[i],values[j]) for j in 1:T); end

    return values[argmax(revenues)]
end

Random.seed!(42)

shift_distribution = Normal(0, 1000) #Normal(0, 1000) #MixtureModel(Normal[Normal(0, .1), Normal(0, 1000)], [.9, .1])

repetitions = 1000
history_length = 50 #20 #30
training_length = 15 #5 #9

value_sequences = [zeros(history_length+1) for _ in 1:repetitions]
value_distributions = [[Normal(0,1) for _ in 1:history_length+1] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 1000
    σ = 200

    for t in 1:history_length+1
        value_distributions[repetition][t] = Normal(μ, σ)
        value_sequences[repetition][t] = rand(value_distributions[repetition][t])

        μ = μ + rand(shift_distribution)     
    end
end



using ProgressBars, IterTools
function train_and_test(solve_for_weights, weight_parameters)

    revenues = zeros(repetitions)
    #weight_parameters_to_test = zeros(repetitions)

    println("training and testing method...")

    for repetition in ProgressBar(1:repetitions)

        training_revenues = [zeros(length(weight_parameters)) for _ in history_length-training_length+1:history_length]
        Threads.@threads for (weight_parameter_index, t) in collect(IterTools.product(eachindex(weight_parameters), history_length-training_length+1:history_length))

            local value_samples = value_sequences[repetition][1:t-1]
            local value_sample_weights = solve_for_weights(value_samples, weight_parameters[weight_parameter_index])
            local price = optimal_price(value_samples, value_sample_weights)
            training_revenues[t-(history_length-training_length)][weight_parameter_index] = price_revenue(price, value_sequences[repetition][t])
        end

        weight_parameter_index = argmax(mean(training_revenues))
        #println([weight_parameter_index, length(weight_parameters)])
        value_samples = value_sequences[repetition][1:history_length]
        value_sample_weights = solve_for_weights(value_samples, weight_parameters[weight_parameter_index])
        price = optimal_price(value_samples, value_sample_weights)
        #revenues[repetition] = price_revenue(price, value_sequences[repetition][history_length+1])
        revenues[repetition] = expected_price_revenue(price, value_distributions[repetition][history_length+1])
    end

    return mean(revenues), sem(revenues)
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
include("weights.jl")

D = 10
display([train_and_test(windowing_weights, history_length)])
display([train_and_test(windowing_weights, round.(Int, LinRange(1,history_length,40)))])
display([train_and_test(SES_weights, [LinRange(0.0001,0.001,D); LinRange(0.001,0.01,D); LinRange(0.01,0.1,D); LinRange(.1,1,D)])])

display([train_and_test(WPF_weights, [LinRange(0.01,0.1,D); LinRange(0.1,1,D); LinRange(1,10,D); LinRange(10,100,D); LinRange(100,1000,D)])])





