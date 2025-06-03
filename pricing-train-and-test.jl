using Random, Statistics, StatsBase, Distributions
using QuadGK
using Plots, Measures

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

shift_distribution = Normal(0, 100) #Normal(0, 100)

repetitions = 3000 # 100
history_length = 50 # 50
training_length = 15 # 15

value_sequences = [zeros(history_length+1) for _ in 1:repetitions]
value_distributions = [[Normal(0,1) for _ in 1:history_length+1] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 100 # 100
    σ = 20 # 20

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

    #println("training and testing method...")

    for repetition in ProgressBar(1:repetitions)

        training_revenues = [zeros(length(weight_parameters)) for _ in history_length-training_length+1:history_length]
        Threads.@threads for (weight_parameter_index, t) in collect(IterTools.product(eachindex(weight_parameters), history_length-training_length+1:history_length))

            local value_samples = value_sequences[repetition][1:t-1]
            local value_sample_weights = solve_for_weights(value_samples, weight_parameters[weight_parameter_index])
            local price = optimal_price(value_samples, value_sample_weights)
            training_revenues[t-(history_length-training_length)][weight_parameter_index] = price_revenue(price, value_sequences[repetition][t])
        end

        #display(plot(weight_parameters, mean(training_revenues), xscale=:log10))

        weight_parameter_index = argmax(mean(training_revenues))
        #println([weight_parameter_index, length(weight_parameters)])
        value_samples = value_sequences[repetition][1:history_length]
        value_sample_weights = solve_for_weights(value_samples, weight_parameters[weight_parameter_index])
        price = optimal_price(value_samples, value_sample_weights)
        #revenues[repetition] = price_revenue(price, value_sequences[repetition][history_length+1])
        revenues[repetition] = expected_price_revenue(price, value_distributions[repetition][history_length+1])
    end

    μ = mean(revenues)
    σ = sem(revenues)

    println("$μ ± $σ")

    return revenues

end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
include("weights.jl")


SAA = train_and_test(windowing_weights, [history_length])
windowing = train_and_test(windowing_weights, round.(Int, LinRange(1,history_length,30)))
smoothing = train_and_test(SES_weights, [LinRange(0.001,0.01,10); LinRange(0.01,0.1,10); LinRange(0.1,1.0,10)])

WPF = train_and_test(WPF_weights, [LinRange(0.1,1,10); LinRange(2,10,9); LinRange(20,100,9)])  # train_and_test(WPF_weights, [LinRange(0.1,1,D); LinRange(1,10,D); LinRange(10,100,D); LinRange(100,1000,D)])

display(sem(WPF - SAA))

# (262.0736477091183, 13.092936082748315)
# (248.95441024652078, 13.366638093872824)
# (250.05230443034603, 13.40248200641896)
# (274.4764549759506, 13.767884442532075)

