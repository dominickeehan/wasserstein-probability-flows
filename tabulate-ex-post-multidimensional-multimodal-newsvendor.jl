using Random, Distributions, Statistics, StatsBase
using LinearAlgebra
using IterTools, ProgressBars

include("weights.jl")

Cu = 4  # Underage cost.
Co = 1  # Overage cost.

Random.seed!(42)

Dimensions = [1,2,3]
Modes = [1,2,3]

percentage_average_differences = zeros((length(Dimensions), length(Modes)))
percentage_sem_differences = zeros((length(Dimensions), length(Modes)))

for dimensions in Dimensions
    for modes in Modes

        newsvendor_loss(order, demand) =
            sum(Cu * max(demand[i] - order[i], 0) + Co * max(order[i] - demand[i], 0) for i in eachindex(order))

        function newsvendor_order(demands, weights)
            q = Cu / (Cu + Co)
            
            return [quantile([demands[t][i] for t in eachindex(demands)], Weights(weights), q) for i in 1:dimensions]
        end

        repetitions = 1000
        history_length = 30

        # Initial demand-distribution parameters. Mixture of axis-aligned normals.
        μ = [i*100 for i in 1:modes]
        σ = 10

        # Demand-mode shift-distribution parameters.
        shift_distribution = [MvNormal(zeros(dimensions), (10^2) * I) for _ in 1:modes]

        demands = [[zeros(dimensions) for _ in 1:history_length] for _ in 1:repetitions]
        final_demand = [[Vector{Float64}(undef, dimensions) for _ in 1:10000] for _ in 1:repetitions]

        for repetition in 1:repetitions

            μs = [ones(dimensions) * μ[i] for i in 1:modes]

            for t in 1:history_length
                demands[repetition][t] = rand(MixtureModel(MvNormal, [(μs[i], Diagonal(fill((σ)^2, dimensions))) for i in 1:modes]))
                
                for i in eachindex(μs); μs[i] += rand(shift_distribution[1]); end

            end

            for j in 1:length(final_demand[1])
                final_demand[repetition][j] = 
                    rand(MixtureModel(MvNormal, [(μs[i] + rand(shift_distribution[i]), Diagonal(fill((σ)^2, dimensions))) for i in 1:modes]))

            end
        end


        function parameter_fit(solve_for_weights, weight_parameters, distance_function)
            costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

            Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
                demand_samples = demands[repetition]
                weights = solve_for_weights(demand_samples, weight_parameters[weight_parameter_index], distance_function)
                order = newsvendor_order(demand_samples, weights)

                costs[repetition][weight_parameter_index] = mean([
                    newsvendor_loss(order, final_demand[repetition][i]) for i in eachindex(final_demand[repetition])])
            
            end

            minimal_index = argmin(mean(costs))
            minimal_costs = [costs[repetition][minimal_index] for repetition in 1:repetitions]

            display([solve_for_weights distance_function])

            mean_minimal_costs = mean(minimal_costs)
            sem_minimal_costs = sem(minimal_costs)
            optimal_weight_parameter = weight_parameters[minimal_index]

            println("Ex-post minimal average cost: $mean_minimal_costs ± $sem_minimal_costs")
            println("Optimal weight parameter: $optimal_weight_parameter")

            return minimal_costs
        end

        LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

        smoothing_costs = parameter_fit(smoothing_weights, [[0]; LogRange(1e-3, 1, 30)], 0)

        L1(ξ, ζ) = norm(ξ - ζ, 1)
        WPF_L1_costs = parameter_fit(WPF_weights, [[0]; LogRange(1e-3, 1, 30); [Inf]], L1)
        percentage_average_difference = mean(WPF_L1_costs - smoothing_costs) / mean(smoothing_costs) * 100
        percentage_sem_difference = sem(WPF_L1_costs - smoothing_costs) / mean(smoothing_costs) * 100
        println("WPF L1 difference from smoothing: $percentage_average_difference ± $percentage_sem_difference %")

        percentage_average_differences[dimensions, modes] = percentage_average_difference
        percentage_sem_differences[dimensions, modes] = percentage_sem_difference

    end
end

display(percentage_average_differences)
display(percentage_sem_differences)


digits = 1

for dimensions in Dimensions
    print("    \$$dimensions\$")

    for modes in Modes
        percentage_average_difference = round(percentage_average_differences[dimensions, modes], digits = digits)
        kern = ifelse(sign(percentage_average_difference) == 1, "\\kern8.5167pt", "\\kern0.0pt")
        percentage_sem_difference = round(percentage_sem_differences[dimensions, modes], digits = digits)
        print(" & \$"*kern*"$percentage_average_difference \\pm $percentage_sem_difference\\,\\%\$")

    end
    println(" \\\\")

end