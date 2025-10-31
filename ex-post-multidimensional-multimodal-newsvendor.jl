using Random, Distributions, Statistics, StatsBase
using LinearAlgebra
using IterTools, ProgressBars

include("weights.jl")

Cu = 4  # Underage cost.
Co = 1  # Overage cost.

Random.seed!(42)

dimensions = 1
modes = 1

newsvendor_loss(order, demand) =
    sum(Cu * max(demand[i] - order[i], 0) + Co * max(order[i] - demand[i], 0) for i in eachindex(order))

function newsvendor_order(demands, weights)
    q = Cu / (Cu + Co)
    
    return [quantile([demands[t][i] for t in eachindex(demands)], Weights(weights), q) for i in 1:dimensions]
end

repetitions = 300
history_length = 100

# Initial demand-distribution parameters. Mixture of axis-aligned normals.
μ = [i*100 for i in 1:modes]
σ = 20

# Demand-mode shift-distribution parameters.
shift_distribution = [MvNormal(zeros(dimensions), (15^2) * I) for _ in 1:modes]

demands = [[zeros(dimensions) for _ in 1:history_length] for _ in 1:repetitions]
final_demand = [[Vector{Float64}(undef, dimensions) for _ in 1:1000] for _ in 1:repetitions]

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

SAA_costs = parameter_fit(windowing_weights, history_length, 0)

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

windowing_costs = parameter_fit(windowing_weights, unique(ceil.(Int, LogRange(1,history_length,30))), 0)
smoothing_costs = parameter_fit(smoothing_weights, [[0]; LogRange(1e-3, 1, 30)], 0)

WPF_parameters = [[0]; LinRange(1e-3,1e-2,10); LinRange(2e-2,1e-1,9); LinRange(2e-1,1e0,9); Inf] 

L1(ξ, ζ) = norm(ξ - ζ, 1)
WPF_L1_costs = parameter_fit(WPF_weights, WPF_parameters, L1)
percentage_average_difference = mean(WPF_L1_costs - smoothing_costs) / mean(smoothing_costs) * 100
percentage_sem_difference = sem(WPF_L1_costs - smoothing_costs) / mean(smoothing_costs) * 100
println("WPF L1 difference from smoothing: $percentage_average_difference ± $percentage_sem_difference %")

L2(ξ, ζ) = norm(ξ - ζ, 2)
WPF_L2_costs = parameter_fit(WPF_weights, WPF_parameters, L2)
percentage_average_difference = mean(WPF_L2_costs - smoothing_costs) / mean(smoothing_costs) * 100
percentage_sem_difference = sem(WPF_L2_costs - smoothing_costs) / mean(smoothing_costs) * 100
println("WPF L2 difference from smoothing: $percentage_average_difference ± $percentage_sem_difference %")

LInf(ξ, ζ) = norm(ξ - ζ, Inf)
WPF_LInf_costs = parameter_fit(WPF_weights, WPF_parameters, LInf)
percentage_average_difference = mean(WPF_LInf_costs - smoothing_costs) / mean(smoothing_costs) * 100
percentage_sem_difference = sem(WPF_LInf_costs - smoothing_costs) / mean(smoothing_costs) * 100
println("WPF LInf difference from smoothing: $percentage_average_difference ± $percentage_sem_difference %")

digits = 1

SAA_average_cost = round(mean(SAA_costs), digits = digits)
windowing_average_cost = round(mean(windowing_costs), digits = digits)
smoothing_average_cost = round(mean(smoothing_costs), digits = digits)
WPF_L1_average_cost = round(mean(WPF_L1_costs), digits = digits)
WPF_L2_average_cost = round(mean(WPF_L2_costs), digits = digits)
WPF_LInf_average_cost = round(mean(WPF_LInf_costs), digits = digits)

digits = 1

windowing_percentage_average_difference = round(mean(windowing_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
windowing_percentage_sem_difference = round(sem(windowing_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
smoothing_percentage_average_difference = round(mean(smoothing_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
smoothing_percentage_sem_difference = round(sem(smoothing_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
WPF_L1_percentage_average_difference = round(mean(WPF_L1_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
WPF_L1_percentage_sem_difference = round(sem(WPF_L1_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
WPF_L2_percentage_average_difference = round(mean(WPF_L2_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
WPF_L2_percentage_sem_difference = round(sem(WPF_L2_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
WPF_LInf_percentage_average_difference = round(mean(WPF_LInf_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)
WPF_LInf_percentage_sem_difference = round(sem(WPF_LInf_costs - SAA_costs) / mean(SAA_costs) * 100, digits = digits)

println("\\makecell[r]{Ex-post optimal\\\\average cost} 
        & \$\\textcolor{white}{+}$SAA_average_cost\$ 
        & \$\\textcolor{white}{+}$windowing_average_cost\$ 
        & \$\\textcolor{white}{+}$smoothing_average_cost\$ 
        & \$\\textcolor{white}{+}$WPF_L1_average_cost\$ 
        & \$\\textcolor{white}{+}$WPF_L2_average_cost\$ 
        & \$\\textcolor{white}{+}$WPF_LInf_average_cost\$ \\\\")
println("\\addlinespace")
println("\\midrule")
println("\\addlinespace")
println("\\makecell[r]{Difference\\\\from SAA (\\%)} 
        & \$ \$ 
        & \$$windowing_percentage_average_difference \\pm $windowing_percentage_sem_difference\$ 
        & \$$smoothing_percentage_average_difference \\pm $smoothing_percentage_sem_difference\$
        & \$$WPF_L1_percentage_average_difference \\pm $WPF_L1_percentage_sem_difference\$
        & \$$WPF_L2_percentage_average_difference \\pm $WPF_L2_percentage_sem_difference\$
        & \$$WPF_LInf_percentage_average_difference \\pm $WPF_LInf_percentage_sem_difference\$ \\\\")


