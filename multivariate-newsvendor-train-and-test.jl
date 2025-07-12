using Random, Statistics, StatsBase, Distributions
using LinearAlgebra, ProgressBars, IterTools

# --- Cost parameters ---
Cu = 4
Co = 1

# --- Loss and expected loss ---
newsvendor_loss(x::AbstractVector, ξ::AbstractVector) =
    sum(Cu * max(ξ[i] - x[i], 0) + Co * max(x[i] - ξ[i], 0) for i in eachindex(x))

function expected_newsvendor_loss(order::AbstractVector, demand_dists::Vector{<:UnivariateDistribution})
    total_cost = 0.0
    for i in eachindex(order)
        Q = order[i]
        μ = mean(demand_dists[i])
        σ = std(demand_dists[i])
        if σ == 0
            total_cost += Cu * max(μ - Q, 0) + Co * max(Q - μ, 0)
        else
            z = (Q - μ) / σ
            ϕ = pdf(Normal(), z)
            Φ = cdf(Normal(), z)
            overage = σ * (z * Φ + ϕ)
            underage = σ * (ϕ - z * (1 - Φ))
            total_cost += Cu * underage + Co * overage
        end
    end
    return total_cost
end

# --- Quantile-based ordering ---
function newsvendor_order(ξ::Matrix{Float64}, weights::Vector{Float64})
    dim, T = size(ξ)
    q = Cu / (Cu + Co)
    [quantile(ξ[i, :], Weights(weights), q) for i in 1:dim]
end

# --- Setup ---
Random.seed!(42)

dim = 4
repetitions = 100
history_length = 100
training_length = 30
σ = 10.0
shift_distribution = MvNormal(zeros(dim), 1 * I)

num_future_samples = 10000  # <--- PARAMETERIZED HERE

demand_sequences = [zeros(dim, history_length + 1) for _ in 1:repetitions]
final_demand_distributions = [[Vector{UnivariateDistribution}(undef, dim) for _ in 1:num_future_samples] for _ in 1:repetitions]

for rep in 1:repetitions
    μ = ones(dim) * 100.0
    for t in 1:(history_length + 1)
        demand_sequences[rep][:, t] .= rand(MvNormal(μ, Diagonal(fill(σ, dim))))
        μ .+= rand(shift_distribution)
    end
    for s in 1:num_future_samples
        μs = μ .+ rand(shift_distribution)
        final_demand_distributions[rep][s] = [Normal(μs[i], σ) for i in 1:dim]
    end
end

# --- Evaluation with training and analytic expected test loss ---
function train_and_test(solve_for_weights, weight_parameters)
    costs = zeros(repetitions)

    for repetition in ProgressBar(1:repetitions)
        training_costs = [zeros(length(weight_parameters)) for _ in 1:training_length]

        Threads.@threads for (param_index, t_offset) in collect(IterTools.product(eachindex(weight_parameters), 1:training_length))
            t = history_length - training_length + t_offset
            ξ_train = demand_sequences[repetition][:, 1:t-1]
            ξ_true = demand_sequences[repetition][:, t]
            weights = solve_for_weights(eachcol(ξ_train), weight_parameters[param_index])
            x = newsvendor_order(ξ_train, collect(weights))
            training_costs[t_offset][param_index] = newsvendor_loss(x, ξ_true)
        end

        best_index = argmin(mean(training_costs))
        ξ_all = demand_sequences[repetition][:, 1:history_length]
        weights = solve_for_weights(eachcol(ξ_all), weight_parameters[best_index])
        x = newsvendor_order(ξ_all, collect(weights))
        dists = final_demand_distributions[repetition]
        costs[repetition] = mean(expected_newsvendor_loss(x, dists[s]) for s in 1:num_future_samples)
    end

    return mean(costs), sem(costs), costs
end


# --- Distance ---
d(i, j, ξ_i, ξ_j) = norm(ξ_i - ξ_j, 1)

# --- Load weighting methods ---
include("weights.jl")  # defines: smoothing_weights, WPF_weights, windowing_weights

# --- Weight parameters ---
smoothing_params = [LinRange(0.002, 0.01, 9); LinRange(0.02, 0.1, 9); LinRange(0.2, 1, 9)]
WPF_params = [LinRange(.02, .1, 9); LinRange(.2, 1, 9); LinRange(2, 10, 9)]

println("\nEvaluating smoothing:")
smoothing_mean, smoothing_sem, smoothing_costs = train_and_test(smoothing_weights, smoothing_params)

println("\nEvaluating WPF:")
WPF_mean, WPF_sem, WPF_costs = train_and_test(WPF_weights, WPF_params)

println("\nEvaluating SAA:")
SAA_mean, SAA_sem, SAA_costs = train_and_test(windowing_weights, history_length)

# --- Comparison stats ---
function compare(name1, μ1, sem1, name2, μ2, sem2)
    perc = (μ1 - μ2) / μ1
    z = (μ1 - μ2) / sqrt(sem1^2 + sem2^2)
    println("→ ", name2, " vs ", name1)
    println("   Mean ", name1, ": ", round(μ1, digits=3), " ± ", round(sem1, digits=3))
    println("   Mean ", name2, ": ", round(μ2, digits=3), " ± ", round(sem2, digits=3))
    println("   Improvement: ", round(100 * perc, digits=2), "%")
    println("   Z-score: ", round(z, digits=3), "\n")
end

println("\n--- Comparisons ---")
compare("smoothing", smoothing_mean, smoothing_sem, "WPF", WPF_mean, WPF_sem)
compare("SAA", SAA_mean, SAA_sem, "WPF", WPF_mean, WPF_sem)
