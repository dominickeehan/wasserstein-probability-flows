include("test-stock-return-subsets.jl")

@assert subset_size == 350 "Run this diagnostic with SP500_SUBSET_SIZE=350."
@assert grid_size == 30 "Run this diagnostic with WPF_GRID_SIZE=30."

diagnostic_seeds = parse_seeds(get(ENV, "SP500_DIAGNOSTIC_SEEDS", "3,9"))
parameters = [0.0; LogRange(1e-1, 1e2, grid_size); Inf]
saa_parameter_index = length(parameters)

function select_with_guardrail(
        training_costs,
        testing_costs;
        minimum_relative_improvement,
        persistence)
    parameter_costs = [training_costs; testing_costs]
    realised_costs = zeros(testing_T)
    chosen_indices = fill(saa_parameter_index, testing_T)
    unconstrained_best_indices = zeros(Int, testing_T)
    relative_improvements = zeros(testing_T)
    risk_costs = zeros(length(parameters))

    for t in 1:testing_T
        tuning_indices = training_T + t - parameter_tuning_window:training_T + t - 1
        for parameter_index in eachindex(parameters)
            risk_costs[parameter_index] =
                risk_adjusted_cost(@view parameter_costs[tuning_indices, parameter_index])
        end

        best_index = argmin(risk_costs)
        unconstrained_best_indices[t] = best_index
        saa_validation_cost = risk_costs[saa_parameter_index]
        relative_improvement =
            (saa_validation_cost - risk_costs[best_index]) / max(abs(saa_validation_cost), eps())
        relative_improvements[t] = relative_improvement

        stable = persistence == 1 ||
            (t >= persistence && all(==(best_index), @view unconstrained_best_indices[(t - persistence + 1):t]))
        if best_index != saa_parameter_index &&
                relative_improvement >= minimum_relative_improvement && stable
            chosen_indices[t] = best_index
        end
        realised_costs[t] = testing_costs[t, chosen_indices[t]]
    end

    return realised_costs, chosen_indices, relative_improvements
end

policies = [
    (name = "current", threshold = 0.0, persistence = 1),
    (name = "guard_2pct", threshold = 0.02, persistence = 1),
    (name = "guard_5pct", threshold = 0.05, persistence = 1),
    (name = "guard_10pct", threshold = 0.10, persistence = 1),
    (name = "guard_20pct", threshold = 0.20, persistence = 1),
    (name = "guard_5pct_persist2", threshold = 0.05, persistence = 2),
    (name = "guard_10pct_persist2", threshold = 0.10, persistence = 2),
    (name = "always_saa", threshold = Inf, persistence = 1),
]

results = NamedTuple[]
for seed in diagnostic_seeds
    indices = subset_indices(seed)
    subset_returns = [returns[indices] for returns in all_returns]
    saa_costs = testing_saa_costs(subset_returns)
    distances = distance_matrix(subset_returns)
    _, _, _, training_costs, testing_costs = testing_wpf_costs(
        subset_returns,
        parameters,
        subset_returns,
        distances;
        return_parameter_costs = true,
    )

    saa_recovery_error = maximum(abs.(testing_costs[:, saa_parameter_index] - saa_costs))
    println("seed=$seed maximum lambda=Inf versus SAA error: $saa_recovery_error")

    for policy in policies
        realised_costs, chosen_indices, relative_improvements = select_with_guardrail(
            training_costs,
            testing_costs;
            minimum_relative_improvement = policy.threshold,
            persistence = policy.persistence,
        )
        saa_cost = risk_adjusted_cost(saa_costs)
        wpf_cost = risk_adjusted_cost(realised_costs)
        mean_loss_difference = mean(realised_costs - saa_costs)
        cvar_difference = ((wpf_cost - saa_cost) - ρ * mean_loss_difference) / (1 - ρ)
        results_row = (
            seed = seed,
            policy = policy.name,
            minimum_relative_validation_improvement = policy.threshold,
            persistence = policy.persistence,
            difference_from_saa_pct = 100 * (wpf_cost - saa_cost) / saa_cost,
            saa_cost = saa_cost,
            selected_cost = wpf_cost,
            mean_loss_difference = mean_loss_difference,
            cvar_difference = cvar_difference,
            selected_wealth = portfolio_wealth(realised_costs),
            non_saa_share = mean(chosen_indices .!= saa_parameter_index),
            parameter_switches = count(!=(0), diff(chosen_indices)),
            distinct_selected_parameters = length(unique(chosen_indices)),
            mean_validation_improvement = mean(relative_improvements),
            median_validation_improvement = median(relative_improvements),
            saa_recovery_error = saa_recovery_error,
            selected_tail_months = tail_month_labels(realised_costs),
        )
        push!(results, results_row)
        println(
            "seed=$seed policy=$(policy.name) difference=" *
            "$(round(results_row.difference_from_saa_pct, digits = 2))% " *
            "non-SAA=$(round(100 * results_row.non_saa_share, digits = 1))%",
        )
    end
    CSV.write("stock-tuning-guardrail-diagnostics.csv", results)
end
