include("test-stock-return-subsets.jl")

@assert grid_size == 30 "Run the removal investigation with WPF_GRID_SIZE=30."

const influence_output_path = get(
    ENV,
    "SP500_INFLUENCE_OUTPUT",
    "stock-removal-influence-ranking.csv",
)
const removal_output_path = get(
    ENV,
    "SP500_REMOVAL_OUTPUT",
    "stock-removal-sensitivity.csv",
)
const random_control_count = parse(Int, get(ENV, "SP500_REMOVAL_RANDOM_CONTROLS", "4"))

parameters = [0.0; LogRange(1e-1, 1e2, grid_size); Inf]
full_indices = collect(eachindex(all_tickers))
full_distances = distance_matrix(all_returns)

println("Recovering selected SAA and WPF allocations for all $number_of_assets stocks")
baseline_start = time()
baseline_saa_costs, baseline_saa_portfolios =
    testing_saa_costs(all_returns; capture_portfolios = true)
baseline_wpf_costs, baseline_chosen_parameters, baseline_final_parameter, baseline_wpf_portfolios =
    testing_wpf_costs(
        all_returns,
        parameters,
        all_returns,
        full_distances;
        capture_portfolios = true,
    )
println("Recovered baseline allocations in $(round(time() - baseline_start, digits = 1))s")

mean_wpf_weight = vec(mean(baseline_wpf_portfolios; dims = 1))
maximum_wpf_weight = vec(maximum(baseline_wpf_portfolios; dims = 1))
mean_saa_weight = vec(mean(baseline_saa_portfolios; dims = 1))
maximum_saa_weight = vec(maximum(baseline_saa_portfolios; dims = 1))
mean_allocation_difference = vec(mean(abs.(baseline_wpf_portfolios - baseline_saa_portfolios); dims = 1))
maximum_allocation_difference = vec(maximum(abs.(baseline_wpf_portfolios - baseline_saa_portfolios); dims = 1))

metric_candidates = [
    ("largest_mean_wpf_weight", mean_wpf_weight),
    ("largest_maximum_wpf_weight", maximum_wpf_weight),
    ("largest_mean_saa_weight", mean_saa_weight),
    ("largest_maximum_saa_weight", maximum_saa_weight),
    ("largest_mean_allocation_difference", mean_allocation_difference),
    ("largest_maximum_allocation_difference", maximum_allocation_difference),
]

selected_indices = Int[]
selection_reasons = Dict{Int, String}()
for (reason, metric) in metric_candidates
    for index in sortperm(metric; rev = true)
        if !(index in selected_indices)
            push!(selected_indices, index)
            selection_reasons[index] = reason
            break
        end
    end
end

remaining_indices = setdiff(full_indices, selected_indices)
control_rng = MersenneTwister(20260806)
for index in remaining_indices[randperm(control_rng, length(remaining_indices))[1:random_control_count]]
    push!(selected_indices, index)
    selection_reasons[index] = "random_control"
end

influence_rows = [(
    ticker = all_tickers[index],
    mean_wpf_weight = mean_wpf_weight[index],
    maximum_wpf_weight = maximum_wpf_weight[index],
    wpf_months_held = count(>(1e-8), @view baseline_wpf_portfolios[:, index]),
    mean_saa_weight = mean_saa_weight[index],
    maximum_saa_weight = maximum_saa_weight[index],
    saa_months_held = count(>(1e-8), @view baseline_saa_portfolios[:, index]),
    mean_absolute_allocation_difference = mean_allocation_difference[index],
    maximum_absolute_allocation_difference = maximum_allocation_difference[index],
    selected_for_removal = index in selected_indices,
    selection_reason = get(selection_reasons, index, ""),
) for index in full_indices]
CSV.write(influence_output_path, influence_rows)

baseline_saa_cost = risk_adjusted_cost(baseline_saa_costs)
baseline_wpf_cost = risk_adjusted_cost(baseline_wpf_costs)
baseline_difference_pct = 100 * (baseline_wpf_cost - baseline_saa_cost) / baseline_saa_cost

function removal_row(
        removed_ticker,
        selection_reason,
        saa_costs,
        wpf_costs,
        chosen_parameters,
        final_parameter,
        elapsed_seconds,
        bootstrap_seed)
    saa_cost = risk_adjusted_cost(saa_costs)
    wpf_cost = risk_adjusted_cost(wpf_costs)
    difference_pct = 100 * (wpf_cost - saa_cost) / saa_cost
    mean_loss_difference = mean(wpf_costs - saa_costs)
    cvar_difference = ((wpf_cost - saa_cost) - ρ * mean_loss_difference) / (1 - ρ)
    ci_low, ci_high = circular_block_bootstrap(saa_costs, wpf_costs, bootstrap_seed)

    return (
        removed_ticker = removed_ticker,
        selection_reason = selection_reason,
        remaining_assets = removed_ticker == "(none)" ? number_of_assets : number_of_assets - 1,
        saa_cost = saa_cost,
        wpf_cost = wpf_cost,
        difference_from_saa_pct = difference_pct,
        change_from_baseline_percentage_points = difference_pct - baseline_difference_pct,
        saa_cost_change = saa_cost - baseline_saa_cost,
        wpf_cost_change = wpf_cost - baseline_wpf_cost,
        mean_loss_difference = mean_loss_difference,
        cvar_difference = cvar_difference,
        saa_wealth = portfolio_wealth(saa_costs),
        wpf_wealth = portfolio_wealth(wpf_costs),
        block_bootstrap_low_pct = ci_low,
        block_bootstrap_high_pct = ci_high,
        final_lambda = parameter_label(final_parameter),
        distinct_chosen_lambdas = length(unique(chosen_parameters)),
        saa_tail_months = tail_month_labels(saa_costs),
        wpf_tail_months = tail_month_labels(wpf_costs),
        elapsed_seconds = elapsed_seconds,
    )
end

results = [removal_row(
    "(none)",
    "baseline",
    baseline_saa_costs,
    baseline_wpf_costs,
    baseline_chosen_parameters,
    baseline_final_parameter,
    time() - baseline_start,
    2_026_080_600,
)]
CSV.write(removal_output_path, results)

println("Selected exact removal tests:")
for index in selected_indices
    println(
        "  $(all_tickers[index]): $(selection_reasons[index]); " *
        "mean WPF=$(round(mean_wpf_weight[index], digits = 4)), " *
        "max WPF=$(round(maximum_wpf_weight[index], digits = 4)), " *
        "mean SAA=$(round(mean_saa_weight[index], digits = 4)), " *
        "max SAA=$(round(maximum_saa_weight[index], digits = 4))",
    )
end

for (removal_position, removed_index) in pairs(selected_indices)
    removed_ticker = all_tickers[removed_index]
    retained_indices = filter(!=(removed_index), full_indices)
    reduced_returns = [returns[retained_indices] for returns in all_returns]
    start_time = time()
    saa_costs = testing_saa_costs(reduced_returns)
    reduced_distances = distance_matrix(reduced_returns)
    wpf_costs, chosen_parameters, final_parameter = testing_wpf_costs(
        reduced_returns,
        parameters,
        reduced_returns,
        reduced_distances,
    )
    elapsed_seconds = time() - start_time
    row = removal_row(
        removed_ticker,
        selection_reasons[removed_index],
        saa_costs,
        wpf_costs,
        chosen_parameters,
        final_parameter,
        elapsed_seconds,
        2_026_081_000 + removal_position,
    )
    push!(results, row)
    CSV.write(removal_output_path, results)
    println(
        "removed=$(removed_ticker) difference=$(round(row.difference_from_saa_pct, digits = 2))% " *
        "change=$(round(row.change_from_baseline_percentage_points, digits = 2))pp " *
        "CI=[$(round(row.block_bootstrap_low_pct, digits = 2)), " *
        "$(round(row.block_bootstrap_high_pct, digits = 2))]% " *
        "elapsed=$(round(elapsed_seconds, digits = 1))s",
    )
end

println("Completed $(length(selected_indices)) exact single-stock removal tests.")
