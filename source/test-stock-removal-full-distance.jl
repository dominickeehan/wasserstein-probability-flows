include("test-stock-return-subsets.jl")

@assert grid_size == 30 "Run the full-distance removal test with WPF_GRID_SIZE=30."

removed_ticker = get(ENV, "SP500_REMOVED_TICKER", "SJM")
removed_index = findfirst(==(removed_ticker), all_tickers)
isnothing(removed_index) && error("Ticker $removed_ticker is not in the return data.")

parameters = [0.0; LogRange(1e-1, 1e2, grid_size); Inf]
full_distances = distance_matrix(all_returns)

println("Precomputing original 379-stock WPF probability weights")
full_signal_weights = precompute_full_signal_weights(parameters, full_distances)

retained_indices = filter(!=(removed_index), eachindex(all_tickers))
reduced_returns = [returns[retained_indices] for returns in all_returns]
saa_costs = testing_saa_costs(reduced_returns)
wpf_costs, chosen_parameters, final_parameter = testing_wpf_costs(
    reduced_returns,
    parameters,
    all_returns,
    full_distances;
    precomputed_weights = full_signal_weights,
)

saa_cost = risk_adjusted_cost(saa_costs)
wpf_cost = risk_adjusted_cost(wpf_costs)
mean_loss_difference = mean(wpf_costs - saa_costs)
cvar_difference = ((wpf_cost - saa_cost) - ρ * mean_loss_difference) / (1 - ρ)
ci_low, ci_high = circular_block_bootstrap(saa_costs, wpf_costs, 2_026_081_600)

row = (
    removed_ticker = removed_ticker,
    distance_geometry = "original_379_stocks",
    remaining_assets = length(retained_indices),
    saa_cost = saa_cost,
    wpf_cost = wpf_cost,
    difference_from_saa_pct = 100 * (wpf_cost - saa_cost) / saa_cost,
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
)

full_distance_output_path = get(
    ENV,
    "SP500_REMOVAL_FULL_DISTANCE_OUTPUT",
    "stock-removal-full-distance.csv",
)
CSV.write(full_distance_output_path, [row])
println(
    "removed=$removed_ticker with original distances: " *
    "difference=$(round(row.difference_from_saa_pct, digits = 2))% " *
    "CI=[$(round(ci_low, digits = 2)), $(round(ci_high, digits = 2))]%",
)
