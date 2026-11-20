using CSV
using LinearAlgebra
using Random
using Statistics
using StatsBase

include("weights.jl")

ρ = 0.1
α = 0.05
include("risk-averse-portfolio-optimizations.jl")

function LogRange(start, stop, len)
    @assert len > 0
    return len == 1 ? [float(start)] : exp.(LinRange(log(start), log(stop), len))
end

function parse_seeds(value)
    seeds = Int[]
    for part in split(value, ',')
        stripped = strip(part)
        if occursin(':', stripped)
            bounds = parse.(Int, split(stripped, ':'))
            @assert length(bounds) == 2 "Seed ranges must have the form start:stop."
            append!(seeds, bounds[1]:bounds[2])
        else
            push!(seeds, parse(Int, stripped))
        end
    end
    return unique(seeds)
end

const csv_file_path = get(ENV, "SP500_RETURNS_CSV", "data/stock-returns-S&P-500-2014-survivors.csv")
const rows = collect(CSV.File(csv_file_path))
const return_column_names = collect(propertynames(first(rows)))[2:end]
const all_tickers = String.(return_column_names)
const all_dates = String[row[:Date] for row in rows]
const all_returns = [Float64[row[column_name] for column_name in return_column_names] for row in rows]

const number_of_months = length(all_returns)
const number_of_assets = length(all_tickers)
const subset_size = parse(Int, get(ENV, "SP500_SUBSET_SIZE", "100"))
const subset_seeds = parse_seeds(get(ENV, "SP500_SUBSET_SEEDS", "1:10"))
const grid_size = parse(Int, get(ENV, "WPF_GRID_SIZE", "10"))
const bootstrap_repetitions = parse(Int, get(ENV, "SP500_BOOTSTRAPS", "1000"))
const output_path = get(ENV, "SP500_SUBSET_OUTPUT", "data/stock-subset-results.csv")

@assert 1 <= subset_size <= number_of_assets
@assert !isempty(subset_seeds)
@assert grid_size > 0
@assert bootstrap_repetitions > 0

const training_testing_split = ceil(Int, 0.5 * number_of_months)
const warm_up_period = ceil(Int, 0.5 * training_testing_split) - 1
const training_T = training_testing_split - warm_up_period
const testing_T = number_of_months - training_testing_split
const parameter_tuning_window = 2 * 12
const first_training_stage = max(1, training_T - parameter_tuning_window + 1)

portfolio_loss(x, returns) = -dot(x, returns)
risk_adjusted_cost(costs) = ρ * mean(costs) + (1 - ρ) * unweighted_cvar(costs)
portfolio_wealth(costs) = prod(1 .- costs)

function subset_indices(seed)
    rng = MersenneTwister(seed)
    return sort(randperm(rng, number_of_assets)[1:subset_size])
end

function distance_matrix(observations; normalizer = 1.0)
    return WPF_distance_matrix(
        observations,
        (left, right) -> sum(abs(left[k] - right[k]) for k in eachindex(left, right)) / normalizer,
    )
end

function testing_saa_costs(subset_returns; capture_portfolios = false)
    costs = zeros(testing_T)
    portfolios = capture_portfolios ? zeros(testing_T, length(first(subset_returns))) : nothing
    Threads.@threads :greedy for t in 1:testing_T
        sample_end = training_testing_split + t - 1
        samples = @view subset_returns[1:sample_end]
        weights = fill(1 / sample_end, sample_end)
        portfolio = solve_risk_averse_portfolio(samples, weights)
        costs[t] = portfolio_loss(portfolio, subset_returns[training_testing_split + t])
        if capture_portfolios
            portfolios[t, :] .= portfolio
        end
    end
    return capture_portfolios ? (costs, portfolios) : costs
end

function stage_definitions()
    stages = NamedTuple[]
    for t in first_training_stage:training_T
        push!(stages, (
            destination = :training,
            destination_index = t,
            sample_end = warm_up_period + t - 1,
            target_index = warm_up_period + t,
        ))
    end
    for t in 1:testing_T
        push!(stages, (
            destination = :testing,
            destination_index = t,
            sample_end = training_testing_split + t - 1,
            target_index = training_testing_split + t,
        ))
    end
    return stages
end

const stages = stage_definitions()

function precompute_full_signal_weights(parameters, full_distances)
    weights_by_sample_end = Vector{Vector{Vector{Float64}}}(undef, number_of_months)
    Threads.@threads :greedy for stage_index in eachindex(stages)
        stage = stages[stage_index]
        observations = @view all_returns[1:stage.sample_end]
        weights_by_sample_end[stage.sample_end] = WPF_weights_batch(
            observations,
            parameters,
            nothing;
            distances = full_distances,
        )
    end
    return weights_by_sample_end
end

function testing_wpf_costs(
        subset_returns,
        parameters,
        weight_observations,
        distances;
        precomputed_weights = nothing,
        capture_portfolios = false,
        return_parameter_costs = false)
    number_of_parameters = length(parameters)
    training_costs = zeros(training_T, number_of_parameters)
    testing_costs = zeros(testing_T, number_of_parameters)
    testing_portfolios = capture_portfolios ?
        zeros(testing_T, number_of_parameters, length(first(subset_returns))) : nothing

    Threads.@threads :greedy for stage_index in eachindex(stages)
        stage = stages[stage_index]
        samples = @view subset_returns[1:stage.sample_end]
        sample_weights = if isnothing(precomputed_weights)
            WPF_weights_batch(
                @view(weight_observations[1:stage.sample_end]),
                parameters,
                nothing;
                distances = distances,
            )
        else
            precomputed_weights[stage.sample_end]
        end

        target = subset_returns[stage.target_index]
        destination = stage.destination == :training ? training_costs : testing_costs
        for parameter_index in eachindex(parameters)
            portfolio = solve_risk_averse_portfolio(samples, sample_weights[parameter_index])
            destination[stage.destination_index, parameter_index] = portfolio_loss(portfolio, target)
            if capture_portfolios && stage.destination == :testing
                testing_portfolios[stage.destination_index, parameter_index, :] .= portfolio
            end
        end
    end

    parameter_costs = [training_costs; testing_costs]
    realised_costs = zeros(testing_T)
    chosen_parameter_indices = zeros(Int, testing_T)
    final_parameter_costs = zeros(number_of_parameters)

    for t in 1:testing_T
        tuning_indices = training_T + t - parameter_tuning_window:training_T + t - 1
        for parameter_index in eachindex(parameters)
            final_parameter_costs[parameter_index] =
                risk_adjusted_cost(@view parameter_costs[tuning_indices, parameter_index])
        end
        chosen_parameter_indices[t] = argmin(final_parameter_costs)
        realised_costs[t] = testing_costs[t, chosen_parameter_indices[t]]
    end

    final_parameter = parameters[argmin(final_parameter_costs)]
    if capture_portfolios
        realised_portfolios = zeros(testing_T, length(first(subset_returns)))
        for t in 1:testing_T
            realised_portfolios[t, :] .= @view testing_portfolios[t, chosen_parameter_indices[t], :]
        end
        captured = (realised_costs, chosen_parameter_indices, final_parameter, realised_portfolios)
        return return_parameter_costs ? (captured..., training_costs, testing_costs) : captured
    end
    standard = (realised_costs, chosen_parameter_indices, final_parameter)
    return return_parameter_costs ? (standard..., training_costs, testing_costs) : standard
end

function circular_block_bootstrap(saa_costs, wpf_costs, seed; block_length = 6)
    rng = MersenneTwister(seed)
    differences = zeros(bootstrap_repetitions)
    T = length(saa_costs)
    number_of_blocks = ceil(Int, T / block_length)

    for repetition in 1:bootstrap_repetitions
        indices = Int[]
        for _ in 1:number_of_blocks
            block_start = rand(rng, 1:T)
            append!(indices, ((block_start - 1 + offset) % T) + 1 for offset in 0:(block_length - 1))
        end
        resize!(indices, T)
        saa_cost = risk_adjusted_cost(@view saa_costs[indices])
        wpf_cost = risk_adjusted_cost(@view wpf_costs[indices])
        differences[repetition] = 100 * (wpf_cost - saa_cost) / saa_cost
    end

    return quantile(differences, 0.025), quantile(differences, 0.975)
end

function parameter_label(parameter)
    parameter == Inf && return "Inf"
    return string(round(parameter, sigdigits = 6))
end

function tail_month_labels(costs)
    tail_count = ceil(Int, α * length(costs))
    indices = partialsortperm(costs, 1:tail_count; rev = true)
    return join(all_dates[training_testing_split .+ indices], ";")
end

function result_row(seed, variant, indices, parameters, saa_costs, wpf_costs,
        chosen_parameter_indices, final_parameter, elapsed_seconds, bootstrap_seed)
    saa_cost = risk_adjusted_cost(saa_costs)
    wpf_cost = risk_adjusted_cost(wpf_costs)
    ci_low, ci_high = circular_block_bootstrap(saa_costs, wpf_costs, bootstrap_seed)
    endpoint_share = mean(index in (1, length(parameters)) for index in chosen_parameter_indices)
    finite_edge_share = mean(index in (2, length(parameters) - 1) for index in chosen_parameter_indices)

    return (
        seed = seed,
        variant = variant,
        subset_size = length(indices),
        saa_cost = saa_cost,
        wpf_cost = wpf_cost,
        difference_from_saa_pct = 100 * (wpf_cost - saa_cost) / saa_cost,
        block_bootstrap_low_pct = ci_low,
        block_bootstrap_high_pct = ci_high,
        saa_wealth = portfolio_wealth(saa_costs),
        wpf_wealth = portfolio_wealth(wpf_costs),
        paired_mean_loss_difference = mean(wpf_costs - saa_costs),
        final_lambda = parameter_label(final_parameter),
        endpoint_lambda_share = endpoint_share,
        finite_grid_edge_share = finite_edge_share,
        distinct_chosen_lambdas = length(unique(chosen_parameter_indices)),
        saa_tail_months = tail_month_labels(saa_costs),
        wpf_tail_months = tail_month_labels(wpf_costs),
        elapsed_seconds = elapsed_seconds,
        tickers = join(all_tickers[indices], ";"),
    )
end

function checkpoint(results)
    CSV.write(output_path, results)
    return
end

function main()
    current_parameters = [0.0; LogRange(1e-1, 1e2, grid_size); Inf]

    # Dividing L1 by the subset dimension and multiplying lambda by the full
    # dimension puts every subset on the numerical scale of the full-universe run.
    normalised_parameters = [0.0; number_of_assets .* LogRange(1e-1, 1e2, grid_size); Inf]

    variants_setting = split(get(ENV, "SP500_SUBSET_VARIANTS", "current,normalised,full_signal"), ',')
    variants = strip.(variants_setting)
    valid_variants = ("current", "normalised", "full_signal")
    @assert all(variant in valid_variants for variant in variants)

    println("Testing $(length(subset_seeds)) reproducible subsets of $subset_size stocks")
    println("Variants: $(join(variants, ", ")); Julia threads: $(Threads.nthreads())")
    println("Output: $output_path")

    full_distances = "full_signal" in variants ? distance_matrix(all_returns) : nothing
    full_signal_weights = "full_signal" in variants ?
        precompute_full_signal_weights(current_parameters, full_distances) : nothing

    results = NamedTuple[]
    for (seed_position, seed) in pairs(subset_seeds)
        indices = subset_indices(seed)
        subset_returns = [returns[indices] for returns in all_returns]
        saa_costs = testing_saa_costs(subset_returns)
        subset_sum_distances = any(variant in ("current", "normalised") for variant in variants) ?
            distance_matrix(subset_returns) : nothing

        for (variant_position, variant) in pairs(variants)
            start_time = time()
            parameters, weight_observations, distances, precomputed = if variant == "current"
                current_parameters, subset_returns, subset_sum_distances, nothing
            elseif variant == "normalised"
                normalised_parameters, subset_returns, subset_sum_distances ./ subset_size, nothing
            else
                current_parameters, all_returns, full_distances, full_signal_weights
            end

            wpf_costs, chosen_parameter_indices, final_parameter = testing_wpf_costs(
                subset_returns,
                parameters,
                weight_observations,
                distances;
                precomputed_weights = precomputed,
            )
            elapsed_seconds = time() - start_time
            bootstrap_seed = 1_000_000 + 100 * seed_position + variant_position
            row = result_row(
                seed,
                variant,
                indices,
                parameters,
                saa_costs,
                wpf_costs,
                chosen_parameter_indices,
                final_parameter,
                elapsed_seconds,
                bootstrap_seed,
            )
            push!(results, row)
            checkpoint(results)
            println(
                "seed=$(seed) variant=$(variant) difference=$(round(row.difference_from_saa_pct, digits=2))% " *
                "CI=[$(round(row.block_bootstrap_low_pct, digits=2)), $(round(row.block_bootstrap_high_pct, digits=2))]% " *
                "final_lambda=$(row.final_lambda) elapsed=$(round(elapsed_seconds, digits=1))s",
            )
        end
    end

    println("Completed $(length(results)) paired subset/variant experiments.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
