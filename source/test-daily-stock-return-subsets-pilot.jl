using CSV
using Dates
using JuMP
using LinearAlgebra
using MathOptInterface
using Random
using Statistics
using StatsBase

include("weights.jl")

const mean_loss_weight = 0.1
const daily_csv_path = get(ENV, "PILOT_DAILY_CSV", "data/stock-returns-daily-pilot.csv")
const output_path = get(ENV, "PILOT_RESULTS_CSV", "data/stock-daily-pilot-results.csv")
const subset_size = parse(Int, get(ENV, "PILOT_SUBSET_SIZE", "20"))
const subset_seeds = parse.(Int, split(get(ENV, "PILOT_SUBSET_SEEDS", join(1:20, ',')), ','))
const estimation_window = parse(Int, get(ENV, "PILOT_ESTIMATION_DAYS", "126"))
const test_months = parse(Int, get(ENV, "PILOT_TEST_MONTHS", "12"))
const max_runtime_seconds = parse(Float64, get(ENV, "PILOT_MAX_SECONDS", "540"))
const print_monthly_progress = lowercase(get(ENV, "PILOT_MONTHLY_PROGRESS", "false")) in
    ("1", "true", "yes")
const tail_probabilities = [0.05, 0.10, 0.20]
const tuning_windows = [
    parse(Int, get(ENV, "PILOT_TUNING_DAYS_05", "504")),
    parse(Int, get(ENV, "PILOT_TUNING_DAYS_10", "252")),
    parse(Int, get(ENV, "PILOT_TUNING_DAYS_20", "126")),
]

# Each tail problem gets 30 finite eta values plus 0 and Inf. The three grids
# are overlapping slices of one 34-point master grid, so only 34 distinct WPF
# solves are needed at each stage instead of 90.
const master_eta_parameters = exp.(range(log(1e-2), log(3.0), length = 34))
const tail_master_ranges = [1:30, 3:32, 5:34]
const all_eta_parameters = [0.0; master_eta_parameters; Inf]
const tail_parameter_indices = [
    [1; collect(parameter_range .+ 1); length(all_eta_parameters)]
    for parameter_range in tail_master_ranges
]
const eta_parameters_by_tail = [all_eta_parameters[indices] for indices in tail_parameter_indices]

const pilot_portfolio_optimizer = optimizer_with_attributes(
    COPT.Optimizer,
    "Logging" => 0,
    "LogToConsole" => 0,
    "Threads" => 1,
)

function empirical_cvar(costs, tail_probability)
    number_of_costs = length(costs)
    tail_mass = tail_probability * number_of_costs
    full_tail_samples = floor(Int, tail_mass)
    tail_samples = ceil(Int, tail_mass)
    largest_costs = partialsort(costs, 1:tail_samples; rev = true)
    tail_sum = sum(@view largest_costs[1:full_tail_samples])
    if full_tail_samples < tail_samples
        tail_sum += (tail_mass - full_tail_samples) * largest_costs[tail_samples]
    end
    return tail_sum / tail_mass
end

risk_adjusted_cost(costs, tail_probability) =
    mean_loss_weight * mean(costs) +
    (1 - mean_loss_weight) * empirical_cvar(costs, tail_probability)

function solve_capped_portfolio(samples, sample_weights, tail_probability, maximum_weight)
    nonzero = sample_weights .> 0
    weights = sample_weights[nonzero]
    weights ./= sum(weights)
    returns = samples[nonzero]
    number_of_samples = length(returns)
    number_of_assets = length(first(returns))

    model = Model(pilot_portfolio_optimizer)
    @variable(model, 0 <= x[1:number_of_assets] <= maximum_weight)
    @variable(model, threshold)
    @variable(model, slack[1:number_of_samples] >= 0)
    @constraint(model, sum(x) == 1)
    @constraint(model, [i in 1:number_of_samples],
        slack[i] >= -dot(x, returns[i]) - threshold)
    @objective(model, Min,
        -mean_loss_weight * sum(weights[i] * dot(x, returns[i]) for i in 1:number_of_samples) +
        (1 - mean_loss_weight) * threshold +
        (1 - mean_loss_weight) / tail_probability *
            sum(weights[i] * slack[i] for i in 1:number_of_samples))
    optimize!(model)
    termination_status(model) in (MathOptInterface.OPTIMAL, MathOptInterface.ALMOST_OPTIMAL) ||
        error("portfolio solve failed with status $(termination_status(model))")
    return value.(x)
end

function month_ranges(dates)
    ranges = NamedTuple[]
    range_start = 1
    for index in 2:(length(dates) + 1)
        new_month = index > length(dates) ||
            (year(dates[index]), month(dates[index])) !=
            (year(dates[range_start]), month(dates[range_start]))
        if new_month
            push!(ranges, (
                first = range_start,
                last = index - 1,
                label = Dates.format(dates[range_start], "yyyy-mm"),
            ))
            range_start = index
        end
    end
    return ranges
end

function normalised_distances(samples)
    sample_matrix = reduce(vcat, permutedims.(samples))
    scales = vec(std(sample_matrix; dims = 1, corrected = true))
    scales .= max.(scales, 1e-8)
    standardised = [samples[i] ./ scales for i in eachindex(samples)]
    distances = WPF_distance_matrix(
        standardised,
        (left, right) -> mean(abs(left[k] - right[k]) for k in eachindex(left, right)),
    )
    upper_triangle = [distances[i, j] for j in 2:length(samples) for i in 1:(j - 1)]
    typical_distance = median(upper_triangle)
    typical_distance > 0 || error("all pairwise distances are zero")
    distances ./= typical_distance
    return distances
end

function buy_and_hold_losses(portfolio, daily_returns, first_day, last_day)
    cumulative_asset_growth = ones(length(portfolio))
    previous_portfolio_value = 1.0
    losses = zeros(last_day - first_day + 1)
    for (output_index, day) in enumerate(first_day:last_day)
        cumulative_asset_growth .*= 1 .+ @view(daily_returns[day, :])
        portfolio_value = dot(portfolio, cumulative_asset_growth)
        losses[output_index] = -(portfolio_value / previous_portfolio_value - 1)
        previous_portfolio_value = portfolio_value
    end
    return losses
end

function subset_indices(seed, number_of_assets)
    rng = MersenneTwister(seed)
    return sort(randperm(rng, number_of_assets)[1:subset_size])
end

function parameter_label(parameter)
    parameter == Inf && return "Inf"
    return string(round(parameter, sigdigits = 5))
end

function precompute_signal_weights(all_returns, stages, deadline)
    weights_by_stage = Vector{Vector{Vector{Float64}}}(undef, length(stages))
    completed = zeros(UInt8, length(stages))
    Threads.@threads :greedy for stage_position in eachindex(stages)
        time() < deadline || continue
        stage = stages[stage_position]
        training_start = stage.first - estimation_window
        signal_samples = [
            Vector(@view all_returns[day, :])
            for day in training_start:(stage.first - 1)
        ]
        distances = normalised_distances(signal_samples)
        lambdas = [
            isfinite(eta) ? eta * length(signal_samples) : Inf
            for eta in all_eta_parameters
        ]
        weights_by_stage[stage_position] = WPF_weights_batch(
            signal_samples,
            lambdas,
            nothing;
            distances = distances,
        )
        completed[stage_position] = 1
    end
    all(==(1), completed) ||
        error("runtime cap expired while precomputing shared-signal WPF weights")
    return weights_by_stage
end

function run_subset(seed, all_returns, all_tickers, dates, stages, signal_weights_by_stage,
        first_test_stage, tail_first_stage_positions, deadline)
    time() < deadline || return nothing
    indices = subset_indices(seed, size(all_returns, 2))
    returns = all_returns[:, indices]
    tickers = all_tickers[indices]
    number_of_days = size(returns, 1)
    number_of_tail_probabilities = length(tail_probabilities)
    candidate_losses = [
        fill(NaN, number_of_days, length(eta_parameters_by_tail[tail_index]))
        for tail_index in eachindex(tail_probabilities)
    ]
    selected_test_losses = [Float64[] for _ in tail_probabilities]
    saa_test_losses = [Float64[] for _ in tail_probabilities]
    selected_parameters = [Float64[] for _ in tail_probabilities]
    maximum_weight = min(1.0, 5 / subset_size)

    println("seed=$seed started on thread=$(Threads.threadid())")
    for (stage_position, stage) in pairs(stages)
        time() < deadline || return nothing
        training_start = stage.first - estimation_window
        training_start >= 1 || error("insufficient estimation history")
        samples = [Vector(@view returns[day, :]) for day in training_start:(stage.first - 1)]
        sample_weights = signal_weights_by_stage[stage_position]

        for (tail_index, tail_probability) in pairs(tail_probabilities)
            stage_position >= tail_first_stage_positions[tail_index] || continue
            parameter_indices = tail_parameter_indices[tail_index]
            tail_candidate_losses = candidate_losses[tail_index]
            for local_parameter_index in eachindex(parameter_indices)
                global_parameter_index = parameter_indices[local_parameter_index]
                portfolio = solve_capped_portfolio(
                    samples,
                    sample_weights[global_parameter_index],
                    tail_probability,
                    maximum_weight,
                )
                losses = buy_and_hold_losses(portfolio, returns, stage.first, stage.last)
                tail_candidate_losses[stage.first:stage.last, local_parameter_index] .= losses
            end

            if stage_position >= first_test_stage
                tuning_window = tuning_windows[tail_index]
                tuning_indices = (stage.first - tuning_window):(stage.first - 1)
                all(isfinite, @view tail_candidate_losses[tuning_indices, :]) ||
                    error("candidate shadow history is incomplete")
                scores = [
                    risk_adjusted_cost(
                        @view(tail_candidate_losses[tuning_indices, parameter_index]),
                        tail_probability,
                    )
                    for parameter_index in axes(tail_candidate_losses, 2)
                ]
                selected_index = argmin(scores)
                append!(selected_test_losses[tail_index],
                    @view tail_candidate_losses[stage.first:stage.last, selected_index])
                append!(saa_test_losses[tail_index],
                    @view tail_candidate_losses[stage.first:stage.last, end])
                push!(selected_parameters[tail_index], eta_parameters_by_tail[tail_index][selected_index])
            end
        end
        print_monthly_progress && println("  seed=$seed $(stage.label) complete")
    end

    rows = NamedTuple[]
    for (tail_index, tail_probability) in pairs(tail_probabilities)
        selected_losses = selected_test_losses[tail_index]
        saa_losses = saa_test_losses[tail_index]
        selected_cost = risk_adjusted_cost(selected_losses, tail_probability)
        saa_cost = risk_adjusted_cost(saa_losses, tail_probability)
        difference = selected_cost - saa_cost
        chosen = selected_parameters[tail_index]
        push!(rows, (
            seed = seed,
            signal_variant = "shared_50_stock",
            subset_size = subset_size,
            tail_probability = tail_probability,
            estimation_days = estimation_window,
            tuning_days = tuning_windows[tail_index],
            finite_eta_min = first(eta_parameters_by_tail[tail_index][2:end-1]),
            finite_eta_max = last(eta_parameters_by_tail[tail_index][2:end-1]),
            finite_parameter_count = length(eta_parameters_by_tail[tail_index]) - 2,
            test_days = length(selected_losses),
            saa_cost = saa_cost,
            selected_wpf_cost = selected_cost,
            absolute_difference = difference,
            difference_from_saa_pct = 100 * difference / abs(saa_cost),
            saa_wealth = prod(1 .- saa_losses),
            selected_wpf_wealth = prod(1 .- selected_losses),
            saa_selection_share = mean(parameter == Inf for parameter in chosen),
            distinct_selected_parameters = length(unique(chosen)),
            selected_eta_by_month = join(parameter_label.(chosen), ';'),
            tickers = join(tickers, ';'),
        ))
    end
    println("seed=$seed completed")
    return rows
end

function main()
    rows = collect(CSV.File(daily_csv_path))
    isempty(rows) && error("daily return file is empty")
    all_tickers = String.(propertynames(first(rows))[2:end])
    dates = Date.(string.([row[:Date] for row in rows]))
    all_returns = reduce(vcat, [
        permutedims(Float64[row[property] for property in propertynames(first(rows))[2:end]])
        for row in rows
    ])
    @assert issorted(dates) && allunique(dates)
    @assert subset_size <= length(all_tickers)
    @assert estimation_window > 0 && all(tuning_windows .> 0) && test_months > 0

    ranges = month_ranges(dates)
    length(ranges) > test_months || error("not enough calendar months")
    first_test_range = length(ranges) - test_months + 1
    first_test_day = ranges[first_test_range].first
    shadow_target_days = first_test_day .- tuning_windows
    all(shadow_target_days .> estimation_window) || error("not enough warm-up data")
    tail_first_stage_ranges = [
        findfirst(range -> range.first <= target_day <= range.last, ranges)
        for target_day in shadow_target_days
    ]
    any(isnothing, tail_first_stage_ranges) && error("could not locate shadow-history start")
    first_stage_range = minimum(tail_first_stage_ranges)
    stages = ranges[first_stage_range:end]
    first_test_stage = first_test_range - first_stage_range + 1
    tail_first_stage_positions = tail_first_stage_ranges .- first_stage_range .+ 1

    println("Daily WPF pilot: $(length(all_tickers))-stock universe, $(length(subset_seeds)) subsets")
    println("Stages $(first(stages).label) to $(last(stages).label); final $(test_months) months tested")
    println("estimation=$estimation_window days, tuning=$(tuning_windows) days")
    for tail_index in eachindex(tail_probabilities)
        finite_grid = eta_parameters_by_tail[tail_index][2:end-1]
        println(
            "tail=$(tail_probabilities[tail_index]) eta=$(round(first(finite_grid), sigdigits=5))" *
            "..$(round(last(finite_grid), sigdigits=5)) ($(length(finite_grid)) finite + endpoints)",
        )
    end
    println("Julia threads: $(Threads.nthreads()); runtime cap: $max_runtime_seconds seconds")
    started_at = time()
    deadline = started_at + max_runtime_seconds
    println("Precomputing shared 50-stock WPF scenario weights")
    signal_weights_by_stage = precompute_signal_weights(all_returns, stages, deadline)
    println("Shared WPF scenario weights complete after $(round(time() - started_at, digits=1)) seconds")
    results = NamedTuple[]
    results_lock = ReentrantLock()
    Threads.@threads :greedy for seed in subset_seeds
        subset_rows = run_subset(
            seed,
            all_returns,
            all_tickers,
            dates,
            stages,
            signal_weights_by_stage,
            first_test_stage,
            tail_first_stage_positions,
            deadline,
        )
        if isnothing(subset_rows)
            println("Runtime cap reached before seed=$seed completed; discarding that partial subset.")
            continue
        end
        lock(results_lock)
        try
            append!(results, subset_rows)
            sort!(results; by = row -> (row.seed, row.tail_probability))
            CSV.write(output_path, results)
        finally
            unlock(results_lock)
        end
    end
    isempty(results) && error("runtime cap expired before any subset completed")
    println("Wrote $(length(results)) result rows to $output_path")
    println("Elapsed: $(round(time() - started_at, digits = 1)) seconds")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
