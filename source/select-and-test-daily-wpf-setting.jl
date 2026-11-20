using CSV
using Dates
using Random
using Serialization
using Statistics
using StatsBase

const candidate_cache_path = get(
    ENV,
    "WPF_CANDIDATE_CACHE",
    "data/stock-daily-extended-candidate-losses.jls",
)
const setting_results_path = get(
    ENV,
    "WPF_SETTING_RESULTS",
    "data/stock-daily-heldout-setting-results.csv",
)
const development_start = Date(2022, 1, 1)
const development_end = Date(2023, 12, 31)
const holdout_start = Date(2024, 1, 1)
const holdout_end = Date(2025, 12, 31)
const calibration_seeds = collect(1:10)
const evaluation_seeds = collect(11:20)
const improvement_margins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]
const persistence_lengths = [1, 2, 3]
const tuning_window_candidates = [
    [252, 378, 504, 630, 756],
    [126, 252, 378, 504, 756],
    [126, 189, 252, 378, 504],
]
const bootstrap_repetitions = 2000
const bootstrap_block_length = 63

function empirical_cvar(costs, tail_probability)
    tail_mass = tail_probability * length(costs)
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
    0.1 * mean(costs) + 0.9 * empirical_cvar(costs, tail_probability)

function stages_between(stages, dates, start_date, end_date)
    return [
        stage
        for stage in stages
        if start_date <= dates[stage.first] <= end_date
    ]
end

function pooled_scores(cache, tail_index, selection_seeds, stages, tuning_window)
    parameter_count = size(cache.candidates[first(selection_seeds)].losses_by_tail[tail_index], 2)
    scores = zeros(length(stages), parameter_count)
    tail_probability = cache.tail_probabilities[tail_index]
    for (stage_index, stage) in pairs(stages)
        tuning_indices = (stage.first - tuning_window):(stage.first - 1)
        for parameter_index in 1:parameter_count
            scores[stage_index, parameter_index] = mean(
                risk_adjusted_cost(
                    @view(cache.candidates[seed].losses_by_tail[tail_index][
                        tuning_indices,
                        parameter_index,
                    ]),
                    tail_probability,
                )
                for seed in selection_seeds
            )
        end
    end
    return scores
end

function choose_parameters(scores, margin, persistence)
    saa_index = size(scores, 2)
    choices = fill(saa_index, size(scores, 1))
    qualifying_streak = 0
    for stage_index in axes(scores, 1)
        finite_index = argmin(@view scores[stage_index, 2:(saa_index - 1)]) + 1
        finite_score = scores[stage_index, finite_index]
        saa_score = scores[stage_index, saa_index]
        relative_improvement = (saa_score - finite_score) / abs(saa_score)
        if relative_improvement >= margin
            qualifying_streak += 1
        else
            qualifying_streak = 0
        end
        if qualifying_streak >= persistence
            choices[stage_index] = finite_index
        end
    end
    return choices
end

function realised_losses(cache, tail_index, seeds, stages, choices, capture_start, capture_end)
    output = Dict(seed => Float64[] for seed in seeds)
    saa = Dict(seed => Float64[] for seed in seeds)
    for (stage_index, stage) in pairs(stages)
        stage_date = cache.dates[stage.first]
        capture_start <= stage_date <= capture_end || continue
        for seed in seeds
            losses = cache.candidates[seed].losses_by_tail[tail_index]
            append!(output[seed], @view losses[stage.first:stage.last, choices[stage_index]])
            append!(saa[seed], @view losses[stage.first:stage.last, end])
        end
    end
    return output, saa
end

function result_differences(selected, saa, seeds, tail_probability)
    return [
        risk_adjusted_cost(selected[seed], tail_probability) -
        risk_adjusted_cost(saa[seed], tail_probability)
        for seed in seeds
    ]
end

function choose_development_setting(cache, tail_index, selection_stages, development_stages)
    tail_probability = cache.tail_probabilities[tail_index]
    best = nothing
    development_rows = NamedTuple[]
    for tuning_window in tuning_window_candidates[tail_index]
        scores = pooled_scores(
            cache,
            tail_index,
            calibration_seeds,
            selection_stages,
            tuning_window,
        )
        for margin in improvement_margins, persistence in persistence_lengths
            choices = choose_parameters(scores, margin, persistence)
            selected, saa = realised_losses(
                cache,
                tail_index,
                calibration_seeds,
                selection_stages,
                choices,
                development_start,
                development_end,
            )
            differences = result_differences(
                selected,
                saa,
                calibration_seeds,
                tail_probability,
            )
            row = (
                tuning_window = tuning_window,
                margin = margin,
                persistence = persistence,
                mean_difference = mean(differences),
                median_difference = median(differences),
                wins = count(<(0), differences),
                saa_share = mean(choices .== size(scores, 2)),
            )
            push!(development_rows, row)
            ranking = (
                row.mean_difference,
                row.median_difference,
                -row.wins,
                -row.tuning_window,
                -row.margin,
                -row.persistence,
            )
            if isnothing(best) || ranking < best.ranking
                best = (row = row, ranking = ranking)
            end
        end
    end
    return best.row, development_rows
end

function circular_block_indices(rng, length_of_series, block_length)
    indices = Int[]
    while length(indices) < length_of_series
        block_start = rand(rng, 1:length_of_series)
        append!(
            indices,
            ((block_start - 1 + offset) % length_of_series) + 1
            for offset in 0:(block_length - 1)
        )
    end
    resize!(indices, length_of_series)
    return indices
end

function time_block_interval(selected, saa, seeds, tail_probability, seed)
    rng = MersenneTwister(seed)
    test_days = length(selected[first(seeds)])
    bootstrap_means = zeros(bootstrap_repetitions)
    for repetition in 1:bootstrap_repetitions
        indices = circular_block_indices(rng, test_days, bootstrap_block_length)
        bootstrap_means[repetition] = mean(
            risk_adjusted_cost(@view(selected[subset_seed][indices]), tail_probability) -
            risk_adjusted_cost(@view(saa[subset_seed][indices]), tail_probability)
            for subset_seed in seeds
        )
    end
    return quantile(bootstrap_means, 0.025), quantile(bootstrap_means, 0.975)
end

function main()
    cache = open(deserialize, candidate_cache_path)
    all(haskey(cache.candidates, seed) for seed in [calibration_seeds; evaluation_seeds]) ||
        error("candidate cache does not contain all required seeds")
    selection_stages = stages_between(
        cache.stages,
        cache.dates,
        development_start,
        holdout_end,
    )
    development_stages = stages_between(
        cache.stages,
        cache.dates,
        development_start,
        development_end,
    )
    println("Selecting settings only from $development_start through $development_end")
    println("Final holdout: $holdout_start through $holdout_end, seeds 11:20")

    final_rows = NamedTuple[]
    for tail_index in eachindex(cache.tail_probabilities)
        tail_probability = cache.tail_probabilities[tail_index]
        chosen_setting, _ = choose_development_setting(
            cache,
            tail_index,
            selection_stages,
            development_stages,
        )
        println("tail=$tail_probability frozen setting=$chosen_setting")

        scores = pooled_scores(
            cache,
            tail_index,
            calibration_seeds,
            selection_stages,
            chosen_setting.tuning_window,
        )
        choices = choose_parameters(
            scores,
            chosen_setting.margin,
            chosen_setting.persistence,
        )
        selected, saa = realised_losses(
            cache,
            tail_index,
            evaluation_seeds,
            selection_stages,
            choices,
            holdout_start,
            holdout_end,
        )
        differences = result_differences(
            selected,
            saa,
            evaluation_seeds,
            tail_probability,
        )
        ci_low, ci_high = time_block_interval(
            selected,
            saa,
            evaluation_seeds,
            tail_probability,
            10_000 + tail_index,
        )
        holdout_stage_indices = [
            stage_index
            for (stage_index, stage) in pairs(selection_stages)
            if holdout_start <= cache.dates[stage.first] <= holdout_end
        ]
        holdout_choices = choices[holdout_stage_indices]
        eta_grid = cache.eta_parameters_by_tail[tail_index]
        for (seed_position, seed) in pairs(evaluation_seeds)
            saa_cost = risk_adjusted_cost(saa[seed], tail_probability)
            selected_cost = risk_adjusted_cost(selected[seed], tail_probability)
            push!(final_rows, (
                seed = seed,
                tail_probability = tail_probability,
                development_start = development_start,
                development_end = development_end,
                holdout_start = holdout_start,
                holdout_end = holdout_end,
                tuning_window = chosen_setting.tuning_window,
                improvement_margin = chosen_setting.margin,
                persistence = chosen_setting.persistence,
                development_mean_difference = chosen_setting.mean_difference,
                saa_cost = saa_cost,
                selected_cost = selected_cost,
                absolute_difference = differences[seed_position],
                difference_from_saa_pct =
                    100 * differences[seed_position] / abs(saa_cost),
                saa_wealth = prod(1 .- saa[seed]),
                selected_wealth = prod(1 .- selected[seed]),
                aggregate_mean_difference = mean(differences),
                aggregate_median_difference = median(differences),
                aggregate_wins = count(<(0), differences),
                aggregate_ci_low = ci_low,
                aggregate_ci_high = ci_high,
                holdout_saa_selection_share = mean(holdout_choices .== length(eta_grid)),
                selected_eta_by_month = join([
                    parameter == Inf ? "Inf" : string(round(parameter, sigdigits = 5))
                    for parameter in eta_grid[holdout_choices]
                ], ';'),
                tickers = join(cache.candidates[seed].tickers, ';'),
            ))
        end
        println(
            "tail=$tail_probability HOLDOUT mean=$(mean(differences)) " *
            "wins=$(count(<(0), differences))/$(length(differences)) " *
            "CI=[$ci_low,$ci_high]",
        )
    end
    CSV.write(setting_results_path, final_rows)
    println("Wrote $setting_results_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
