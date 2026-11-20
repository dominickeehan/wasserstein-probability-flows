using Serialization

include("test-daily-stock-return-subsets-pilot.jl")

const candidate_cache_path = get(
    ENV,
    "WPF_CANDIDATE_CACHE",
    "data/stock-daily-extended-candidate-losses.jls",
)
const development_start_date = Date(get(ENV, "WPF_DEVELOPMENT_START", "2022-01-01"))
const maximum_candidate_tuning_window = parse(
    Int,
    get(ENV, "WPF_MAX_TUNING_DAYS", "756"),
)

function generate_candidate_losses(seed, all_returns, all_tickers, stages,
        signal_weights_by_stage, deadline)
    time() < deadline || return nothing
    indices = subset_indices(seed, size(all_returns, 2))
    returns = all_returns[:, indices]
    number_of_days = size(returns, 1)
    losses_by_tail = [
        fill(NaN, number_of_days, length(eta_parameters_by_tail[tail_index]))
        for tail_index in eachindex(tail_probabilities)
    ]
    maximum_weight = min(1.0, 5 / subset_size)

    println("candidate seed=$seed started on thread=$(Threads.threadid())")
    for (stage_position, stage) in pairs(stages)
        time() < deadline || return nothing
        training_start = stage.first - estimation_window
        samples = [Vector(@view returns[day, :]) for day in training_start:(stage.first - 1)]
        signal_weights = signal_weights_by_stage[stage_position]

        for (tail_index, tail_probability) in pairs(tail_probabilities)
            tail_losses = losses_by_tail[tail_index]
            parameter_indices = tail_parameter_indices[tail_index]
            for local_parameter_index in eachindex(parameter_indices)
                portfolio = solve_capped_portfolio(
                    samples,
                    signal_weights[parameter_indices[local_parameter_index]],
                    tail_probability,
                    maximum_weight,
                )
                tail_losses[stage.first:stage.last, local_parameter_index] .=
                    buy_and_hold_losses(portfolio, returns, stage.first, stage.last)
            end
        end
    end
    println("candidate seed=$seed completed")
    return (
        indices = indices,
        tickers = all_tickers[indices],
        losses_by_tail = losses_by_tail,
    )
end

function checkpoint_candidate_cache(path, dates, all_tickers, stages, candidates)
    open(path, "w") do output
        serialize(output, (
            dates = dates,
            all_tickers = all_tickers,
            stages = stages,
            tail_probabilities = tail_probabilities,
            eta_parameters_by_tail = eta_parameters_by_tail,
            estimation_window = estimation_window,
            subset_size = subset_size,
            signal_variant = "shared_50_stock",
            candidates = candidates,
        ))
    end
end

function build_cache_main()
    rows = collect(CSV.File(daily_csv_path))
    isempty(rows) && error("daily return file is empty")
    all_tickers = String.(propertynames(first(rows))[2:end])
    dates = Date.(string.([row[:Date] for row in rows]))
    all_returns = reduce(vcat, [
        permutedims(Float64[row[property] for property in propertynames(first(rows))[2:end]])
        for row in rows
    ])
    ranges = month_ranges(dates)
    development_start_day = findfirst(>=(development_start_date), dates)
    isnothing(development_start_day) && error("development start is outside the dataset")
    shadow_target_day = development_start_day - maximum_candidate_tuning_window
    shadow_target_day > estimation_window || error("not enough candidate warm-up history")
    first_stage_range = findfirst(
        range -> range.first <= shadow_target_day <= range.last,
        ranges,
    )
    isnothing(first_stage_range) && error("could not locate first candidate stage")
    stages = ranges[first_stage_range:end]

    started_at = time()
    deadline = started_at + max_runtime_seconds
    println(
        "Building candidate cache for $(length(subset_seeds)) subsets, " *
        "$(first(stages).label) to $(last(stages).label)",
    )
    println("Precomputing shared signal weights")
    signal_weights_by_stage = precompute_signal_weights(all_returns, stages, deadline)
    println("Signal weights completed after $(round(time() - started_at, digits=1)) seconds")

    candidates = Dict{Int, Any}()
    candidates_lock = ReentrantLock()
    Threads.@threads :greedy for seed in subset_seeds
        candidate = generate_candidate_losses(
            seed,
            all_returns,
            all_tickers,
            stages,
            signal_weights_by_stage,
            deadline,
        )
        isnothing(candidate) && continue
        lock(candidates_lock)
        try
            candidates[seed] = candidate
            checkpoint_candidate_cache(
                candidate_cache_path,
                dates,
                all_tickers,
                stages,
                candidates,
            )
        finally
            unlock(candidates_lock)
        end
    end
    length(candidates) == length(subset_seeds) ||
        error("runtime cap completed only $(length(candidates)) of $(length(subset_seeds)) subsets")
    println(
        "Wrote $candidate_cache_path with $(length(candidates)) subsets in " *
        "$(round(time() - started_at, digits=1)) seconds",
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_cache_main()
end
