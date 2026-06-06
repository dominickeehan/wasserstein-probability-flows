using LinearAlgebra
using Printf
using Statistics

include("ipopt-wpf.jl")

const PRODUCTS = ["amf", "bmp", "but", "smp", "wmp"]

struct EventRecord
    trading_event::Int
    date::String
    prices::Vector{Float64}
    log_prices::Vector{Float64}
end

struct BacktestOptions
    dairy_repo::String
    first_observation::Int
    data_fraction::Float64
    lookback::Int
    parameter_tuning_window::Int
    test_size::Int
    training_fraction::Float64
    lambdas::Vector{Float64}
    output_dir::String
    ridge::Float64
    ipopt_max_iter::Int
end

function paper_wpf_lambdas()
    return vcat(
        collect(10.0:10.0:100.0),
        collect(200.0:100.0:1000.0),
        collect(2000.0:1000.0:10000.0),
        [Inf],
    )
end

function parse_float_or_inf(value::AbstractString)
    lowercase(strip(value)) == "inf" && return Inf
    return parse(Float64, value)
end

function parse_lambdas(value::AbstractString)
    lowercase(strip(value)) == "paper" && return paper_wpf_lambdas()
    return parse_float_or_inf.(split(value, ","))
end

function parse_args(args)
    opts = Dict{String, String}(
        "--dairy-repo" => "../dairy-analytics-master",
        "--first-observation" => "101",
        "--data-fraction" => "1.0",
        "--lookback" => "0",
        "--parameter-tuning-window" => "24",
        "--validation-size" => "24",
        "--test-size" => "0",
        "--training-fraction" => "0.7",
        "--lambdas" => "paper",
        "--output-dir" => "results",
        "--ridge" => "0.0",
        "--ipopt-max-iter" => "500",
    )

    i = 1
    saw_parameter_tuning_window = false
    while i <= length(args)
        key = args[i]
        haskey(opts, key) || error("Unknown argument: $key")
        i == length(args) && error("Missing value for argument: $key")
        if key == "--parameter-tuning-window"
            saw_parameter_tuning_window = true
        end
        opts[key] = args[i + 1]
        i += 2
    end

    parameter_tuning_window = saw_parameter_tuning_window ?
        parse(Int, opts["--parameter-tuning-window"]) :
        parse(Int, opts["--validation-size"])

    return BacktestOptions(
        opts["--dairy-repo"],
        parse(Int, opts["--first-observation"]),
        parse(Float64, opts["--data-fraction"]),
        parse(Int, opts["--lookback"]),
        parameter_tuning_window,
        parse(Int, opts["--test-size"]),
        parse(Float64, opts["--training-fraction"]),
        parse_lambdas(opts["--lambdas"]),
        opts["--output-dir"],
        parse(Float64, opts["--ridge"]),
        parse(Int, opts["--ipopt-max-iter"]),
    )
end

function load_gdt_events(path::AbstractString; first_observation::Int)
    lines = readlines(path)
    length(lines) >= 2 || error("No event data found at $path")

    records = EventRecord[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ",")
        length(fields) >= 7 || error("Malformed GDT row: $line")
        prices = parse.(Float64, fields[3:7])
        push!(records, EventRecord(parse(Int, fields[1]), fields[2], prices, log.(prices)))
    end

    first_observation <= length(records) || error("--first-observation is beyond the data length")
    return records[first_observation:end]
end

function truncate_records(records, options::BacktestOptions)
    0 < options.data_fraction <= 1 || error("--data-fraction must be in (0, 1]")
    options.data_fraction == 1 && return records

    kept = max(1, floor(Int, options.data_fraction * length(records)))
    return records[1:kept]
end

function design_row(samples, target_index::Int, lag::Int)
    m = length(samples[1])
    row = zeros(1 + lag * m)
    row[1] = 1.0

    col = 2
    for lag_index in 1:lag
        row[col:col + m - 1] .= samples[target_index - lag_index]
        col += m
    end

    return row
end

function fit_var(samples, lag::Int, weights::Vector{Float64}; ridge::Float64)
    n = length(samples)
    m = length(samples[1])
    rows = n - lag
    rows > 0 || error("Need more observations than lags")
    length(weights) == rows || error("Expected $rows regression weights, got $(length(weights))")

    X = zeros(rows, 1 + lag * m)
    Y = zeros(rows, m)

    for row_index in 1:rows
        target_index = lag + row_index
        X[row_index, :] .= design_row(samples, target_index, lag)
        Y[row_index, :] .= samples[target_index]
    end

    normalized_weights = max.(weights, 0.0)
    weight_sum = sum(normalized_weights)
    weight_sum <= 0 && error("Regression weights must have positive mass")
    normalized_weights ./= weight_sum

    W = Diagonal(normalized_weights)
    penalty = Matrix{Float64}(I, size(X, 2), size(X, 2))
    penalty[1, 1] = 0.0
    coefficients = (X' * W * X + ridge * penalty) \ (X' * W * Y)

    return coefficients
end

function forecast_var(samples, lag::Int; weights = nothing, ridge::Float64)
    rows = length(samples) - lag
    regression_weights = weights === nothing ? fill(1 / rows, rows) : weights
    coefficients = fit_var(samples, lag, regression_weights; ridge)
    return design_row(samples, length(samples) + 1, lag)' * coefficients |> vec
end

function wpf_regression_observations(samples, lag::Int)
    observations = Vector{Vector{Vector{Float64}}}()
    for target_index in (lag + 1):length(samples)
        push!(observations, [samples[target_index - lag_index] for lag_index in 1:lag])
        push!(observations[end], samples[target_index])
    end
    return observations
end

function regression_observation_distance(a, b)
    length(a) == length(b) || error("Cannot compare WPF observations with different lag lengths")
    # L1 ground metric, matching the dairy-prices WPF experiment.
    return sum(norm(a[i] - b[i], 1) for i in eachindex(a))
end

function forecast_wpf_var(samples, lag::Int, lambda::Float64, options::BacktestOptions)
    if options.lookback > 0 && length(samples) > options.lookback
        samples = samples[end - options.lookback + 1:end]
    end

    if lambda == Inf
        return forecast_var(samples, lag; ridge = options.ridge)
    end

    observations = wpf_regression_observations(samples, lag)
    weights = ipopt_wpf_weights(
        observations,
        lambda,
        regression_observation_distance;
        max_iter = options.ipopt_max_iter,
    )

    return forecast_var(samples, lag; weights, ridge = options.ridge)
end

function squared_l2_loss(prediction, actual)
    return sum((prediction .- actual) .^ 2)
end

function mean_absolute_percentage_price_error(prediction_log, actual_log)
    prediction = exp.(prediction_log)
    actual = exp.(actual_log)
    return mean(abs.(prediction .- actual) ./ actual) * 100
end

function samples_before(records, target_index::Int)
    return [record.log_prices for record in records[1:target_index - 1]]
end

function build_backtest_split(records, options::BacktestOptions)
    0 < options.training_fraction < 1 || error("--training-fraction must be between 0 and 1")

    training_testing_split = ceil(Int, options.training_fraction * length(records))
    warm_up_period = ceil(Int, options.training_fraction * training_testing_split) - 1

    training_indices = collect((warm_up_period + 1):training_testing_split)
    all_testing_indices = collect((training_testing_split + 1):length(records))

    isempty(training_indices) && error("No training stages after applying the warm-up split")
    isempty(all_testing_indices) && error("No testing stages after applying the training split")
    length(training_indices) >= options.parameter_tuning_window || error(
        "The rolling fitting approach needs at least $(options.parameter_tuning_window) " *
        "training stages before testing; found $(length(training_indices)).",
    )

    if options.test_size > 0
        all_testing_indices = all_testing_indices[1:min(options.test_size, length(all_testing_indices))]
    end

    return (
        warm_up_period = warm_up_period,
        training_indices = training_indices,
        testing_indices = all_testing_indices,
    )
end

function compute_wpf_parameter_panel(records, stage_indices, lag::Int, options::BacktestOptions; label::String)
    costs = zeros(length(stage_indices), length(options.lambdas))
    predictions = Array{Vector{Float64}}(undef, length(stage_indices), length(options.lambdas))

    total_solves = length(stage_indices) * length(options.lambdas)
    solve_count = 0
    for (stage_row, target_index) in enumerate(stage_indices)
        samples = samples_before(records, target_index)
        for (lambda_index, lambda) in enumerate(options.lambdas)
            prediction = forecast_wpf_var(samples, lag, lambda, options)
            predictions[stage_row, lambda_index] = prediction
            costs[stage_row, lambda_index] = squared_l2_loss(prediction, records[target_index].log_prices)

            solve_count += 1
            if solve_count == 1 || solve_count % 100 == 0 || solve_count == total_solves
                println("$label WPF parameter panel: $solve_count / $total_solves fits")
            end
        end
    end

    return (costs = costs, predictions = predictions)
end

function validate_wpf_inf_matches_var(records, stage_indices, panel, lag::Int, options::BacktestOptions)
    inf_index = findfirst(isinf, options.lambdas)
    inf_index === nothing && return

    max_abs_difference = 0.0
    for (stage_row, target_index) in enumerate(stage_indices)
        samples = samples_before(records, target_index)
        dairy_prediction = forecast_var(samples, lag; ridge = options.ridge)
        wpf_prediction = panel.predictions[stage_row, inf_index]
        max_abs_difference = max(max_abs_difference, maximum(abs.(dairy_prediction .- wpf_prediction)))
    end

    if max_abs_difference > 1e-6
        error(
            "WPF lambda=Inf should match the expanding VAR($lag) baseline, " *
            "but max abs log-price difference was $max_abs_difference",
        )
    end

    println("Verified WPF lambda=Inf matches expanding VAR($lag) baseline; max abs difference = $(csv_number(max_abs_difference))")
end

function summarize_lambdas(lambdas)
    formatted = unique(format_lambda.(lambdas))
    length(formatted) == 1 && return formatted[1]
    return "rolling"
end

function evaluate_wpf_rolling(records, training_indices, testing_indices, method::String, lag::Int, options::BacktestOptions)
    stage_indices = [training_indices; testing_indices]
    panel = compute_wpf_parameter_panel(records, stage_indices, lag, options; label = method)
    validate_wpf_inf_matches_var(records, stage_indices, panel, lag, options)

    training_T = length(training_indices)
    rows = []
    losses = Float64[]
    absolute_percentage_errors = Float64[]

    for (test_row, target_index) in enumerate(testing_indices)
        previous_end = training_T + test_row - 1
        previous_start = previous_end - options.parameter_tuning_window + 1
        previous_start >= 1 || error("Not enough previous stages to fit lambda")

        rolling_costs = vec(sum(panel.costs[previous_start:previous_end, :], dims = 1))
        lambda_index = argmin(rolling_costs)
        lambda = options.lambdas[lambda_index]
        prediction = panel.predictions[training_T + test_row, lambda_index]
        actual = records[target_index].log_prices

        push!(losses, squared_l2_loss(prediction, actual))
        push!(absolute_percentage_errors, mean_absolute_percentage_price_error(prediction, actual))
        push!(rows, (target_index = target_index, method = method, lambda = lambda, prediction = prediction))
    end

    return (
        method = method,
        lambda = summarize_lambdas([row.lambda for row in rows]),
        mean_l2_log_loss = mean(losses),
        rmse_log = sqrt(mean(losses) / length(PRODUCTS)),
        mape_price = mean(absolute_percentage_errors),
        rows = rows,
    )
end

function evaluate_method(records, target_indices, method::String, predictor)
    rows = []
    losses = Float64[]
    absolute_percentage_errors = Float64[]

    for target_index in target_indices
        prediction, lambda = predictor(target_index)
        actual = records[target_index].log_prices
        push!(losses, squared_l2_loss(prediction, actual))
        push!(absolute_percentage_errors, mean_absolute_percentage_price_error(prediction, actual))
        push!(rows, (target_index = target_index, method = method, lambda = lambda, prediction = prediction))
    end

    return (
        method = method,
        lambda = summarize_lambdas([row.lambda for row in rows]),
        mean_l2_log_loss = mean(losses),
        rmse_log = sqrt(mean(losses) / length(PRODUCTS)),
        mape_price = mean(absolute_percentage_errors),
        rows = rows,
    )
end

function ensure_backtest_is_feasible(records, options::BacktestOptions)
    minimum_required = max(options.lookback, 3) + options.parameter_tuning_window + 1
    length(records) >= minimum_required || error(
        "Need at least $minimum_required usable records; found $(length(records)). " *
        "Reduce --lookback or --parameter-tuning-window.",
    )
end

function format_lambda(lambda)
    lambda isa AbstractString && return lambda
    lambda == Inf && return "Inf"
    isapprox(lambda, round(lambda); atol = 1e-8, rtol = 1e-8) && return string(round(Int, lambda))
    return string(lambda)
end

function csv_number(value)
    return @sprintf("%.10g", value)
end

function write_summary(path, summaries)
    open(path, "w") do io
        println(io, "method,lambda,mean_l2_log_loss,rmse_log,mape_price_percent")
        for summary in summaries
            println(
                io,
                join([
                    summary.method,
                    format_lambda(summary.lambda),
                    csv_number(summary.mean_l2_log_loss),
                    csv_number(summary.rmse_log),
                    csv_number(summary.mape_price),
                ], ","),
            )
        end
    end
end

function write_predictions(path, records, summaries)
    header = [
        "trading_event",
        "date",
        "method",
        "lambda",
        ["actual_$product" for product in PRODUCTS]...,
        ["predicted_$product" for product in PRODUCTS]...,
    ]

    open(path, "w") do io
        println(io, join(header, ","))
        for summary in summaries
            for row in summary.rows
                record = records[row.target_index]
                values = [
                    string(record.trading_event),
                    record.date,
                    row.method,
                    format_lambda(row.lambda),
                    csv_number.(record.prices)...,
                    csv_number.(exp.(row.prediction))...,
                ]
                println(io, join(values, ","))
            end
        end
    end
end

function print_summary_table(summaries)
    println()
    println("Backtest summary")
    println("method, lambda, mean_l2_log_loss, rmse_log, mape_price_percent")
    for summary in summaries
        @printf(
            "%s, %s, %.6f, %.6f, %.3f\n",
            summary.method,
            format_lambda(summary.lambda),
            summary.mean_l2_log_loss,
            summary.rmse_log,
            summary.mape_price,
        )
    end
end

function main(args)
    options = parse_args(args)
    dairy_data_path = joinpath(options.dairy_repo, "data", "processed", "gdt_events.csv")
    records = load_gdt_events(dairy_data_path; first_observation = options.first_observation)
    records = truncate_records(records, options)
    ensure_backtest_is_feasible(records, options)

    split = build_backtest_split(records, options)
    training_indices = split.training_indices
    testing_indices = split.testing_indices

    println("Loaded $(length(records)) GDT events from $dairy_data_path")
    if options.data_fraction < 1
        println("Using first $(round(options.data_fraction * 100; digits = 1))% of post-warmup records")
    end
    println("Warm-up events: $(records[1].date) to $(records[split.warm_up_period].date)")
    println("Training-cost events: $(records[first(training_indices)].date) to $(records[last(training_indices)].date)")
    println("Test events: $(records[first(testing_indices)].date) to $(records[last(testing_indices)].date)")
    println("Rolling lambda tuning window: $(options.parameter_tuning_window) stages")
    if options.lookback > 0
        println("WPF lookback: $(options.lookback) observations")
    else
        println("WPF lookback: expanding history")
    end
    println("WPF lambda grid: " * join(format_lambda.(options.lambdas), ", "))

    summaries = []

    push!(
        summaries,
        evaluate_method(records, testing_indices, "dairy_var2", target_index -> begin
            samples = samples_before(records, target_index)
            (forecast_var(samples, 2; ridge = options.ridge), Inf)
        end),
    )

    push!(
        summaries,
        evaluate_wpf_rolling(records, training_indices, testing_indices, "wpf_var2_ipopt", 2, options),
    )

    mkpath(options.output_dir)
    summary_path = joinpath(options.output_dir, "wpf_ipopt_vs_dairy_analytics_summary.csv")
    prediction_path = joinpath(options.output_dir, "wpf_ipopt_vs_dairy_analytics_predictions.csv")
    write_summary(summary_path, summaries)
    write_predictions(prediction_path, records, summaries)

    print_summary_table(summaries)
    println()
    println("Wrote summary to $summary_path")
    println("Wrote predictions to $prediction_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
