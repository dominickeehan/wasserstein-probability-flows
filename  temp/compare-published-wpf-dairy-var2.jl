using LinearAlgebra
using Printf
using Statistics

const PRODUCTS = ["AMF", "BUT", "BMP", "SMP", "WMP"]
const PUBLISHED_WPF_L1_TESTING_COST = 0.0242
const PUBLISHED_WPF_L1_DIFFERENCE_FROM_SAA_PERCENT = -7.5
const PUBLISHED_WPF_L1_DIFFERENCE_FROM_SAA_SE_PERCENT = 5.7

function load_monthly_log_prices(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || error("No dairy-price rows found in $path")

    dates = String[]
    log_prices = Vector{Float64}[]
    for line in lines[2:end]
        fields = split(line, ",")
        length(fields) == 6 || error("Malformed dairy-price row: $line")
        push!(dates, fields[1])
        push!(log_prices, log.(parse.(Float64, fields[2:end])))
    end

    return dates, log_prices
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

function fit_var(samples, lag::Int)
    rows = length(samples) - lag
    rows > 0 || error("Need more observations than lags")
    m = length(samples[1])

    X = zeros(rows, 1 + lag * m)
    Y = zeros(rows, m)
    for row in 1:rows
        target_index = lag + row
        X[row, :] .= design_row(samples, target_index, lag)
        Y[row, :] .= samples[target_index]
    end

    return X \ Y
end

function forecast_var(samples, lag::Int)
    coefficients = fit_var(samples, lag)
    return design_row(samples, length(samples) + 1, lag)' * coefficients |> vec
end

loss(prediction, actual) = sum((prediction .- actual) .^ 2)

function expanding_var_testing_costs(data, lag::Int)
    training_testing_split = ceil(Int, 0.7 * length(data))
    warm_up_period = ceil(Int, 0.7 * training_testing_split) - 1

    warm_up_data = data[1:warm_up_period]
    training_data = data[(warm_up_period + 1):training_testing_split]
    testing_data = data[(training_testing_split + 1):end]

    costs = Float64[]
    for t in eachindex(testing_data)
        samples = [warm_up_data; training_data; testing_data[1:(t - 1)]]
        prediction = forecast_var(samples, lag)
        push!(costs, loss(prediction, testing_data[t]))
    end

    return (
        costs = costs,
        training_testing_split = training_testing_split,
        warm_up_period = warm_up_period,
        testing_T = length(testing_data),
    )
end

function print_row(method, average_cost; difference = nothing)
    if difference === nothing
        @printf("%-24s  %.6f     %s\n", method, average_cost, "")
    else
        @printf("%-24s  %.6f     %+0.2f%%\n", method, average_cost, difference)
    end
end

function main()
    dates, data = load_monthly_log_prices("dairy-prices.csv")
    var1 = expanding_var_testing_costs(data, 1)
    var2 = expanding_var_testing_costs(data, 2)

    var1_average = mean(var1.costs)
    var2_average = mean(var2.costs)
    var2_difference_from_published_wpf =
        (var2_average - PUBLISHED_WPF_L1_TESTING_COST) / PUBLISHED_WPF_L1_TESTING_COST * 100
    var2_difference_from_var1 =
        (var2_average - var1_average) / var1_average * 100

    println("Monthly dairy-price VAR comparison")
    println("Products: " * join(PRODUCTS, ", "))
    println("Data: $(dates[1]) to $(dates[end]) ($(length(data)) monthly observations)")
    println("Warm-up: $(dates[1]) to $(dates[var1.warm_up_period])")
    println("Training-cost period: $(dates[var1.warm_up_period + 1]) to $(dates[var1.training_testing_split])")
    println("Testing period: $(dates[var1.training_testing_split + 1]) to $(dates[end]) ($(var1.testing_T) months)")
    println()
    println("Method                    Avg cost   Difference")
    println("------------------------------------------------")
    print_row("Published WPF L1", PUBLISHED_WPF_L1_TESTING_COST)
    print_row("Expanding VAR(1)", var1_average; difference = (var1_average - PUBLISHED_WPF_L1_TESTING_COST) / PUBLISHED_WPF_L1_TESTING_COST * 100)
    print_row("Expanding VAR(2)", var2_average; difference = var2_difference_from_published_wpf)
    println()
    @printf("VAR(2) vs VAR(1): %+0.2f%%\n", var2_difference_from_var1)
    @printf("Published WPF L1 vs SAA in paper: %+0.1f ± %.1f%%\n", PUBLISHED_WPF_L1_DIFFERENCE_FROM_SAA_PERCENT, PUBLISHED_WPF_L1_DIFFERENCE_FROM_SAA_SE_PERCENT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

