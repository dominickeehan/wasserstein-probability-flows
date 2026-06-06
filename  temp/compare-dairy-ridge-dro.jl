using CSV
using Ipopt
using JuMP
using LinearAlgebra
import MathOptInterface as MOI
using Printf
using Statistics

const PRODUCTS = [:AMF, :BUT, :BMP, :SMP, :WMP]

const IPOPT_ACCEPTABLE_TERMINATION_STATUSES = Set([
    MOI.OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_OPTIMAL,
    MOI.ALMOST_LOCALLY_SOLVED,
])

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
sem(x) = std(x) / sqrt(length(x))
loss_function(prediction, realised) = norm(prediction - realised, 2)^2

function paired_parameter_grid(center_parameters, ridge_parameters)
    length(center_parameters) == length(ridge_parameters) ||
        error("Center and ridge parameter grids must have the same length")

    n = length(center_parameters)
    stride = findfirst(stride -> gcd(stride, n) == 1, max(1, n ÷ 2):n-1)
    stride === nothing && (stride = 1)
    ridge_indices = [mod((i - 1) * stride, n) + 1 for i in 1:n]
    return [(center_parameters[i], ridge_parameters[ridge_indices[i]]) for i in 1:n]
end

function unique_integer_log_grid(start, stop, len)
    raw_len = len
    values = Int[]
    while length(values) < len
        values = unique(ceil.(Int, LogRange(start, stop, raw_len)))
        raw_len *= 2
    end

    indices = round.(Int, LinRange(1, length(values), len))
    length(unique(indices)) == len || error("Failed to construct a unique integer log grid")
    return values[indices]
end

function load_dairy_log_prices(path::AbstractString)
    data = CSV.File(path)
    return [log.(Float64.([getproperty(row, product) for product in PRODUCTS])) for row in data]
end

function fit_weighted_AR1_ridge_model(time_series, weights, ridge)
    m = length(time_series[1])
    inputs = time_series[1:end-1]
    outputs = time_series[2:end]
    n = length(inputs)

    X = hcat(ones(n), stack(inputs)')
    Y = stack(outputs)'
    W = Diagonal(weights ./ sum(weights))
    penalty = Diagonal([0.0; ones(m)])

    coefficients = (X' * W * X + ridge * penalty) \ (X' * W * Y)
    μ = coefficients[1, :]
    A = coefficients[2:end, :]'
    return μ, A
end

function ridge_forecast(samples, weights, ridge)
    μ, A = fit_weighted_AR1_ridge_model(samples, weights, ridge)
    return μ + A * samples[end]
end

function saa_weights(transition_count)
    return fill(1 / transition_count, transition_count)
end

function windowing_weights(transition_count, window_length)
    active_window_length = min(window_length, transition_count)
    weights = zeros(transition_count)
    weights[(transition_count - active_window_length + 1):transition_count] .= 1 / active_window_length
    return weights
end

function smoothing_weights(transition_count, decay)
    decay == 0 && return saa_weights(transition_count)
    weights = [decay * (1 - decay)^(lag - 1) for lag in transition_count:-1:1]
    return weights ./ sum(weights)
end

function dlba_wasserstein_weights(wasserstein_order, transition_count, rho_over_epsilon)
    rho_over_epsilon == 0 && return saa_weights(transition_count)
    if rho_over_epsilon >= 1
        weights = zeros(transition_count)
        weights[end] = 1
        return weights
    elseif rho_over_epsilon < 0
        error("rho_over_epsilon must be nonnegative")
    end

    epsilon = 10.0
    rho = rho_over_epsilon * epsilon
    ages = [(transition_count - t + 1)^wasserstein_order for t in 1:transition_count]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "sb", "yes")
    set_optimizer_attribute(model, "max_iter", 500)

    @variable(model, 0 <= weights[1:transition_count] <= 1)
    @constraint(model, sum(weights) == 1)
    @constraint(
        model,
        sum(weights[t] * ages[t] for t in 1:transition_count) * rho^wasserstein_order <= epsilon^wasserstein_order
    )
    @NLobjective(
        model,
        Max,
        (1 / sum(weights[t]^2 for t in 1:transition_count)) *
        (epsilon - sum(weights[t] * ages[t] for t in 1:transition_count)^(1 / wasserstein_order) * rho)^(2 * wasserstein_order)
    )

    for t in 1:transition_count
        set_start_value(weights[t], 1 / transition_count)
    end

    optimize!(model)

    status = termination_status(model)
    if !(status in IPOPT_ACCEPTABLE_TERMINATION_STATUSES)
        error("Ipopt failed to solve DLBA weight model: termination_status=$status")
    end

    result = max.(value.(weights), 0.0)
    return result ./ sum(result)
end

function cached_dlba_wasserstein_weights!(cache, wasserstein_order, transition_count, rho_over_epsilon)
    key = (wasserstein_order, transition_count, rho_over_epsilon)
    return get!(cache, key) do
        dlba_wasserstein_weights(wasserstein_order, transition_count, rho_over_epsilon)
    end
end

function wpf_transition_distance(a, b)
    return norm(a[1] - b[1], 1) + norm(a[2] - b[2], 1)
end

function wpf_transition_weights(samples, lambda)
    observations = [(samples[i], samples[i+1]) for i in 1:length(samples)-1]
    T = length(observations)
    T > 0 || error("WPF ridge needs at least two samples")

    if lambda == Inf
        return fill(1 / T, T)
    elseif lambda == 0
        weights = zeros(T)
        weights[end] = 1
        return weights
    elseif lambda < 0
        error("lambda must be nonnegative")
    end

    distances = [wpf_transition_distance(observations[i], observations[j]) for i in 1:T, j in 1:T]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "sb", "yes")
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "tol", 1e-7)

    @variable(model, 0 <= p[1:T, 1:2] <= 1)
    @variable(model, 1e-8 <= p_diag[1:T] <= 1)
    @variable(model, 0 <= gamma[1:T, 1:T] <= 1)

    for i in 1:T
        set_start_value(p[i, 1], 1 / T)
        set_start_value(p[i, 2], 1 / T)
        set_start_value(p_diag[i], 1 / T)
        for j in 1:T
            set_start_value(gamma[i, j], 0.0)
        end
    end

    @constraint(model, sum(p[i, 1] for i in 1:T) == 1)
    @constraint(model, sum(p[i, 2] for i in 1:T) == 1)

    for i in 1:T, j in 1:i
        fix(gamma[i, j], 0.0; force = true)
    end

    for t in 1:T
        @constraint(model, p[t, 1] + sum(gamma[i, t] for i in 1:T) == p_diag[t])
        @constraint(model, p_diag[t] == p[t, 2] + sum(gamma[t, j] for j in 1:T))
    end

    transport_cost = sum(distances[i, j] * gamma[i, j] for i in 1:T, j in 1:T)
    @objective(model, Max, sum(log(p_diag[t]) for t in 1:T) - lambda * transport_cost)

    optimize!(model)

    status = termination_status(model)
    if !(status in IPOPT_ACCEPTABLE_TERMINATION_STATUSES)
        error("Ipopt failed to solve WPF weight model: termination_status=$status")
    end

    weights = [max(value(p[i, 2]), 0.0) for i in 1:T]
    total_weight = sum(weights)
    total_weight <= 0 && error("Ipopt returned zero terminal probability mass")
    return weights ./ total_weight
end

function cached_wpf_transition_weights!(cache, samples, lambda)
    key = (length(samples) - 1, lambda)
    return get!(cache, key) do
        wpf_transition_weights(samples, lambda)
    end
end

function choose_parameter_costs(parameter_costs, training_T, testing_T, parameter_tuning_window)
    realised_costs = Float64[]
    selected_parameters = Int[]

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        previous_start >= 1 || error("Not enough previous stages for the tuning window")

        parameter_index = argmin(vec(sum(parameter_costs[previous_start:previous_end, :], dims = 1)))
        push!(selected_parameters, parameter_index)
        push!(realised_costs, parameter_costs[training_T + t, parameter_index])
    end

    final_parameter_index = argmin(vec(sum(parameter_costs[end-(parameter_tuning_window-1):end, :], dims = 1)))
    return realised_costs, final_parameter_index, selected_parameters
end

function evaluate_ridge_family(name, parameters, make_weights, warm_up_data, training_data, testing_data, parameter_tuning_window)
    training_T = length(training_data)
    testing_T = length(testing_data)
    parameter_costs = zeros(training_T + testing_T, length(parameters))

    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        transition_count = length(samples) - 1
        for (i, parameter) in enumerate(parameters)
            center_parameter, ridge = parameter
            weights = make_weights(samples, transition_count, center_parameter)
            prediction = ridge_forecast(samples, weights, ridge)
            parameter_costs[t, i] = loss_function(prediction, training_data[t])
        end
        println("$name training stage $t / $training_T")
    end

    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        transition_count = length(samples) - 1
        for (i, parameter) in enumerate(parameters)
            center_parameter, ridge = parameter
            weights = make_weights(samples, transition_count, center_parameter)
            prediction = ridge_forecast(samples, weights, ridge)
            parameter_costs[training_T + t, i] = loss_function(prediction, testing_data[t])
        end
        println("$name testing stage $t / $testing_T")
    end

    costs, final_parameter_index, selected_parameter_indices =
        choose_parameter_costs(parameter_costs, training_T, testing_T, parameter_tuning_window)

    return costs, parameters[final_parameter_index], parameters[unique(selected_parameter_indices)]
end

function evaluate_plain_saa(warm_up_data, training_data, testing_data)
    realised_costs = Float64[]
    for t in eachindex(testing_data)
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        transition_count = length(samples) - 1
        prediction = ridge_forecast(samples, saa_weights(transition_count), 0.0)
        push!(realised_costs, loss_function(prediction, testing_data[t]))
    end
    return realised_costs
end

function print_summary_row(method, costs, baseline_costs, baseline_average_cost, parameter, selected_parameters)
    average_cost = mean(costs)
    difference = (average_cost - baseline_average_cost) / baseline_average_cost * 100
    difference_se = sem(costs - baseline_costs) / baseline_average_cost * 100
    cost_se = sem(costs)

    println(join([
        method,
        @sprintf("%.10g", average_cost),
        @sprintf("%.10g", cost_se),
        @sprintf("%.4f", difference),
        @sprintf("%.4f", difference_se),
        string(parameter),
        join(selected_parameters, ";"),
    ], ","))
end

function main()
    data = load_dairy_log_prices("dairy-prices.csv")

    training_testing_split = ceil(Int, 0.7 * length(data))
    warm_up_period = ceil(Int, 0.7 * training_testing_split) - 1
    warm_up_data = data[1:warm_up_period]
    training_data = data[warm_up_period+1:training_testing_split]
    testing_data = data[training_testing_split+1:end]
    parameter_tuning_window = 2 * 12
    parameter_budget = 31
    run_dlba_ridge = true
    run_wpf_ridge = false

    window_parameters = unique_integer_log_grid(10, length(data), parameter_budget)
    smoothing_parameters = [0.0; collect(LogRange(1e-4, 0.9, parameter_budget - 1))]
    dlba_parameters = [0.0; collect(LogRange(1e-4, 1e-1, parameter_budget - 1))]
    ridge_parameters = [0.0; collect(LogRange(1e-8, 1e1, parameter_budget - 1))]
    dlba_wasserstein_order = 2.0

    saa_ridge_parameters = [(0.0, ridge) for ridge in ridge_parameters]
    windowing_ridge_parameters = paired_parameter_grid(window_parameters, ridge_parameters)
    smoothing_ridge_parameters = paired_parameter_grid(smoothing_parameters, ridge_parameters)
    dlba_ridge_parameters = paired_parameter_grid(dlba_parameters, ridge_parameters)

    println("Loaded $(length(data)) monthly dairy observations")
    println("Testing observations: $(length(testing_data))")
    println("Parameter budget per tuned method: $parameter_budget")
    println("SAA ridge candidates: $(length(saa_ridge_parameters))")
    println("Windowing ridge candidates: $(length(windowing_ridge_parameters))")
    println("Smoothing ridge candidates: $(length(smoothing_ridge_parameters))")
    println("DLBA ridge candidates: $(run_dlba_ridge ? parameter_budget : "skipped")")
    println("DLBA rho/epsilon range: 0 plus logRange(1e-4, 1e-1, $(parameter_budget - 1))")
    println("WPF ridge candidates: $(run_wpf_ridge ? parameter_budget : "skipped")")
    println("DLBA weighting order: W$(Int(dlba_wasserstein_order))")

    plain_saa_costs = evaluate_plain_saa(warm_up_data, training_data, testing_data)
    plain_saa_average_cost = mean(plain_saa_costs)

    saa_costs, saa_parameter, saa_selected_parameters = evaluate_ridge_family(
        "SAA ridge",
        saa_ridge_parameters,
        (_, transition_count, _) -> saa_weights(transition_count),
        warm_up_data,
        training_data,
        testing_data,
        parameter_tuning_window,
    )
    saa_average_cost = mean(saa_costs)

    windowing_costs, windowing_parameter, windowing_selected_parameters = evaluate_ridge_family(
        "Windowing ridge",
        windowing_ridge_parameters,
        (_, transition_count, window_length) -> windowing_weights(transition_count, window_length),
        warm_up_data,
        training_data,
        testing_data,
        parameter_tuning_window,
    )

    smoothing_costs, smoothing_parameter, smoothing_selected_parameters = evaluate_ridge_family(
        "Smoothing ridge",
        smoothing_ridge_parameters,
        (_, transition_count, decay) -> smoothing_weights(transition_count, decay),
        warm_up_data,
        training_data,
        testing_data,
        parameter_tuning_window,
    )

    dlba_result = nothing
    if run_dlba_ridge
        dlba_cache = Dict{Tuple{Float64, Int, Float64}, Vector{Float64}}()
        dlba_result = evaluate_ridge_family(
            "DLBA W2 ridge",
            dlba_ridge_parameters,
            (_, transition_count, rho_over_epsilon) ->
                cached_dlba_wasserstein_weights!(dlba_cache, dlba_wasserstein_order, transition_count, rho_over_epsilon),
            warm_up_data,
            training_data,
            testing_data,
            parameter_tuning_window,
        )
    end

    wpf_result = nothing
    if run_wpf_ridge
        wpf_parameters = [collect(LinRange(10, 100, 10)); collect(LinRange(200, 1000, 10)); collect(LinRange(2000, 10000, 10)); [Inf]]
        wpf_ridge_parameters = paired_parameter_grid(wpf_parameters, ridge_parameters)
        wpf_cache = Dict{Tuple{Int, Float64}, Vector{Float64}}()
        wpf_result = evaluate_ridge_family(
            "WPF L1 ridge",
            wpf_ridge_parameters,
            (samples, _, lambda) -> cached_wpf_transition_weights!(wpf_cache, samples, lambda),
            warm_up_data,
            training_data,
            testing_data,
            parameter_tuning_window,
        )
    end

    println()
    println("method,average_testing_cost,se_realised_cost,diff_from_plain_saa_percent,se_diff_percent,final_parameter,rolling_selected_parameters")
    print_summary_row("plain_saa", plain_saa_costs, plain_saa_costs, plain_saa_average_cost, "uniform", ["uniform"])
    print_summary_row("saa_ridge", saa_costs, plain_saa_costs, plain_saa_average_cost, saa_parameter, saa_selected_parameters)
    print_summary_row("windowing_ridge", windowing_costs, plain_saa_costs, plain_saa_average_cost, windowing_parameter, windowing_selected_parameters)
    print_summary_row("smoothing_ridge", smoothing_costs, plain_saa_costs, plain_saa_average_cost, smoothing_parameter, smoothing_selected_parameters)
    if run_dlba_ridge
        dlba_costs, dlba_parameter, dlba_selected_parameters = dlba_result
        print_summary_row("dlba_w2_ridge", dlba_costs, plain_saa_costs, plain_saa_average_cost, dlba_parameter, dlba_selected_parameters)
    end
    if run_wpf_ridge
        wpf_costs, wpf_parameter, wpf_selected_parameters = wpf_result
        print_summary_row("wpf_l1_ridge", wpf_costs, plain_saa_costs, plain_saa_average_cost, wpf_parameter, wpf_selected_parameters)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
