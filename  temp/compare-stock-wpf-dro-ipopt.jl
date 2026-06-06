using LinearAlgebra
using Printf
using Statistics

using JuMP
using Ipopt

include("ipopt-wpf.jl")

const GUROBI_AVAILABLE = try
    @eval import Gurobi
    true
catch
    false
end

const PRODUCTS = ["BA", "BRKB", "GS", "JNJ", "JPM", "KO", "MCD", "PFE", "WMT", "XOM"]
const ρ = 0.1
const α = 0.05

const portfolio_optimizer = if GUROBI_AVAILABLE
    const gurobi_env = Gurobi.Env()
    optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0)
else
    optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "sb" => "yes",
    )
end

struct Options
    test_size::Int
    parameter_tuning_window::Int
    lambdas::Vector{Float64}
    radii::Vector{Float64}
    ipopt_max_iter::Int
end

function load_stock_returns(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || error("No stock-return rows found in $path")

    returns = Vector{Float64}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ",")
        length(fields) == length(PRODUCTS) + 1 || error("Malformed stock-return row: $line")
        push!(returns, parse.(Float64, fields[2:end]))
    end
    return returns
end

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
sem(x) = std(x) / sqrt(length(x))
portfolio_return(portfolio, realised_return) = dot(portfolio, realised_return)
wpf_distance(a, b) = norm(a - b, 1)
coarse_wpf_lambdas() = [0.0, 1.0, 10.0, 100.0, 1000.0, Inf]
coarse_dro_radii() = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
smoothing_decays() = [0.0; collect(LogRange(1e-4, 1e0, 30))]
dlba_rho_over_epsilon_values() = [0.0; collect(LogRange(1e-4, 1e0, 30))]
original_wpf_lambdas() = [0.0; collect(LinRange(1, 10, 10)); collect(LinRange(20, 100, 9)); collect(LinRange(200, 1000, 9)); Inf]
default_dro_radii() = [0.0; collect(LogRange(1e-5, 1e-1, 9))]
refined_coarse_wpf_lambdas() = [0.0, 0.5, 1.0, sqrt(10.0), 10.0, sqrt(1000.0), 100.0, sqrt(100000.0), 1000.0, Inf]
refined_coarse_dro_radii() = [0.0, 5e-5, 1e-4, sqrt(1e-7), 1e-3, sqrt(1e-5), 1e-2, sqrt(1e-3), 1e-1]

function refine_parameter_grid_once(values)
    refined = Float64[]
    for i in 1:(length(values) - 1)
        current = values[i]
        next_value = values[i + 1]
        push!(refined, current)
        if isfinite(current) && isfinite(next_value)
            push!(refined, current == 0 ? next_value / 2 : sqrt(current * next_value))
        end
    end
    push!(refined, values[end])
    return refined
end

double_refined_coarse_wpf_lambdas() = refine_parameter_grid_once(refined_coarse_wpf_lambdas())
double_refined_coarse_dro_radii() = refine_parameter_grid_once(refined_coarse_dro_radii())

function parse_float_or_inf(value::AbstractString)
    lowercase(strip(value)) == "inf" && return Inf
    return parse(Float64, value)
end

function parse_float_list(value::AbstractString)
    return parse_float_or_inf.(split(value, ","))
end

function parse_args(args)
    opts = Dict{String, String}(
        "--test-size" => "0",
        "--parameter-tuning-window" => "24",
        "--lambdas" => join(coarse_wpf_lambdas(), ","),
        "--radii" => join(coarse_dro_radii(), ","),
        "--ipopt-max-iter" => "500",
    )

    for arg in args
        if startswith(arg, "--")
            parts = split(arg, "=", limit = 2)
            length(parts) == 2 || error("Arguments must use --key=value form: $arg")
            haskey(opts, parts[1]) || error("Unknown argument: $(parts[1])")
            opts[parts[1]] = parts[2]
        else
            error("Unknown positional argument: $arg")
        end
    end

    return Options(
        parse(Int, opts["--test-size"]),
        parse(Int, opts["--parameter-tuning-window"]),
        parse_float_list(opts["--lambdas"]),
        parse_float_list(opts["--radii"]),
        parse(Int, opts["--ipopt-max-iter"]),
    )
end

function empirical_cvar(costs)
    return minimum(tau + mean(max.(costs .- tau, 0)) / α for tau in costs)
end

function risk_adjusted_cost(costs)
    return ρ * mean(costs) + (1 - ρ) * empirical_cvar(costs)
end

function solve_weighted_mean_cvar_portfolio(sample_returns, sample_weights)
    N = length(sample_returns)
    m = length(sample_returns[1])

    model = Model(portfolio_optimizer)

    @variables(model, begin
        x[i=1:m] >= 0
        τ
        z[i=1:N] >= 0
    end)

    @constraint(model, sum(x) == 1)
    for i in 1:N
        @constraint(model, z[i] >= -dot(x, sample_returns[i]) - τ)
    end

    @objective(model, Min,
        -ρ * sum(sample_weights[i] * dot(x, sample_returns[i]) for i in 1:N) +
        (1 - ρ) * τ +
        (1 - ρ) * sum(sample_weights[i] * (1 / α) * z[i] for i in 1:N)
    )

    optimize!(model)
    return value.(x)
end

function solve_weighted_wasserstein_dro_portfolio(sample_returns, center_weights, radius)
    N = length(sample_returns)
    m = length(sample_returns[1])
    robust_lipschitz_coefficient = ρ + (1 - ρ) / α

    model = Model(portfolio_optimizer)

    @variables(model, begin
        x[i=1:m] >= 0
        τ
        z[i=1:N] >= 0
        u >= 0
    end)

    @constraint(model, sum(x) == 1)
    @constraint(model, [i=1:m], x[i] <= u)
    for i in 1:N
        @constraint(model, z[i] >= -dot(x, sample_returns[i]) - τ)
    end

    @objective(model, Min,
        -ρ * sum(center_weights[i] * dot(x, sample_returns[i]) for i in 1:N) +
        (1 - ρ) * τ +
        (1 - ρ) * sum(center_weights[i] * (1 / α) * z[i] for i in 1:N) +
        radius * robust_lipschitz_coefficient * u
    )

    optimize!(model)
    return value.(x)
end

function wpf_center_weights(samples, lambda, options::Options)
    return ipopt_wpf_weights(
        samples,
        lambda,
        wpf_distance;
        max_iter = options.ipopt_max_iter,
    )
end

function cached_wpf_center_weights!(weight_cache, stage::Symbol, t::Int, samples, lambda::Float64, options::Options)
    key = (stage, t, lambda)
    return get!(weight_cache, key) do
        wpf_center_weights(samples, lambda, options)
    end
end

function smoothing_center_weights(samples, smoothing_decay)
    N = length(samples)

    if smoothing_decay == 0
        return fill(1 / N, N)
    end

    weights = [smoothing_decay * (1 - smoothing_decay)^(lag - 1) for lag in N:-1:1]
    return weights ./ sum(weights)
end

function windowing_center_weights(samples, window_length::Int)
    N = length(samples)
    active_window_length = min(window_length, N)
    weights = zeros(N)
    weights[(N - active_window_length + 1):N] .= 1 / active_window_length
    return weights
end

function dlba_wasserstein_weights(wasserstein_order, T, rho_over_epsilon)
    if rho_over_epsilon == 0
        return fill(1 / T, T)
    elseif rho_over_epsilon >= 1
        weights = zeros(T)
        weights[end] = 1
        return weights
    elseif rho_over_epsilon < 0
        error("rho_over_epsilon must be nonnegative")
    end

    epsilon = 10.0
    rho = rho_over_epsilon * epsilon
    ages = [(T - t + 1)^wasserstein_order for t in 1:T]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "sb", "yes")

    @variable(model, 0 <= weights[1:T] <= 1)
    @constraint(model, sum(weights) == 1)
    @constraint(model, sum(weights[t] * ages[t] for t in 1:T) * rho^wasserstein_order <= epsilon^wasserstein_order)
    @NLobjective(
        model,
        Max,
        (1 / sum(weights[t]^2 for t in 1:T)) *
        (epsilon - sum(weights[t] * ages[t] for t in 1:T)^(1 / wasserstein_order) * rho)^(2 * wasserstein_order)
    )

    for t in 1:T
        set_start_value(weights[t], 1 / T)
    end

    optimize!(model)

    status = termination_status(model)
    if !(status in IPOPT_ACCEPTABLE_TERMINATION_STATUSES)
        error("Ipopt failed to solve DLBA weight model: termination_status=$status")
    end

    result = max.(value.(weights), 0.0)
    return result ./ sum(result)
end

function cached_dlba_center_weights!(weight_cache, wasserstein_order, T, rho_over_epsilon)
    key = (wasserstein_order, T, rho_over_epsilon)
    return get!(weight_cache, key) do
        dlba_wasserstein_weights(wasserstein_order, T, rho_over_epsilon)
    end
end

function evaluate_saa(warm_up_data, training_data, testing_data)
    realised_costs = Float64[]
    for t in eachindex(testing_data)
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        weights = fill(1 / length(samples), length(samples))
        x = solve_weighted_mean_cvar_portfolio(samples, weights)
        push!(realised_costs, -portfolio_return(x, testing_data[t]))
    end
    return realised_costs
end

function evaluate_saa_centered_dro(warm_up_data, training_data, testing_data, options::Options)
    training_T = length(training_data)
    testing_T = length(testing_data)
    parameters = options.radii

    parameter_costs_in_training_stages = zeros((training_T, length(parameters)))
    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        center_weights = fill(1 / length(samples), length(samples))
        for (i, radius) in enumerate(parameters)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_training_stages[t, i] = -portfolio_return(x, training_data[t])
        end
        println("SAA-centered DRO training stage $t / $training_T")
    end

    parameter_costs_in_testing_stages = zeros((testing_T, length(parameters)))
    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        center_weights = fill(1 / length(samples), length(samples))
        for (i, radius) in enumerate(parameters)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_testing_stages[t, i] = -portfolio_return(x, testing_data[t])
        end
        println("SAA-centered DRO testing stage $t / $testing_T")
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]
    realised_costs = Float64[]
    selected_parameters = Float64[]

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (options.parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        previous_start >= 1 || error("Not enough previous stages for the tuning window")

        average_parameter_costs =
            ρ * vec(mean(parameter_costs[previous_start:previous_end, :], dims = 1)) +
            (1 - ρ) * [empirical_cvar(parameter_costs[previous_start:previous_end, i]) for i in eachindex(parameters)]
        parameter_index = argmin(average_parameter_costs)
        push!(selected_parameters, parameters[parameter_index])
        push!(realised_costs, parameter_costs_in_testing_stages[t, parameter_index])
    end

    final_parameter_costs =
        ρ * vec(mean(parameter_costs[end-(options.parameter_tuning_window-1):end, :], dims = 1)) +
        (1 - ρ) * [empirical_cvar(parameter_costs[end-(options.parameter_tuning_window-1):end, i]) for i in eachindex(parameters)]

    return realised_costs, parameters[argmin(final_parameter_costs)], selected_parameters
end

function evaluate_windowing_centered_dro(warm_up_data, training_data, testing_data, window_lengths, options::Options)
    training_T = length(training_data)
    testing_T = length(testing_data)
    parameters = [(window_length, radius) for window_length in window_lengths for radius in options.radii]

    parameter_costs_in_training_stages = zeros((training_T, length(parameters)))
    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        for (i, (window_length, radius)) in enumerate(parameters)
            center_weights = windowing_center_weights(samples, window_length)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_training_stages[t, i] = -portfolio_return(x, training_data[t])
        end
        println("Windowing-centered DRO training stage $t / $training_T")
    end

    parameter_costs_in_testing_stages = zeros((testing_T, length(parameters)))
    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        for (i, (window_length, radius)) in enumerate(parameters)
            center_weights = windowing_center_weights(samples, window_length)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_testing_stages[t, i] = -portfolio_return(x, testing_data[t])
        end
        println("Windowing-centered DRO testing stage $t / $testing_T")
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]
    realised_costs = Float64[]
    selected_parameters = []

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (options.parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        previous_start >= 1 || error("Not enough previous stages for the tuning window")

        average_parameter_costs =
            ρ * vec(mean(parameter_costs[previous_start:previous_end, :], dims = 1)) +
            (1 - ρ) * [empirical_cvar(parameter_costs[previous_start:previous_end, i]) for i in eachindex(parameters)]
        parameter_index = argmin(average_parameter_costs)
        push!(selected_parameters, parameters[parameter_index])
        push!(realised_costs, parameter_costs_in_testing_stages[t, parameter_index])
    end

    final_parameter_costs =
        ρ * vec(mean(parameter_costs[end-(options.parameter_tuning_window-1):end, :], dims = 1)) +
        (1 - ρ) * [empirical_cvar(parameter_costs[end-(options.parameter_tuning_window-1):end, i]) for i in eachindex(parameters)]

    return realised_costs, parameters[argmin(final_parameter_costs)], selected_parameters
end

function evaluate_smoothing_centered_dro(warm_up_data, training_data, testing_data, smoothing_parameters, options::Options)
    training_T = length(training_data)
    testing_T = length(testing_data)
    parameters = [(smoothing_decay, radius) for smoothing_decay in smoothing_parameters for radius in options.radii]

    parameter_costs_in_training_stages = zeros((training_T, length(parameters)))
    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        for (i, (smoothing_decay, radius)) in enumerate(parameters)
            center_weights = smoothing_center_weights(samples, smoothing_decay)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_training_stages[t, i] = -portfolio_return(x, training_data[t])
        end
        println("Smoothing-centered DRO training stage $t / $training_T")
    end

    parameter_costs_in_testing_stages = zeros((testing_T, length(parameters)))
    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        for (i, (smoothing_decay, radius)) in enumerate(parameters)
            center_weights = smoothing_center_weights(samples, smoothing_decay)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_testing_stages[t, i] = -portfolio_return(x, testing_data[t])
        end
        println("Smoothing-centered DRO testing stage $t / $testing_T")
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]
    realised_costs = Float64[]
    selected_parameters = []

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (options.parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        previous_start >= 1 || error("Not enough previous stages for the tuning window")

        average_parameter_costs =
            ρ * vec(mean(parameter_costs[previous_start:previous_end, :], dims = 1)) +
            (1 - ρ) * [empirical_cvar(parameter_costs[previous_start:previous_end, i]) for i in eachindex(parameters)]
        parameter_index = argmin(average_parameter_costs)
        push!(selected_parameters, parameters[parameter_index])
        push!(realised_costs, parameter_costs_in_testing_stages[t, parameter_index])
    end

    final_parameter_costs =
        ρ * vec(mean(parameter_costs[end-(options.parameter_tuning_window-1):end, :], dims = 1)) +
        (1 - ρ) * [empirical_cvar(parameter_costs[end-(options.parameter_tuning_window-1):end, i]) for i in eachindex(parameters)]

    return realised_costs, parameters[argmin(final_parameter_costs)], selected_parameters
end

function evaluate_dlba_centered_dro(
    warm_up_data,
    training_data,
    testing_data,
    rho_over_epsilon_parameters,
    options::Options,
    dlba_weight_cache;
    wasserstein_order = 1.0,
)
    training_T = length(training_data)
    testing_T = length(testing_data)
    parameters = [(rho_over_epsilon, radius) for rho_over_epsilon in rho_over_epsilon_parameters for radius in options.radii]

    parameter_costs_in_training_stages = zeros((training_T, length(parameters)))
    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        for (i, (rho_over_epsilon, radius)) in enumerate(parameters)
            center_weights = cached_dlba_center_weights!(dlba_weight_cache, wasserstein_order, length(samples), rho_over_epsilon)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_training_stages[t, i] = -portfolio_return(x, training_data[t])
        end
        println("DLBA-centered DRO training stage $t / $training_T")
    end

    parameter_costs_in_testing_stages = zeros((testing_T, length(parameters)))
    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        for (i, (rho_over_epsilon, radius)) in enumerate(parameters)
            center_weights = cached_dlba_center_weights!(dlba_weight_cache, wasserstein_order, length(samples), rho_over_epsilon)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_testing_stages[t, i] = -portfolio_return(x, testing_data[t])
        end
        println("DLBA-centered DRO testing stage $t / $testing_T")
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]
    realised_costs = Float64[]
    selected_parameters = []

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (options.parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        previous_start >= 1 || error("Not enough previous stages for the tuning window")

        average_parameter_costs =
            ρ * vec(mean(parameter_costs[previous_start:previous_end, :], dims = 1)) +
            (1 - ρ) * [empirical_cvar(parameter_costs[previous_start:previous_end, i]) for i in eachindex(parameters)]
        parameter_index = argmin(average_parameter_costs)
        push!(selected_parameters, parameters[parameter_index])
        push!(realised_costs, parameter_costs_in_testing_stages[t, parameter_index])
    end

    final_parameter_costs =
        ρ * vec(mean(parameter_costs[end-(options.parameter_tuning_window-1):end, :], dims = 1)) +
        (1 - ρ) * [empirical_cvar(parameter_costs[end-(options.parameter_tuning_window-1):end, i]) for i in eachindex(parameters)]

    return realised_costs, parameters[argmin(final_parameter_costs)], selected_parameters
end

function evaluate_wpf_centered_dro(warm_up_data, training_data, testing_data, options::Options, weight_cache)
    training_T = length(training_data)
    testing_T = length(testing_data)
    parameters = [(lambda, radius) for lambda in options.lambdas for radius in options.radii]

    parameter_costs_in_training_stages = zeros((training_T, length(parameters)))
    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        for (i, (lambda, radius)) in enumerate(parameters)
            center_weights = cached_wpf_center_weights!(weight_cache, :train, t, samples, lambda, options)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_training_stages[t, i] = -portfolio_return(x, training_data[t])
        end
        println("WPF-centered DRO training stage $t / $training_T")
    end

    parameter_costs_in_testing_stages = zeros((testing_T, length(parameters)))
    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        for (i, (lambda, radius)) in enumerate(parameters)
            center_weights = cached_wpf_center_weights!(weight_cache, :test, t, samples, lambda, options)
            x = solve_weighted_wasserstein_dro_portfolio(samples, center_weights, radius)
            parameter_costs_in_testing_stages[t, i] = -portfolio_return(x, testing_data[t])
        end
        println("WPF-centered DRO testing stage $t / $testing_T")
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]
    realised_costs = Float64[]
    selected_parameters = []

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (options.parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        previous_start >= 1 || error("Not enough previous stages for the tuning window")

        average_parameter_costs =
            ρ * vec(mean(parameter_costs[previous_start:previous_end, :], dims = 1)) +
            (1 - ρ) * [empirical_cvar(parameter_costs[previous_start:previous_end, i]) for i in eachindex(parameters)]
        parameter_index = argmin(average_parameter_costs)
        push!(selected_parameters, parameters[parameter_index])
        push!(realised_costs, parameter_costs_in_testing_stages[t, parameter_index])
    end

    final_parameter_costs =
        ρ * vec(mean(parameter_costs[end-(options.parameter_tuning_window-1):end, :], dims = 1)) +
        (1 - ρ) * [empirical_cvar(parameter_costs[end-(options.parameter_tuning_window-1):end, i]) for i in eachindex(parameters)]

    return realised_costs, parameters[argmin(final_parameter_costs)], selected_parameters
end

function print_summary_row(method, costs, saa_costs, saa_risk, parameter, selected_parameters)
    method_risk = risk_adjusted_cost(costs)
    difference = (method_risk - saa_risk) / saa_risk * 100
    difference_se = sem(costs - saa_costs) / saa_risk * 100
    cost_se = sem(costs)

    println(join([
        method,
        @sprintf("%.10g", method_risk),
        @sprintf("%.10g", cost_se),
        @sprintf("%.4f", difference),
        @sprintf("%.4f", difference_se),
        string(parameter),
        join(unique(selected_parameters), ";"),
    ], ","))
end

function main(args)
    options = parse_args(args)
    data = load_stock_returns("stock-returns.csv")

    training_testing_split = ceil(Int, 0.7 * length(data))
    warm_up_period = ceil(Int, 0.7 * training_testing_split) - 1
    warm_up_data = data[1:warm_up_period]
    training_data = data[warm_up_period+1:training_testing_split]
    testing_data = data[training_testing_split+1:end]
    if options.test_size > 0
        testing_data = testing_data[1:min(options.test_size, length(testing_data))]
    end

    length(training_data) >= options.parameter_tuning_window || error("Training data shorter than tuning window")
    window_lengths = unique(ceil.(Int, LogRange(1, length(data), 30)))
    smoothing_parameters = smoothing_decays()
    dlba_parameters = dlba_rho_over_epsilon_values()
    dlba_wasserstein_order = 1.0

    println("Loaded $(length(data)) monthly stock-return observations")
    println("Testing observations: $(length(testing_data))")
    println("Portfolio DRO order: W1 with L1 ground metric")
    println("DLBA weighting order: W$(Int(dlba_wasserstein_order))")
    println("WPF lambdas: " * join(options.lambdas, ", "))
    println("DRO radii: " * join(options.radii, ", "))
    println("WPF-centered DRO candidate pairs: $(length(options.lambdas) * length(options.radii))")
    println("SAA-centered DRO candidate radii: $(length(options.radii))")
    println("Windowing-centered DRO candidate pairs: $(length(window_lengths) * length(options.radii))")
    println("Smoothing-centered DRO candidate pairs: $(length(smoothing_parameters) * length(options.radii))")
    println("DLBA-centered DRO candidate pairs: $(length(dlba_parameters) * length(options.radii))")

    saa_costs = evaluate_saa(warm_up_data, training_data, testing_data)
    saa_risk = risk_adjusted_cost(saa_costs)
    wpf_weight_cache = Dict{Tuple{Symbol, Int, Float64}, Vector{Float64}}()
    dlba_weight_cache = Dict{Tuple{Float64, Int, Float64}, Vector{Float64}}()
    saa_dro_costs, saa_dro_parameter, saa_dro_selected_parameters =
        evaluate_saa_centered_dro(warm_up_data, training_data, testing_data, options)
    windowing_dro_costs, windowing_dro_parameter, windowing_dro_selected_parameters =
        evaluate_windowing_centered_dro(warm_up_data, training_data, testing_data, window_lengths, options)
    smoothing_dro_costs, smoothing_dro_parameter, smoothing_dro_selected_parameters =
        evaluate_smoothing_centered_dro(warm_up_data, training_data, testing_data, smoothing_parameters, options)
    dlba_dro_costs, dlba_dro_parameter, dlba_dro_selected_parameters =
        evaluate_dlba_centered_dro(
            warm_up_data,
            training_data,
            testing_data,
            dlba_parameters,
            options,
            dlba_weight_cache;
            wasserstein_order = dlba_wasserstein_order,
        )
    wpf_dro_costs, wpf_dro_parameter, wpf_dro_selected_parameters =
        evaluate_wpf_centered_dro(warm_up_data, training_data, testing_data, options, wpf_weight_cache)

    println()
    println("method,risk_adjusted_cost,se_realised_cost,diff_from_saa_percent,se_diff_percent,final_parameter,rolling_selected_parameters")
    print_summary_row("saa", saa_costs, saa_costs, saa_risk, "uniform", ["uniform"])
    print_summary_row("saa_centered_dro_l1", saa_dro_costs, saa_costs, saa_risk, saa_dro_parameter, saa_dro_selected_parameters)
    print_summary_row("windowing_centered_dro_l1", windowing_dro_costs, saa_costs, saa_risk, windowing_dro_parameter, windowing_dro_selected_parameters)
    print_summary_row("smoothing_centered_dro_l1", smoothing_dro_costs, saa_costs, saa_risk, smoothing_dro_parameter, smoothing_dro_selected_parameters)
    print_summary_row("dlba_w1_centered_dro_l1", dlba_dro_costs, saa_costs, saa_risk, dlba_dro_parameter, dlba_dro_selected_parameters)
    print_summary_row("wpf_centered_dro_l1_ipopt", wpf_dro_costs, saa_costs, saa_risk, wpf_dro_parameter, wpf_dro_selected_parameters)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
