using LinearAlgebra
using Printf
using Statistics

using JuMP
using Ipopt

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

sem(x) = std(x) / sqrt(length(x))
portfolio_return(portfolio, realised_return) = dot(portfolio, realised_return)
LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

function empirical_cvar(costs)
    return minimum(tau + mean(max.(costs .- tau, 0)) / α for tau in costs)
end

function risk_adjusted_cost(costs)
    return ρ * mean(costs) + (1 - ρ) * empirical_cvar(costs)
end

function solve_risk_averse_portfolio(sample_returns, sample_weights)
    N = length(sample_returns)
    m = length(sample_returns[1])

    model = Model(portfolio_optimizer)

    @variables(model, begin
        x[i=1:m] >= 0
        τ
        z[i=1:N] >= 0
    end)

    @constraint(model, sum(x) == 1)
    @objective(model, Min,
        -ρ * sum(sample_weights[i] * dot(x, sample_returns[i]) for i in 1:N) +
        (1 - ρ) * τ +
        (1 - ρ) * sum(sample_weights[i] * (1 / α) * z[i] for i in 1:N)
    )

    for i in 1:N
        @constraint(model, z[i] >= -dot(x, sample_returns[i]) - τ)
    end

    optimize!(model)
    return value.(x)
end

function solve_wasserstein_dro_portfolio(sample_returns, radius)
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
        -ρ * sum(dot(x, sample_returns[i]) for i in 1:N) / N +
        (1 - ρ) * τ +
        (1 - ρ) * sum((1 / N) * (1 / α) * z[i] for i in 1:N) +
        radius * robust_lipschitz_coefficient * u
    )

    optimize!(model)
    return value.(x)
end

function smoothing_weights(observations, α_decay, d)
    T = length(observations)

    if α_decay == 0
        weights = zeros(T)
        weights .= 1 / T
        return weights
    end

    weights = [α_decay * (1 - α_decay)^(t - 1) for t in T:-1:1]
    return weights / sum(weights)
end

function windowing_weights(observations, s, d)
    T = length(observations)
    weights = zeros(T)

    if s >= T
        weights .= 1
    else
        for t in T:-1:T-(s-1)
            weights[t] = 1
        end
    end

    return weights / sum(weights)
end

function evaluate_fixed_mix(testing_data)
    m = length(testing_data[1])
    portfolio = fill(1 / m, m)
    return [-portfolio_return(portfolio, realised_return) for realised_return in testing_data]
end

function evaluate_portfolio_strategy(
    parameters,
    portfolio_builder,
    warm_up_data,
    training_data,
    testing_data;
    parameter_tuning_window::Int,
)
    training_T = length(training_data)
    testing_T = length(testing_data)

    parameter_costs_in_training_stages = zeros((training_T, length(parameters)))
    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        for i in eachindex(parameters)
            x = portfolio_builder(samples, parameters[i])
            parameter_costs_in_training_stages[t, i] = -portfolio_return(x, training_data[t])
        end
    end

    parameter_costs_in_testing_stages = zeros((testing_T, length(parameters)))
    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        for i in eachindex(parameters)
            x = portfolio_builder(samples, parameters[i])
            parameter_costs_in_testing_stages[t, i] = -portfolio_return(x, testing_data[t])
        end
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]
    realised_costs = Float64[]
    selected_parameters = []

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        average_parameter_costs =
            ρ * vec(mean(parameter_costs[previous_start:previous_end, :], dims = 1)) +
            (1 - ρ) * [empirical_cvar(parameter_costs[previous_start:previous_end, i]) for i in eachindex(parameters)]
        parameter_index = argmin(average_parameter_costs)
        push!(selected_parameters, parameters[parameter_index])
        push!(realised_costs, parameter_costs_in_testing_stages[t, parameter_index])
    end

    final_parameter_costs =
        ρ * vec(mean(parameter_costs[end-(parameter_tuning_window-1):end, :], dims = 1)) +
        (1 - ρ) * [empirical_cvar(parameter_costs[end-(parameter_tuning_window-1):end, i]) for i in eachindex(parameters)]

    return realised_costs, parameters[argmin(final_parameter_costs)], selected_parameters
end

function evaluate_weighted_method(
    parameters,
    weights,
    d,
    warm_up_data,
    training_data,
    testing_data;
    parameter_tuning_window::Int,
)
    training_T = length(training_data)
    testing_T = length(testing_data)

    parameter_costs_in_training_stages = zeros((training_T, length(parameters)))
    for t in training_T:-1:1
        samples = [warm_up_data; training_data[1:t-1]]
        for i in eachindex(parameters)
            sample_weights = weights(samples, parameters[i], d)
            x = solve_risk_averse_portfolio(samples, sample_weights)
            parameter_costs_in_training_stages[t, i] = -portfolio_return(x, training_data[t])
        end
    end

    parameter_costs_in_testing_stages = zeros((testing_T, length(parameters)))
    for t in testing_T:-1:1
        samples = [warm_up_data; training_data; testing_data[1:t-1]]
        for i in eachindex(parameters)
            sample_weights = weights(samples, parameters[i], d)
            x = solve_risk_averse_portfolio(samples, sample_weights)
            parameter_costs_in_testing_stages[t, i] = -portfolio_return(x, testing_data[t])
        end
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]
    realised_costs = Float64[]
    selected_parameters = []

    for t in 1:testing_T
        previous_start = training_T + (t - 1) - (parameter_tuning_window - 1)
        previous_end = training_T + (t - 1)
        average_parameter_costs =
            ρ * vec(mean(parameter_costs[previous_start:previous_end, :], dims = 1)) +
            (1 - ρ) * [empirical_cvar(parameter_costs[previous_start:previous_end, i]) for i in eachindex(parameters)]
        parameter_index = argmin(average_parameter_costs)
        push!(selected_parameters, parameters[parameter_index])
        push!(realised_costs, parameter_costs_in_testing_stages[t, parameter_index])
    end

    final_parameter_costs =
        ρ * vec(mean(parameter_costs[end-(parameter_tuning_window-1):end, :], dims = 1)) +
        (1 - ρ) * [empirical_cvar(parameter_costs[end-(parameter_tuning_window-1):end, i]) for i in eachindex(parameters)]

    return realised_costs, parameters[argmin(final_parameter_costs)], selected_parameters
end

function requested_methods(args)
    for arg in args
        if startswith(arg, "--methods=")
            return split(replace(arg, "--methods=" => ""), ",")
        end
    end
    return ["fixed_mix", "wasserstein_dro", "saa", "windowing", "smoothing"]
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
    methods = Set(requested_methods(args))
    allowed_methods = Set(["fixed_mix", "wasserstein_dro", "saa", "windowing", "smoothing"])
    all(method -> method in allowed_methods, methods) || error(
        "Unknown method requested. Use --methods=fixed_mix,wasserstein_dro,saa,windowing,smoothing",
    )

    data = load_stock_returns("stock-returns.csv")

    training_testing_split = ceil(Int, 0.7 * length(data))
    warm_up_period = ceil(Int, 0.7 * training_testing_split) - 1
    warm_up_data = data[1:warm_up_period]
    training_data = data[warm_up_period+1:training_testing_split]
    testing_data = data[training_testing_split+1:end]
    parameter_tuning_window = 2 * 12

    println("Loaded $(length(data)) monthly stock-return observations")
    println("Testing observations: $(length(testing_data))")
    println("Methods requested: " * join(sort(collect(methods)), ", "))

    d_zero(a, b) = 0

    saa_costs, saa_parameter, saa_selected_parameters = evaluate_weighted_method(
        [length(data)],
        windowing_weights,
        d_zero,
        warm_up_data,
        training_data,
        testing_data;
        parameter_tuning_window,
    )
    saa_risk = risk_adjusted_cost(saa_costs)

    println()
    println("method,risk_adjusted_cost,se_realised_cost,diff_from_saa_percent,se_diff_percent,final_parameter,rolling_selected_parameters")
    print_summary_row("saa", saa_costs, saa_costs, saa_risk, saa_parameter, saa_selected_parameters)

    if "fixed_mix" in methods
        fixed_mix_costs = evaluate_fixed_mix(testing_data)
        print_summary_row("fixed_mix", fixed_mix_costs, saa_costs, saa_risk, "equal_weight", ["equal_weight"])
    end

    if "wasserstein_dro" in methods
        wasserstein_dro_radius_parameters = [0; LogRange(1e-5, 1e-1, 29)]
        wasserstein_dro_costs, wasserstein_dro_parameter, wasserstein_dro_selected_parameters =
            evaluate_portfolio_strategy(
                wasserstein_dro_radius_parameters,
                solve_wasserstein_dro_portfolio,
                warm_up_data,
                training_data,
                testing_data;
                parameter_tuning_window,
            )
        print_summary_row("wasserstein_dro_l1", wasserstein_dro_costs, saa_costs, saa_risk, wasserstein_dro_parameter, wasserstein_dro_selected_parameters)
    end

    if "windowing" in methods
        windowing_parameters = unique(ceil.(Int, LogRange(1, length(data), 30)))
        windowing_costs, windowing_parameter, windowing_selected_parameters = evaluate_weighted_method(
            windowing_parameters,
            windowing_weights,
            d_zero,
            warm_up_data,
            training_data,
            testing_data;
            parameter_tuning_window,
        )
        print_summary_row("windowing", windowing_costs, saa_costs, saa_risk, windowing_parameter, windowing_selected_parameters)
    end

    if "smoothing" in methods
        smoothing_parameters = [0; LogRange(1e-4, 1e0, 30)]
        smoothing_costs, smoothing_parameter, smoothing_selected_parameters = evaluate_weighted_method(
            smoothing_parameters,
            smoothing_weights,
            d_zero,
            warm_up_data,
            training_data,
            testing_data;
            parameter_tuning_window,
        )
        print_summary_row("smoothing", smoothing_costs, saa_costs, saa_risk, smoothing_parameter, smoothing_selected_parameters)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
