using Random, Distributions, Statistics, StatsBase
using LinearAlgebra

include("weights.jl")

const Cu = 4.0
const Co = 1.0

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
sem(x) = std(x) / sqrt(length(x))

function paired_parameter_grid(first_parameters, second_parameters)
    length(first_parameters) == length(second_parameters) ||
        error("Parameter grids must have the same length")

    n = length(first_parameters)
    stride = findfirst(stride -> gcd(stride, n) == 1, max(1, n ÷ 2):n-1)
    stride === nothing && (stride = 1)
    second_indices = [mod((i - 1) * stride, n) + 1 for i in 1:n]
    return [(first_parameters[i], second_parameters[second_indices[i]]) for i in 1:n]
end

function sample_mode_value(means, standard_deviation, dimensions)
    mode = rand(1:length(means))
    return means[mode] + standard_deviation * randn(dimensions)
end

function weighted_newsvendor_order(demands, weights, dimensions)
    q = Cu / (Cu + Co)
    return [
        quantile([demands[t][i] for t in eachindex(demands)], Weights(weights), q)
        for i in 1:dimensions
    ]
end

function robust_dual_value(demands, weights, order, lambda, radius)
    value = lambda * radius^2
    for t in eachindex(demands)
        sample_loss = 0.0
        for i in eachindex(order)
            under_piece = Cu * (demands[t][i] - order[i]) + Cu^2 / (4 * lambda)
            over_piece = Co * (order[i] - demands[t][i]) + Co^2 / (4 * lambda)
            sample_loss += max(under_piece, over_piece)
        end
        value += weights[t] * sample_loss
    end
    return value
end

function golden_section_minimize(f, lower, upper; iterations = 80)
    inverse_phi = (sqrt(5) - 1) / 2
    inverse_phi_squared = (3 - sqrt(5)) / 2

    a = lower
    b = upper
    h = b - a
    c = a + inverse_phi_squared * h
    d = a + inverse_phi * h
    yc = f(c)
    yd = f(d)

    for _ in 1:iterations
        if yc < yd
            b = d
            d = c
            yd = yc
            h = inverse_phi * h
            c = a + inverse_phi_squared * h
            yc = f(c)
        else
            a = c
            c = d
            yc = yd
            h = inverse_phi * h
            d = a + inverse_phi * h
            yd = f(d)
        end
    end

    return (a + b) / 2
end

function w2_wdro_newsvendor_order(demands, weights, radius, dimensions)
    radius == 0 && return weighted_newsvendor_order(demands, weights, dimensions)

    q = Cu / (Cu + Co)

    function order_for_lambda(lambda)
        shift = (Cu - Co) / (4 * lambda)
        return [
            quantile([demands[t][i] for t in eachindex(demands)], Weights(weights), q) + shift
            for i in 1:dimensions
        ]
    end

    function objective_on_log_lambda(log_lambda)
        lambda = exp(log_lambda)
        order = order_for_lambda(lambda)
        return robust_dual_value(demands, weights, order, lambda, radius)
    end

    log_lambda = golden_section_minimize(objective_on_log_lambda, -10.0, 10.0)
    return order_for_lambda(exp(log_lambda))
end

function normal_newsvendor_loss(order, mean, standard_deviation)
    total = 0.0
    standard_normal = Normal()

    for i in eachindex(order)
        z = (order[i] - mean[i]) / standard_deviation
        pdf_z = pdf(standard_normal, z)
        cdf_z = cdf(standard_normal, z)

        expected_underage = standard_deviation * pdf_z + (mean[i] - order[i]) * (1 - cdf_z)
        expected_overage = standard_deviation * pdf_z + (order[i] - mean[i]) * cdf_z
        total += Cu * expected_underage + Co * expected_overage
    end

    return total
end

function mixture_newsvendor_loss(order, means, standard_deviation)
    return mean(normal_newsvendor_loss(order, mean, standard_deviation) for mean in means)
end

function main()
    Random.seed!(42)

    dimensions = 2
    modes = 3
    repetitions = 1000
    history_length = 100
    parameter_budget = 31

    initial_means = [i * 100 for i in 1:modes]
    observation_standard_deviation = 20.0
    shift_standard_deviation = 15.0
    final_standard_deviation = sqrt(observation_standard_deviation^2 + shift_standard_deviation^2)

    dlba_parameters = [0.0; collect(LogRange(1e-4, 1, parameter_budget - 1))]
    radius_parameters = [0.0; collect(LogRange(1e-2, 1e2, parameter_budget - 1))]
    parameter_pairs = paired_parameter_grid(dlba_parameters, radius_parameters)

    dlba_weight_cache = Dict{Float64, Vector{Float64}}()
    function center_weights(rho_over_epsilon)
        return get!(dlba_weight_cache, rho_over_epsilon) do
            DLBA_W2_weights(1:history_length, rho_over_epsilon, 0)
        end
    end

    saa_weights = fill(1 / history_length, history_length)
    saa_costs = zeros(repetitions)
    wdro_parameter_costs = zeros(repetitions, length(parameter_pairs))

    println("Candidate pairs: $(length(parameter_pairs))")
    println("DLBA W2 rho/epsilon grid: 0 plus LogRange(1e-4, 1, 30)")
    println("W2 WDRO radius grid: 0 plus LogRange(1e-2, 1e2, 30)")

    for repetition in 1:repetitions
        means = [fill(Float64(initial_means[i]), dimensions) for i in 1:modes]
        demands = [zeros(dimensions) for _ in 1:history_length]

        for t in 1:history_length
            demands[t] = sample_mode_value(means, observation_standard_deviation, dimensions)
            for i in eachindex(means)
                means[i] += shift_standard_deviation * randn(dimensions)
            end
        end

        saa_order = weighted_newsvendor_order(demands, saa_weights, dimensions)
        saa_costs[repetition] = mixture_newsvendor_loss(saa_order, means, final_standard_deviation)

        for (i, (rho_over_epsilon, radius)) in enumerate(parameter_pairs)
            weights = center_weights(rho_over_epsilon)
            order = w2_wdro_newsvendor_order(demands, weights, radius, dimensions)
            wdro_parameter_costs[repetition, i] = mixture_newsvendor_loss(order, means, final_standard_deviation)
        end

        repetition % 100 == 0 && println("Completed repetition $repetition / $repetitions")
    end

    selected_index = argmin(vec(mean(wdro_parameter_costs, dims = 1)))
    wdro_costs = wdro_parameter_costs[:, selected_index]
    selected_parameter = parameter_pairs[selected_index]

    saa_average_cost = mean(saa_costs)
    wdro_average_cost = mean(wdro_costs)
    difference = (wdro_average_cost - saa_average_cost) / saa_average_cost * 100
    difference_se = sem(wdro_costs - saa_costs) / saa_average_cost * 100

    println("method,average_cost,se_cost,diff_from_saa_percent,se_diff_percent,parameter")
    println(join([
        "saa_reference",
        string(round(saa_average_cost, digits = 6)),
        string(round(sem(saa_costs), digits = 6)),
        "0.0",
        "0.0",
        "uniform",
    ], ","))
    println(join([
        "dlba_w2_centered_w2_wdro",
        string(round(wdro_average_cost, digits = 6)),
        string(round(sem(wdro_costs), digits = 6)),
        string(round(difference, digits = 4)),
        string(round(difference_se, digits = 4)),
        string(selected_parameter),
    ], ","))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
