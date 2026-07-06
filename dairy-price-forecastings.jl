using JuMP, LinearAlgebra, Statistics
using COPT

forecast_optimizer = optimizer_with_attributes(COPT.ConeOptimizer, "Logging" => 0, "LogToConsole" => 0, "BarIterLimit" => 50)

function fit_weighted_AR1_model(time_series, weights)
    m = length(time_series[1])

    lagged_time_series = time_series[2:end]
    time_series = time_series[1:end-1]

    X = hcat(ones(length(time_series)), stack(time_series)') # Leading column of ones for the additive term.
    weights = Diagonal(weights)
    least_squares_solution = (X'*weights*X) \ (X'*weights*stack(lagged_time_series)')

    μ = least_squares_solution[1, :] # First row is the intercept vector.
    A = least_squares_solution[2:end, :]' # Due to transposing these are ordered differently.
    return μ, A
end

function fit_W2_DRO_weighted_AR1_model(time_series, weights, radius)
    n = length(time_series) - 1
    m = length(time_series[1])
    weights = Float64.(weights ./ sum(weights))
    inputs = time_series[1:end-1]
    outputs = time_series[2:end]

    model = Model(forecast_optimizer)

    @variables(model, begin
                            μ[1:m]
                            A[1:m, 1:m]
                            empirical_error >= 0
                            coefficient_norm >= 0
                      end)

    @expression(model, residual_terms[i = 1:n, k = 1:m],
        sqrt(weights[i])*(outputs[i][k] - μ[k] - sum(A[k,j]*inputs[i][j] for j in 1:m)))

    @constraint(model, vcat([empirical_error], vec(residual_terms)) in SecondOrderCone())

    @expression(model, coefficient_terms[k = 1:m, j = 1:m], A[k,j])
    @expression(model, response_terms[k = 1:m], 1.0)
    @constraint(model, vcat([coefficient_norm], vec(coefficient_terms), response_terms) in SecondOrderCone())

    @objective(model, Min, empirical_error + radius*coefficient_norm)

    optimize!(model)

    return value.(μ), value.(A)
end

function median_pairwise_distance(samples)
    distances = [
        norm(samples[i] - samples[j], 2)
        for i in 1:length(samples)-1 for j in i+1:length(samples)
    ]

    scale = median(distances)
    return scale > 0 ? scale : 1.0
end

function kernel_analog_forecast(samples, h_multiplier)
    inputs = samples[1:end-1]
    outputs = [samples[i+1] - samples[i] for i in 1:length(samples)-1]
    bandwidth = h_multiplier * median_pairwise_distance(inputs)

    scaled_distances = [-norm(samples[end] - input, 2)^2 / (2*bandwidth^2) for input in inputs]
    kernel_weights = exp.(scaled_distances .- maximum(scaled_distances))
    kernel_weights ./= sum(kernel_weights)

    predicted_change = zeros(length(samples[1]))
    for i in eachindex(outputs)
        predicted_change .+= kernel_weights[i] .* outputs[i]
    end

    return samples[end] + predicted_change
end
