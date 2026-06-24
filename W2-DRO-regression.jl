using JuMP, LinearAlgebra
import MathOptInterface as MOI

"""
    fit_W2_DRO_weighted_AR1_model(time_series, weights, radius; optimizer)

Fit the robust AR(1) model

    y[t+1] = mu + A * y[t]

using the conservative W2-DRO square-root least-squares objective centered at
the weighted empirical transition distribution. The conic objective minimized is

    sqrt(sum_i weights[i] * ||y[i+1] - mu - A * y[i]||_2^2) +
        radius * ||[-A I]||_F

using the full transition-pair residual map
`(y_now, y_next) -> y_next - mu - A * y_now`.

For scalar responses this is the exact W2-DRO reformulation, up to the monotone
outer square. For vector responses it is a conservative Frobenius-norm upper
bound for the exact joint vector W2-DRO objective.
"""
function fit_W2_DRO_weighted_AR1_model(
    time_series,
    weights,
    radius;
    optimizer,
    return_diagnostics = false,
)
    isfinite(radius) || error("W2-DRO radius must be finite")
    radius >= 0 || error("W2-DRO radius must be nonnegative")

    n = length(time_series) - 1
    n >= 1 || error("At least two observations are required to fit an AR(1) model")
    length(weights) == n || error("Expected one weight per transition")

    total_weight = sum(weights)
    total_weight > 0 || error("Weights must have positive total mass")
    center_weights = Float64.(weights ./ total_weight)
    all(isfinite, center_weights) || error("Weights must be finite")
    all(center_weights .>= 0) || error("Weights must be nonnegative")

    m = length(time_series[1])
    all(length(observation) == m for observation in time_series) ||
        error("All observations must have the same dimension")

    inputs = time_series[1:end-1]
    outputs = time_series[2:end]

    model = Model(optimizer)

    @variables(model, begin
        mu[1:m]
        A[1:m, 1:m]
        empirical_error >= 0
        coefficient_norm >= 0
    end)

    @expression(
        model,
        residual_terms[i = 1:n, k = 1:m],
        sqrt(center_weights[i]) *
            (outputs[i][k] - mu[k] - sum(A[k, j] * inputs[i][j] for j in 1:m))
    )

    @constraint(model, vcat([empirical_error], vec(residual_terms)) in SecondOrderCone())

    @expression(model, coefficient_terms[k = 1:m, j = 1:m], A[k, j])
    @expression(model, response_terms[k = 1:m], 1.0)
    @constraint(
        model,
        vcat([coefficient_norm], vec(coefficient_terms), response_terms) in SecondOrderCone()
    )

    @objective(model, Min, empirical_error + radius * coefficient_norm)

    optimize!(model)

    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED))
        error("Failed to solve conservative W2-DRO AR(1) model: termination_status=$status")
    end

    diagnostics = (
        objective = objective_value(model),
        robust_risk_upper_bound = objective_value(model)^2,
        empirical_error = value(empirical_error),
        coefficient_norm = value(coefficient_norm),
        termination_status = status,
    )

    return return_diagnostics ? (value.(mu), value.(A), diagnostics) : (value.(mu), value.(A))
end
