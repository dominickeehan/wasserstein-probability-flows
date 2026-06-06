using LinearAlgebra
using JuMP
using Ipopt
import MathOptInterface as MOI

const IPOPT_ACCEPTABLE_TERMINATION_STATUSES = Set([
    MOI.OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_OPTIMAL,
    MOI.ALMOST_LOCALLY_SOLVED,
])

"""
    ipopt_wpf_weights(observations, lambda, distance; kwargs...)

Estimate the terminal Wasserstein Probability Flow distribution using Ipopt.

`observations` must be ordered from oldest to newest. `distance(a, b)` defines
the ground metric between two historical observations, and `lambda` controls
the penalty on distributional movement. The returned vector has one normalized
weight per observation.
"""
function ipopt_wpf_weights(
    observations,
    lambda::Real,
    distance;
    min_probability::Float64 = 1e-8,
    print_level::Int = 0,
    max_iter::Int = 500,
    tol::Float64 = 1e-7,
)
    T = length(observations)
    T == 0 && error("ipopt_wpf_weights requires at least one observation")

    if lambda == Inf
        return fill(1 / T, T)
    elseif lambda == 0
        weights = zeros(T)
        weights[end] = 1.0
        return weights
    elseif lambda < 0
        error("lambda must be nonnegative")
    end

    distances = [distance(observations[i], observations[j]) for i in 1:T, j in 1:T]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", print_level)
    set_optimizer_attribute(model, "sb", "yes")
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", tol)

    @variable(model, 0 <= p[1:T, 1:2] <= 1)
    @variable(model, min_probability <= p_diag[1:T] <= 1)
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
        error("Ipopt failed to solve WPF model: termination_status=$status")
    end

    weights = [max(value(p[i, 2]), 0.0) for i in 1:T]
    total_weight = sum(weights)
    total_weight <= 0 && error("Ipopt returned zero terminal probability mass")

    return weights / total_weight
end
