using JuMP, Ipopt
import MathOptInterface as MOI

function _DLBA_W2_acceptable_termination_status(status)
    return status in (
        MOI.OPTIMAL,
        MOI.LOCALLY_SOLVED,
        MOI.ALMOST_OPTIMAL,
        MOI.ALMOST_LOCALLY_SOLVED,
    )
end

function solve_DLBA_W2_weights(T, rho_over_epsilon; epsilon = 10.0, max_iter = 500)
    if rho_over_epsilon == 0
        weights = zeros(T)
        weights .= 1 / T
        return weights
    end

    if rho_over_epsilon >= 1
        weights = zeros(T)
        weights[end] = 1
        return weights
    end

    if rho_over_epsilon < 0
        error("rho_over_epsilon must be nonnegative")
    end

    wasserstein_order = 2.0
    rho = rho_over_epsilon * epsilon
    ages = [(T - t + 1)^wasserstein_order for t in 1:T]

    model = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "sb" => "yes",
        "max_iter" => max_iter,
    ))

    @variables(model, begin
        1 >= weights[1:T] >= 0
    end)

    @constraint(model, sum(weights) == 1)
    @constraint(model, sum(weights[t] * ages[t] for t in 1:T) * rho^wasserstein_order <= epsilon^wasserstein_order)
    @NLobjective(model, Max,
        (1 / sum(weights[t]^2 for t in 1:T)) *
            (epsilon - sum(weights[t] * ages[t] for t in 1:T)^(1 / wasserstein_order) * rho)^(2 * wasserstein_order)
    )

    for t in 1:T
        set_start_value(weights[t], 1 / T)
    end

    optimize!(model)

    status = termination_status(model)
    if !_DLBA_W2_acceptable_termination_status(status)
        error("Ipopt failed to solve DLBA W2 weight model: termination_status=$status")
    end

    weights = max.(value.(weights), 0)
    weights = weights / sum(weights)

    return weights
end

function DLBA_W2_weights(observations, rho_over_epsilon, d)
    return solve_DLBA_W2_weights(length(observations), rho_over_epsilon)
end
