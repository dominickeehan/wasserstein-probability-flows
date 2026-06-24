using Statistics, StatsBase

function _normalised_newsvendor_weights(weights)
    total_weight = sum(weights)
    total_weight > 0 || error("Weights must have positive total mass")

    return weights / total_weight
end

function _weighted_newsvendor_order(demands, weights, underage_cost, overage_cost)
    weights = _normalised_newsvendor_weights(weights)
    dimensions = length(demands[1])
    quantile_level = underage_cost / (underage_cost + overage_cost)

    return [
        quantile([demands[t][i] for t in eachindex(demands)], Weights(weights), quantile_level)
        for i in 1:dimensions
    ]
end

function _W2_DRO_newsvendor_dual_value(demands, weights, order, lambda, radius, underage_cost, overage_cost)
    value = lambda * radius^2

    for t in eachindex(demands)
        sample_value = 0.0
        for i in eachindex(order)
            underage_piece = underage_cost * (demands[t][i] - order[i]) + underage_cost^2 / (4 * lambda)
            overage_piece = overage_cost * (order[i] - demands[t][i]) + overage_cost^2 / (4 * lambda)
            sample_value += max(underage_piece, overage_piece)
        end

        value += weights[t] * sample_value
    end

    return value
end

function _golden_section_minimize(f, lower, upper; iterations = 80)
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

function W2_DRO_newsvendor_order(demands, weights, radius; underage_cost = 4.0, overage_cost = 1.0)
    radius >= 0 || error("W2 DRO radius must be nonnegative")

    weights = _normalised_newsvendor_weights(weights)
    radius == 0 && return _weighted_newsvendor_order(demands, weights, underage_cost, overage_cost)

    base_order = _weighted_newsvendor_order(demands, weights, underage_cost, overage_cost)

    function order_for_lambda(lambda)
        shift = (underage_cost - overage_cost) / (4 * lambda)
        return base_order .+ shift
    end

    function objective_on_log_lambda(log_lambda)
        lambda = exp(log_lambda)
        order = order_for_lambda(lambda)
        return _W2_DRO_newsvendor_dual_value(demands, weights, order, lambda, radius, underage_cost, overage_cost)
    end

    log_lambda = _golden_section_minimize(objective_on_log_lambda, -20.0, 20.0)
    return order_for_lambda(exp(log_lambda))
end
