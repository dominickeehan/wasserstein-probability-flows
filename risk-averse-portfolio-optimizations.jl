using JuMP, LinearAlgebra, COPT

portfolio_optimizer = optimizer_with_attributes(COPT.Optimizer, "Logging" => 0, "LogToConsole" => 0,)

function unweighted_cvar(costs)

    N = length(costs) # Number of cost samples.

    model = Model(portfolio_optimizer)  

    @variables(model, begin  
                            τ             # CVaR threshold.
                            z[i=1:N] >= 0 # Slack variables for CVaR.
                      end)

    @objective(model, Min, τ + sum((1/N)*(1/α)*z[i] for i in 1:N))

    for i in 1:N; @constraint(model, z[i] >= costs[i] - τ); end # CVaR constraints.

    optimize!(model)

    return objective_value(model)
end

function solve_risk_averse_portfolio(sample_returns, sample_weights)

    N = length(sample_returns) # Number of return samples.
    m = length(sample_returns[1]) # Number of assets.

    model = Model(portfolio_optimizer)  

    @variables(model, begin
                            x[i=1:m] >= 0 # Portfolio weights (non-negative).
                            τ             # CVaR threshold.
                            z[i=1:N] >= 0 # Slack variables for CVaR.
                      end)

    @constraint(model, sum(x) == 1) # Portfolio weights sum to 1

    @objective(model, Min,
        - ρ*sum(sample_weights[i]*dot(x, sample_returns[i]) for i in 1:N) + (1-ρ) * τ + (1-ρ) * sum(sample_weights[i]*(1/α)*z[i] for i in 1:N)
    )

    for i in 1:N; @constraint(model, z[i] >= -dot(x, sample_returns[i]) - τ); end # CVaR constraints.

    optimize!(model)

    return value.(x)
end

function solve_W1_DRO_risk_averse_portfolio(sample_returns, sample_weights, radius)

    N = length(sample_returns)
    m = length(sample_returns[1])

    sample_weights = sample_weights/sum(sample_weights)
    robust_lipschitz_coefficient = ρ + (1-ρ)/α

    model = Model(portfolio_optimizer)

    @variables(model, begin
                            x[i=1:m] >= 0 # Portfolio weights (non-negative).
                            τ             # CVaR threshold.
                            z[i=1:N] >= 0 # Slack variables for CVaR.
                            u >= 0        # L∞ norm of x, dual to the L1 ground metric.
                      end)

    @constraint(model, sum(x) == 1) # Portfolio weights sum to 1
    @constraint(model, [i=1:m], x[i] <= u)

    for i in 1:N; @constraint(model, z[i] >= -dot(x, sample_returns[i]) - τ); end # CVaR constraints.

    @objective(model, Min,
        - ρ*sum(sample_weights[i]*dot(x, sample_returns[i]) for i in 1:N) +
            (1-ρ)*τ +
            (1-ρ)*sum(sample_weights[i]*(1/α)*z[i] for i in 1:N) +
            radius*robust_lipschitz_coefficient*u
    )

    optimize!(model)

    return value.(x)
end

function fixed_mix_portfolio(sample_returns, parameter = 0)
    m = length(sample_returns[1])
    portfolio = zeros(m)
    portfolio .= 1/m

    return portfolio
end
