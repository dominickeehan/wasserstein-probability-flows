using JuMP, LinearAlgebra

"""
    solve_wasserstein_dro_portfolio(sample_returns, center_weights, radius; optimizer, risk_aversion, cvar_alpha)

Solve the mean-CVaR portfolio problem over a 1-Wasserstein ball centered at
`center_weights` on `sample_returns`, using the L1 ground metric.
"""
function solve_wasserstein_dro_portfolio(sample_returns, center_weights, radius; optimizer, risk_aversion, cvar_alpha)

    N = length(sample_returns)
    m = length(sample_returns[1])

    center_weights = center_weights/sum(center_weights)
    robust_lipschitz_coefficient = risk_aversion + (1-risk_aversion)/cvar_alpha

    model = Model(optimizer)

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
        - risk_aversion*sum(center_weights[i]*dot(x, sample_returns[i]) for i in 1:N) +
            (1-risk_aversion)*τ +
            (1-risk_aversion)*sum(center_weights[i]*(1/cvar_alpha)*z[i] for i in 1:N) +
            radius*robust_lipschitz_coefficient*u
    )

    optimize!(model)

    return value.(x)
end
