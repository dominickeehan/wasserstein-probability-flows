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




"""
    solve_W1_DRO_risk_averse_portfolio(sample_returns, sample_weights, radius)

Restricted-support W1-DRO mean-CVaR portfolio via the Esfahani-Kuhn reformulation,
with support fixed to the economically natural lower bound `return >= -100%`.

For a lower-bound-only support the objective-minimizing support multipliers are
`max(c*x[j] - λ, 0)`, independent of the sample, so the generic per-sample gamma[i,j]
collapse to one shared correction h[j] per affine loss piece. This drops the O(N*m)
support-dual variables and constraints of the generic polyhedral dual.
"""
function solve_W1_DRO_risk_averse_portfolio(sample_returns, sample_weights, radius)

    N = length(sample_returns)
    m = length(sample_returns[1])

    sample_weights = sample_weights/sum(sample_weights)

    mean_loss_coefficient = ρ
    cvar_loss_coefficient = ρ + (1-ρ)/α

    model = Model(portfolio_optimizer)

    @variables(model, begin
                            x[i=1:m] >= 0 # Portfolio weights (non-negative).
                            τ             # CVaR threshold.
                            λ >= 0        # Wasserstein dual multiplier.
                            s[i=1:N]      # Worst-case loss epigraph variables.
                            h_mean[j=1:m] >= 0 # Shared lower-support correction for the mean-loss affine piece.
                            h_tail[j=1:m] >= 0 # Shared lower-support correction for the tail-loss affine piece.
                      end)

    @constraint(model, sum(x) == 1) # Portfolio weights sum to 1

    # Support-dual reduction of the generic Esfahani-Kuhn polyhedral dual. That dual carries
    # per-sample multipliers γ_mean[i,j], γ_tail[i,j] >= 0 with two-sided boxes |γ[i,j] - c*x[j]| <= λ.
    # For the -100% lower-bound support the slacks (sample_returns[i] .+ 1.0) are nonnegative, so the
    # objective drives every multiplier to its lower bound max(c*x[j] - λ, 0), independent of i. Hence
    # (1) one shared h[j] per affine piece replaces the N per-sample multipliers, and (2) the upper-bound
    # rows h[j] - c*x[j] <= λ are dropped: they never bind at the minimizing h, so only these lower
    # bounds are kept.
    @constraint(model, [j=1:m], h_mean[j] >= mean_loss_coefficient*x[j] - λ)
    @constraint(model, [j=1:m], h_tail[j] >= cvar_loss_coefficient*x[j] - λ)

    for i in 1:N
        support_slack = sample_returns[i] .+ 1.0 # Slack against the -100% support lower bound.

        @constraint(model,
            (1-ρ)*τ - mean_loss_coefficient*dot(x, sample_returns[i]) +
                sum(h_mean[j]*support_slack[j] for j in 1:m) <= s[i])
        @constraint(model,
            (1-ρ)*(1-1/α)*τ - cvar_loss_coefficient*dot(x, sample_returns[i]) +
                sum(h_tail[j]*support_slack[j] for j in 1:m) <= s[i])
    end

    @objective(model, Min,
        radius*λ + sum(sample_weights[i]*s[i] for i in 1:N)
    )

    optimize!(model)

    return value.(x)
end




function fixed_mix_portfolio(sample_returns, nothing)
    m = length(sample_returns[1])
    portfolio = zeros(m)
    portfolio .= 1/m

    return portfolio
end

5