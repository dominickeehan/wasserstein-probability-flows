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

function solve_full_support_W1_DRO_risk_averse_portfolio(sample_returns, sample_weights, radius)

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

    # Full-support W1 DRO: no polyhedral support dual is used, so the Wasserstein
    # term reduces to the Lipschitz penalty of the worst affine CVaR piece.
    @objective(model, Min,
        - ρ*sum(sample_weights[i]*dot(x, sample_returns[i]) for i in 1:N) +
            (1-ρ)*τ +
            (1-ρ)*sum(sample_weights[i]*(1/α)*z[i] for i in 1:N) +
            radius*robust_lipschitz_coefficient*u
    )

    optimize!(model)

    return value.(x)
end

function solve_polyhedral_support_W1_DRO_risk_averse_portfolio(sample_returns, sample_weights, radius; support_lower_bound = -1.0)

    N = length(sample_returns)
    m = length(sample_returns[1])

    sample_weights = sample_weights/sum(sample_weights)
    support_lower_bounds = fill(support_lower_bound, m)
    if any(any(sample_returns[i] .< support_lower_bounds) for i in 1:N)
        error("Sample return below W1-DRO support lower bound $support_lower_bound")
    end

    mean_loss_coefficient = ρ
    cvar_loss_coefficient = ρ + (1-ρ)/α

    model = Model(portfolio_optimizer)

    @variables(model, begin
                            x[i=1:m] >= 0 # Portfolio weights (non-negative).
                            τ             # CVaR threshold.
                            λ >= 0        # Wasserstein dual multiplier.
                            s[i=1:N]      # Worst-case loss epigraph variables.
                            γ_mean[i=1:N, j=1:m] >= 0 # Support-dual variables for the mean-loss affine piece.
                            γ_tail[i=1:N, j=1:m] >= 0 # Support-dual variables for the tail-loss affine piece.
                      end)

    @constraint(model, sum(x) == 1) # Portfolio weights sum to 1

    # Esfahani-Kuhn support dual for the polyhedral return support ξ >= support_lower_bound.
    for i in 1:N
        support_slack = sample_returns[i] - support_lower_bounds

        @constraint(model,
            (1-ρ)*τ - mean_loss_coefficient*dot(x, sample_returns[i]) +
                sum(γ_mean[i,j]*support_slack[j] for j in 1:m) <= s[i])
        @constraint(model,
            (1-ρ)*(1-1/α)*τ - cvar_loss_coefficient*dot(x, sample_returns[i]) +
                sum(γ_tail[i,j]*support_slack[j] for j in 1:m) <= s[i])

        @constraint(model, [j=1:m], γ_mean[i,j] - mean_loss_coefficient*x[j] <= λ)
        @constraint(model, [j=1:m], mean_loss_coefficient*x[j] - γ_mean[i,j] <= λ)
        @constraint(model, [j=1:m], γ_tail[i,j] - cvar_loss_coefficient*x[j] <= λ)
        @constraint(model, [j=1:m], cvar_loss_coefficient*x[j] - γ_tail[i,j] <= λ)
    end

    @objective(model, Min,
        radius*λ + sum(sample_weights[i]*s[i] for i in 1:N)
    )

    optimize!(model)

    return value.(x)
end

function solve_W1_DRO_risk_averse_portfolio(sample_returns, sample_weights, radius; support_lower_bound = -1.0)

    # Comment out this return to use the easier full-support W1 DRO formulation below.
    #return solve_polyhedral_support_W1_DRO_risk_averse_portfolio(sample_returns,sample_weights,radius;support_lower_bound = support_lower_bound,)

    return solve_full_support_W1_DRO_risk_averse_portfolio(sample_returns, sample_weights, radius)
end

function fixed_mix_portfolio(sample_returns, parameter = 0)
    m = length(sample_returns[1])
    portfolio = zeros(m)
    portfolio .= 1/m

    return portfolio
end
