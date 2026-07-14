using JuMP, LinearAlgebra, COPT

portfolio_optimizer = let attributes = Pair{String, Int}[
        "Logging" => 0,
        "LogToConsole" => 0,
    ]
    copt_threads = get(ENV, "COPT_THREADS", Threads.nthreads() > 1 ? "1" : "")
    if !isempty(copt_threads)
        number_of_copt_threads = parse(Int, copt_threads)
        @assert number_of_copt_threads > 0 "COPT_THREADS must be positive."
        push!(attributes, "Threads" => number_of_copt_threads)
    end
    optimizer_with_attributes(COPT.Optimizer, attributes...)
end

function unweighted_cvar(costs)
    N = length(costs) # Number of equally weighted cost samples.
    @assert N > 0 "CVaR requires at least one cost sample."
    @assert 0 < α <= 1 "CVaR tail probability α must be in (0, 1]."

    tail_mass = α * N
    full_tail_samples = floor(Int, tail_mass)
    tail_samples = ceil(Int, tail_mass)

    # CVaR is the mean of the largest α*N samples, with fractional weight on
    # the boundary sample when α*N is not an integer.
    largest_costs = partialsort(costs, 1:tail_samples; rev = true)
    tail_sum = sum(@view largest_costs[1:full_tail_samples])
    if full_tail_samples < tail_samples
        tail_sum += (tail_mass - full_tail_samples)*largest_costs[tail_samples]
    end

    return tail_sum / tail_mass
end

function solve_risk_averse_portfolio(returns, weights)

    nonzero_weight_indices = weights .> 0.0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    returns = returns[nonzero_weight_indices]

    N = length(returns) # Number of return samples.
    m = length(returns[1]) # Number of assets.

    model = Model(portfolio_optimizer)  

    @variables(model, begin
                            x[i=1:m] >= 0 # Portfolio weights (non-negative).
                            τ             # CVaR threshold.
                            z[i=1:N] >= 0 # Slack variables for CVaR.
                      end)

    @constraint(model, sum(x) == 1) # Portfolio weights sum to 1

    @objective(model, Min,
        - ρ*sum(weights[i]*dot(x, returns[i]) for i in 1:N) + (1-ρ) * τ + (1-ρ) * sum(weights[i]*(1/α)*z[i] for i in 1:N)
    )

    for i in 1:N; @constraint(model, z[i] >= -dot(x, returns[i]) - τ); end # CVaR constraints.

    optimize!(model)

    return value.(x)
end




"""
    solve_W1_DRO_risk_averse_portfolio(returns, weights, radius)

Restricted-support W1-DRO mean-CVaR portfolio via the Esfahani-Kuhn reformulation,
with support fixed to the economically natural lower bound `return >= -100%`.

For a lower-bound-only support the objective-minimizing support multipliers are
`max(c*x[j] - λ, 0)`, independent of the sample, so the generic per-sample gamma[i,j]
collapse to one shared correction h[j] per affine loss piece. This drops the O(N*m)
support-dual variables and constraints of the generic polyhedral dual.

References:
- Mohajerin Esfahani & Kuhn (2018), Data-driven distributionally robust optimization
  using the Wasserstein metric: performance guarantees and tractable reformulations,
  Mathematical Programming 171(1-2). (Polyhedral W1 dual for piecewise affine losses;
  the mean-CVaR portfolio application.)
"""
function solve_W1_DRO_risk_averse_portfolio(returns, weights, radius)

    nonzero_weight_indices = weights .> 0.0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    returns = returns[nonzero_weight_indices]

    N = length(returns)
    m = length(returns[1])

    weights = weights/sum(weights)

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
    # For the -100% lower-bound support the slacks (returns[i] .+ 1.0) are nonnegative, so the
    # objective drives every multiplier to its lower bound max(c*x[j] - λ, 0), independent of i. Hence
    # (1) one shared h[j] per affine piece replaces the N per-sample multipliers, and (2) the upper-bound
    # rows h[j] - c*x[j] <= λ are dropped: they never bind at the minimizing h, so only these lower
    # bounds are kept.
    @constraint(model, [j=1:m], h_mean[j] >= mean_loss_coefficient*x[j] - λ)
    @constraint(model, [j=1:m], h_tail[j] >= cvar_loss_coefficient*x[j] - λ)

    for i in 1:N
        support_slack = returns[i] .+ 1.0 # Slack against the -100% support lower bound.

        @constraint(model,
            (1-ρ)*τ - mean_loss_coefficient*dot(x, returns[i]) +
                sum(h_mean[j]*support_slack[j] for j in 1:m) <= s[i])
        @constraint(model,
            (1-ρ)*(1-1/α)*τ - cvar_loss_coefficient*dot(x, returns[i]) +
                sum(h_tail[j]*support_slack[j] for j in 1:m) <= s[i])
    end

    @objective(model, Min,
        radius*λ + sum(weights[i]*s[i] for i in 1:N)
    )

    optimize!(model)

    return value.(x)
end




function fixed_mix_portfolio(returns, nothing)
    m = length(returns[1])
    portfolio = zeros(m)
    portfolio .= 1/m

    return portfolio
end

5
