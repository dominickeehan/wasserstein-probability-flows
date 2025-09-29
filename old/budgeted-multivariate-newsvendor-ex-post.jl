using Random, Statistics, StatsBase, Distributions
using LinearAlgebra, Plots, ProgressBars, IterTools

# --- Cost parameters ---
Cu = 4  # Underage cost
Co = 1  # Overage cost

# --- Newsvendor loss per scenario ---
newsvendor_loss(x::AbstractVector, ξ::AbstractVector) =
    sum(Cu * max(ξ[i] - x[i], 0) + Co * max(x[i] - ξ[i], 0) for i in eachindex(x))

# --- Analytic expected loss using univariate approximation per coordinate ---
function expected_newsvendor_loss(order::AbstractVector, demand_dists::Vector{<:UnivariateDistribution})
    total_cost = 0.0
    for i in eachindex(order)
        Q = order[i]
        μ = mean(demand_dists[i])
        σ = std(demand_dists[i])

        if σ == 0
            total_cost += Cu * max(μ - Q, 0) + Co * max(Q - μ, 0)
        else
            z = (Q - μ) / σ
            ϕ = pdf(Normal(), z)
            Φ = cdf(Normal(), z)
            overage = σ * (z * Φ + ϕ)
            underage = σ * (ϕ - z * (1 - Φ))
            total_cost += Cu * underage + Co * overage
        end
    end
    return total_cost
end

B = 300.0

using JuMP, MathOptInterface, Gurobi
env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
GRBsetintparam(env, "BarHomogeneous", 1)

function newsvendor_order(ξ::Matrix{Float64}, weights::Vector{Float64})
    # --- Absolute budget (required) ---
    #@assert isdefined(@__MODULE__, :B_abs) "Define global const B_abs::Float64 = <your absolute budget> before calling."
    #B = float(getfield(@__MODULE__, :B_abs))

    # --- Keep everything else as in your pipeline ---
    dim, T = size(ξ)
    @assert length(weights) == T "weights length must equal number of samples (T)."

    # Normalize positive weights (scaling doesn't affect the argmin)
    w = max.(0.0, weights)
    idx = findall(>(0.0), w)
    @assert !isempty(idx) "All weights are zero."
    w = w[idx]; w ./= sum(w)
    Ξ = ξ[:, idx]
    T_eff = length(idx)

    # Solve LP:
    #   minimize ∑_t w_t ∑_i s[i,t]
    #   s[i,t] ≥ -Cu * x[i] + Cu * Ξ[i,t]
    #   s[i,t] ≥  Co * x[i] - Co * Ξ[i,t]
    #   ∑_i x[i] ≤ B
    #   x[i] ≥ 0, s[i,t] ≥ 0
    #
    # Uses Gurobi quietly, like your original style.
    #
    # NOTE: Only this function is changed. Everything else stays the same.

    # Local imports so the rest of your file stays untouched


    model = Model(() -> Gurobi.Optimizer(env))

    @variable(model, x[1:dim] >= 0)
    @variable(model, s[1:dim, 1:T_eff] >= 0)

    @constraint(model, sum(x[i] for i in 1:dim) <= B)
    @constraint(model, [i=1:dim, t=1:T_eff], s[i,t] >= -Cu * x[i] + Cu * Ξ[i,t])
    @constraint(model, [i=1:dim, t=1:T_eff], s[i,t] >=  Co * x[i] - Co * Ξ[i,t])

    @objective(model, Min, sum(w[t] * sum(s[i,t] for i in 1:dim) for t in 1:T_eff))

    optimize!(model)
    term = JuMP.termination_status(model)

    if term == MathOptInterface.OPTIMAL || term == MathOptInterface.LOCALLY_SOLVED
        return value.(x)
    else
        # Fallback: keep your behavior stable—use decoupled quantiles then scale to meet the absolute budget
        q = Cu / (Cu + Co)
        x_dec = [quantile(ξ[i, :], Weights(weights), q) for i in 1:dim]
        total = sum(x_dec)
        return total > 0 ? (B / total) .* x_dec : zeros(dim)
    end
end

# --- Problem setup ---
Random.seed!(42)

dim = 5                   # Number of products
repetitions = 200         # Number of trials
history_length = 50       # Historical demand points

σ = 10.0



shift_distribution = MvNormal(zeros(dim), 1 * I)

# History: (dim × T) matrix for each repetition
demand_sequences = [zeros(dim, history_length) for _ in 1:repetitions]

# Future marginals: dim univariate normals per sample per rep
final_demand_distributions = [[Vector{UnivariateDistribution}(undef, dim) for _ in 1:1000] for _ in 1:repetitions]

for rep in 1:repetitions
    μ = ones(dim) * 100.0

    for t in 1:history_length
        demand_sequences[rep][:, t] .= rand(MvNormal(μ, Diagonal(fill(σ, dim))))
        μ .+= rand(shift_distribution)
    end

    for s in 1:length(final_demand_distributions[1])
        μs = μ .+ rand(shift_distribution)
        final_demand_distributions[rep][s] = [Normal(μs[i], σ) for i in 1:dim]
    end
end

# --- Fitting and evaluation ---
function parameter_fit(solve_for_weights, weight_parameters)
    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        demand_samples = demand_sequences[repetition]
        dim, T = size(demand_samples)

        weights = solve_for_weights(eachcol(demand_samples), weight_parameters[weight_parameter_index])
        weights = collect(weights)

        order = newsvendor_order(demand_samples, weights)
        dist_list = final_demand_distributions[repetition]

        costs[repetition][weight_parameter_index] = mean([
            expected_newsvendor_loss(order, dist_list[s]) for s in eachindex(dist_list)
        ])
    end

    best_index = argmin(mean(costs))
    minimal_costs = [costs[rep][best_index] for rep in 1:repetitions]

    display([
        round(mean(minimal_costs), digits=4),
        round(sem(minimal_costs), digits=4),
        round(weight_parameters[best_index], digits=4)
    ])

    return minimal_costs
end

# --- L1 distance for weighting ---
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)

# --- Weight methods (from separate file) ---
include("weights.jl")  # defines: windowing_weights, smoothing_weights, WPF_weights

# --- Run ---
parameter_fit(windowing_weights, history_length)

smoothing_costs = parameter_fit(smoothing_weights, [
    #LinRange(0.00001, 0.0001, 10);
    LinRange(0.0001, 0.001, 9);
    LinRange(0.002, 0.01, 9);
    LinRange(0.02, 1.0, 99)
])

WPF_costs = parameter_fit(WPF_weights, [
    LinRange(0.02, 0.1, 9);
    LinRange(0.2, 1.0, 9);
    LinRange(2, 10, 9);
    #LinRange(20, 100, 9);
    #LinRange(200, 1000, 9)
])

# --- Performance comparison ---
display(mean(WPF_costs - smoothing_costs) / mean(smoothing_costs)*100)
display(sign(mean(WPF_costs - smoothing_costs)) * mean(WPF_costs - smoothing_costs) / sem(WPF_costs - smoothing_costs))
