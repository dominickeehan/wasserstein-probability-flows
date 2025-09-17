using Random, Statistics, StatsBase, Distributions
using LinearAlgebra
using IterTools
using ProgressBars, Plots

using Statistics, StatsBase
using JuMP, MathOptInterface, Gurobi

# minimize_x    c'x  +  E[min_y{Qξ'y}]
# subject to    Ay >= Bξ - Tx
#               y >= 0
#               x >= 0

# x ∈ ℜⁿ
# ξ ∈ ℜᵐ
# y ∈ ℜʳ

# 3. Two-Stage Capacity Planning
# x = first-stage capacity investments (expensive but flexible),
# y = second-stage recourse actions (cheaper but constrained),
# ξ = uncertain demand vector.
# Interpretation: decide on factory capacity now, then meet random customer demand later with minimal operating cost.

env = Gurobi.Env()
GRBsetintparam(env, "OutputFlag", 0)
#GRBsetintparam(env, "BarHomogeneous", 1)
#GRBsetintparam(env, "NumericFocus", 3)

optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env))

function loss(x, ξ)

    Problem = Model(optimizer)

    @variables(Problem, begin; y[i=1:r] >= 0; end)
    @constraints(Problem, begin; A*y .>= B*ξ - T*x; end)

    @objective(Problem, Min, c'*x + (Q*ξ)'*y)

    optimize!(Problem)

    assert(is_solved_and_feasible(Problem))

    return objective_value(Problem)

end

function x(ξ, weights)

    nonzero_weight_indices = weights .> 0
    weights = weights[nonzero_weight_indices]
    weights = weights/sum(weights)
    ξ = ξ[nonzero_weight_indices]

    T = length(ξ)

    Problem = Model(optimizer)

    @variables(Problem, begin
                            x[i=1:n] >= 0
                            y[i=1:r] >= 0
                        end)

    for t in 1:T
        @constraints(Problem, begin; A*y[t] .>= B*ξ[t] - T*x; end)
    end

    @objective(Problem, Min, c'*x + sum(weights[t]*(Q*ξ[t])'*y[t] for t in 1:T))

    optimize!(Problem)

    assert(is_solved_and_feasible(Problem))

    return value(x)

end

Random.seed!(42)

dimension = m
repetitions = 100
history_length = 100

μ = 100
σ = 10

shift_distribution = MvNormal(zeros(dimension), 1 * I)

ξ = [zeros(dimension, history_length) for _ in 1:repetitions]
final_ξ = [[Vector{Vector}(undef, dimension) for _ in 1:1000] for _ in 1:repetitions]

for repetition in 1:repetitions
    μs = ones(dimension) * μ

    for t in 1:history_length
        ξ[repetition][:, t] = rand(MvNormal(μs, Diagonal(fill(σ, dimension))))
        μs += rand(shift_distribution)

    end

    for i in 1:length(final_ξ[1])
        ν = μs + rand(shift_distribution)
        final_ξ[repetition][i] = rand(MvNormal(ν, Diagonal(fill(σ, dimension))))

    end
end


function parameter_fit(solve_for_weights, weight_parameters)
    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        ξ_samples = ξ[repetition]
        weights = solve_for_weights(eachcol(ξ_samples), weight_parameters[weight_parameter_index])
        x = x(ξ_samples, weights)

        costs[repetition][weight_parameter_index] = mean([
            loss(x, final_ξ[repetition][i]) for i in eachindex(final_ξ[repetition])])
    end

    minimal_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][minimal_index] for repetition in 1:repetitions]

    display([
        round(mean(minimal_costs), digits=4),
        round(sem(minimal_costs), digits=4),
        round(weight_parameters[minimal_index], digits=4)
    ])

    return minimal_costs
end


d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)

include("weights.jl")

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

display(mean(WPF_costs - smoothing_costs) / mean(smoothing_costs)*100)
display(sign(mean(WPF_costs - smoothing_costs)) * mean(WPF_costs - smoothing_costs) / sem(WPF_costs - smoothing_costs))
