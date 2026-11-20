using Random, Distributions, Statistics, StatsBase
using LinearAlgebra
using IterTools, ProgressBars, Printf

include("weights.jl")
include("interior-point-WPF-weights.jl")

# Trains and tests the methods of ex-post-multi-modal-and-multi-dimensional-newsvendor.jl out of sample,
# as the dairy and stock scripts do, and averages the results over simulated demand trajectories.



# Constant costs keep newsvendor_loss fast.
const Cu = 4  # Underage cost.
const Co = 1  # Overage cost.

dimensions = 3
modes = 3

# Trajectories of periods periods each. Periods 1 to periods - testing_periods are for warm up and
# training, and each testing period uses the parameter of smallest realized cost over the previous
# parameter tuning window, as in the dairy and stock scripts (which use 24 periods). Longer windows
# are compared too.
trajectories = 30
periods = 200
testing_periods = 100
parameter_tuning_windows = [24, 48, 72]

# Initial demand-distribution parameters. Mixture of axis-aligned normals.
μ = [i*100 for i in 1:modes]
σ = 20

# Demand-mode shift-distribution parameters.
shift_distribution = MvNormal(zeros(dimensions), (15^2) * I)

newsvendor_loss(order, demand) =
    sum(Cu * max(demand[i] - order[i], 0) + Co * max(order[i] - demand[i], 0) for i in eachindex(order))

# Order the smallest demand whose cumulative weight reaches the critical ratio Cu / (Cu + Co) in
# each dimension. This is the newsvendor order of the weighted sample.
function newsvendor_order(demands, weights)
        q = Cu / (Cu + Co)

        order = zeros(dimensions)
        for i in 1:dimensions
                sorted = sortperm([demands[t][i] for t in eachindex(demands)])
                cumulative_weights = cumsum(weights[sorted])
                order[i] = demands[sorted[findfirst(>=(q * sum(weights)), cumulative_weights)]][i]

        end

        return order
end

# Expected newsvendor loss of an order when demand is an equal mixture of normals with the given
# mode means and standard deviation σ in every dimension. For one normal and z = (order - mean) / σ
# this is
#       σ * (Cu * (φ(z) - z * (1 - Φ(z))) + Co * (φ(z) + z * Φ(z))).
function expected_newsvendor_loss(order, means)

        loss = 0.0
        for i in eachindex(order), mean in means
                z = (order[i] - mean[i]) / σ
                loss += σ * (Cu * (pdf(Normal(), z) - z * (1 - cdf(Normal(), z))) + Co * (pdf(Normal(), z) + z * cdf(Normal(), z))) / length(means)

        end

        return loss
end



# Mode means and demand of every period of every trajectory.
Random.seed!(42)

mode_means = [Vector{Vector{Vector{Float64}}}(undef, periods) for _ in 1:trajectories]
demands = [Vector{Vector{Float64}}(undef, periods) for _ in 1:trajectories]

for trajectory in 1:trajectories

        μs = [ones(dimensions) * μ[i] for i in 1:modes]

        for t in 1:periods

                mode_means[trajectory][t] = copy.(μs)
                demands[trajectory][t] = rand(MixtureModel(MvNormal, [(μs[i], Diagonal(fill((σ)^2, dimensions))) for i in 1:modes]))

                for i in eachindex(μs); μs[i] += rand(shift_distribution); end

        end
end



testing_range = (periods - testing_periods + 1):periods
costed_range = (first(testing_range) - maximum(parameter_tuning_windows)):periods

# Realized and expected cost of every parameter in every costed period of every trajectory. Each
# period orders from the weights of all earlier demands of its trajectory.
function period_costs(solve_for_weights, parameters, distance_function)

        realized = [zeros(periods, length(parameters)) for _ in 1:trajectories]
        expected = [zeros(periods, length(parameters)) for _ in 1:trajectories]

        Threads.@threads for (t, trajectory) in ProgressBar(collect(IterTools.product(costed_range, 1:trajectories)))

                history = demands[trajectory][1:t-1]

                for k in eachindex(parameters)
                        order = newsvendor_order(history, solve_for_weights(history, parameters[k], distance_function))
                        realized[trajectory][t, k] = newsvendor_loss(order, demands[trajectory][t])
                        expected[trajectory][t, k] = expected_newsvendor_loss(order, mode_means[trajectory][t])
                end

        end

        return realized, expected
end

# Average expected and realized testing cost of each trajectory with parameters tuned out of sample
# over each tuning window, the parameters chosen, and the cost of the best single parameter chosen
# in hindsight over all trajectories (ex post).
function train_and_test(solve_for_weights, parameters, distance_function)

        realized, expected = period_costs(solve_for_weights, parameters, distance_function)

        tuned = Dict()
        for window in parameter_tuning_windows
                expected_costs, realized_costs, chosen_parameters = zeros(trajectories), zeros(trajectories), []
                for trajectory in 1:trajectories
                        for t in testing_range
                                tuning_costs = vec(sum(realized[trajectory][(t - window):(t - 1), :], dims = 1))
                                k = argmin(tuning_costs)
                                expected_costs[trajectory] += expected[trajectory][t, k] / testing_periods
                                realized_costs[trajectory] += realized[trajectory][t, k] / testing_periods
                                push!(chosen_parameters, parameters[k])

                        end
                end
                tuned[window] = (expected_costs, realized_costs, chosen_parameters)

        end

        ex_post_index = argmin([mean(mean(expected[trajectory][testing_range, k]) for trajectory in 1:trajectories) for k in eachindex(parameters)])
        ex_post_costs = [mean(expected[trajectory][testing_range, ex_post_index]) for trajectory in 1:trajectories]

        return tuned, ex_post_costs, parameters[ex_post_index]
end



LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))
L1(ξ, ζ) = norm(ξ - ζ, 1)

methods = [("SAA", windowing_weights, [periods], 0),
           ("Windowing", windowing_weights, unique(ceil.(Int, LogRange(1, periods, 30))), 0),
           ("Smoothing", smoothing_weights, [[0]; LogRange(1e-4, 1, 30)], 0),
           ("WPF L1", interior_point_WPF_weights, [[0]; LogRange(1e-3, 1e0, 30); Inf], L1)]

results = Dict()
for (name, solve_for_weights, parameters, distance_function) in methods
        println(name)
        results[name] = train_and_test(solve_for_weights, parameters, distance_function)

end

# Costs of each method and their percentage differences from smoothing, paired by trajectory, for
# each tuning window and ex post.
for window in parameter_tuning_windows
        println("\nTrained and tested with a tuning window of $window periods")
        smoothing_costs = results["Smoothing"][1][window][1]

        for (name, _, _, _) in methods
                expected_costs, realized_costs, chosen_parameters = results[name][1][window]
                difference = 100 * (expected_costs - smoothing_costs) / mean(smoothing_costs)
                @printf("%-10s expected %.1f ± %.1f (%+.1f ± %.1f%% from smoothing), realized %.1f ± %.1f, median parameter %.4g\n",
                        name, mean(expected_costs), sem(expected_costs), mean(difference), sem(difference), mean(realized_costs), sem(realized_costs), median(chosen_parameters))

        end
end

println("\nEx post")
for (name, _, _, _) in methods
        _, ex_post_costs, ex_post_parameter = results[name]
        difference = 100 * (ex_post_costs - results["Smoothing"][2]) / mean(results["Smoothing"][2])
        @printf("%-10s expected %.1f ± %.1f (%+.1f ± %.1f%% from smoothing) at %.4g\n", name, mean(ex_post_costs), sem(ex_post_costs), mean(difference), sem(difference), ex_post_parameter)

end
