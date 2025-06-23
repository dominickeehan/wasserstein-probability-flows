using Random, Statistics, StatsBase, Distributions

Random.seed!(42)

shift_distribution = Normal(0, 1)

repetitions = 300
history_length = 20

value_sequences = [zeros(history_length+1) for _ in 1:repetitions]
value_distributions = [[Normal(0, 0) for _ in 1:history_length+1] for _ in 1:repetitions]

for repetition in 1:repetitions
    μ = 100
    σ = 10

    for t in 1:history_length+1
        value_distributions[repetition][t] = Normal(μ, σ)
        value_sequences[repetition][t] = rand(value_distributions[repetition][t])

        μ = μ + rand(shift_distribution)
    end
end

using Plots
using ProgressBars
function parameter_fit(solve_for_weights, weight_parameters)

    sample_weights = [zeros(length(history_length)) for _ in 1:repetitions]

    Threads.@threads for repetition in ProgressBar(1:repetitions)
        local value_samples = value_sequences[repetition][1:history_length]
        sample_weights[repetition] = solve_for_weights(value_samples, weight_parameters[1])

    end

    plt = plot(1:history_length, mean(sample_weights), ribbon=sem.(sample_weights))
    plot!(1:history_length, SES_weights(1:history_length, 0.09))

    display(plt)

end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
# Dominic Keehan : 2025

using JuMP, Ipopt

function catch_WPF_weights(history_of_observations, λ)

    Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    T = length(history_of_observations)
    
	distances = zeros((T,T))
    for i in 1:T
        for j in 1:T
            distances[i,j] = d(i,j,history_of_observations[i],history_of_observations[j])
        end
    end
	
    @variables(Problem, begin
                            1>= p[i=1:T, t=1:2] >=0 # Probability allocated to observation i at time 1, Probability allocated to observation i at time T.
                            1>= p_diag[1:T] >= 0   # Diagonal probailities.
                            1>= γ[i=1:T, j=1:T] >=0 # Probability transported from observation i to observation j between times i and i+1. 
                        end)

    for t in 1:2
        @constraint(Problem, sum([p[i,t] for i in 1:T]) == 1) # Initial and terminal probabilities sum to 1.
    end

    for i in 1:T
        for j in 1:i
            JuMP.fix(γ[i,j], 0; force = true) # Only transport probability from an earlier observation to a later observation (diagonality constraints).
        end
    end

    for t in 1:T # Conservation of probability flow.
        @constraint(Problem, p[t,1] + sum([γ[i,t] for i in 1:T]) == p_diag[t]) # Entering.
        @constraint(Problem, p_diag[t] == p[t,2] + sum([γ[t,j] for j in 1:T])) # Exiting.
    end

    @objective(Problem, Max,
        sum([ifelse(p_diag[t] > 0, log(p_diag[t]), -Inf) for t in 1:T]) - # Defining log(0) = -Inf seems to result in faster solves here.
            λ*sum([distances[i,j]*γ[i,j] for j in 1:T for i in 1:T]))

    optimize!(Problem)

    return [max(value(p[i, 2]),0) for i in 1:T] # Maximum likelihood terminal probability distribution.
end






parameter_fit(catch_WPF_weights, [10])



function SES_weights(history_of_observations, α)

    T = length(history_of_observations)

    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights .= weights/sum(weights)

    return weights
end
