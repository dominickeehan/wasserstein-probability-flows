# Dominic Keehan : 2025

using JuMP, Ipopt
using LinearAlgebra

d(ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1) # 1-norm distance function.

"""
    compute_wasserstein_probability_flow

history_of_observations [vector of vectors] : Historical observations, with history_of_observations[1] being the oldest and history_of_observations[end] being the most recent.
λ [scalar] : regularisation parameter.

Solve for the terminal probability distribution under the Wasserstein probability flow.
"""

function compute_wasserstein_probability_flow(history_of_observations, λ)

    Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    T = length(history_of_observations)
    
	distances = zeros((T,T))
    for i in 1:T
        for j in 1:T
            distances[i,j] = d(history_of_observations[i],history_of_observations[j])
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



# compute_wasserstein_probability_flow([1, 5, 3, 1], 2)