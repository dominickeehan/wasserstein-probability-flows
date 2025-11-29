# Dominic Keehan : 2025

using JuMP, MathOptInterface
using COPT, Ipopt


WPF_optimizer = optimizer_with_attributes(COPT.ConeOptimizer, "Logging" => 0, "LogToConsole" => 0, "BarIterLimit" => 50)  

"""
    WPF_weights

observations [vector of vectors] : Historical observations. ([1] being the oldest [end] being the most recent.)
λ [scalar] : Regularisation parameter.
d [function] : Metric defining distances between observations.

Solve for the terminal probability distribution under the Wasserstein probability flow as represented with a hypographical formulation using the exponential cone. 
Occasionaly this gets stuck; see Ipopt_WPF_weights for a slower but reliable alternative.
"""
function WPF_weights(observations, λ, d)

    Problem = Model(WPF_optimizer)

    T = length(observations)
    
    if λ == Inf; weights = zeros(T); weights .= 1/T; return weights; end
    if λ == 0; weights = zeros(T); weights[end] = 1; return weights; end
	
    @variables(Problem, begin
                            1>= p[i=1:T, t=1:2] >=0 # Probability allocated to observation i at time 1, Probability allocated to observation i at time T.
                            1>= p_diag[1:T] >= 0   # Diagonal probabilities.
                            1>= γ[i=1:T, j=1:T] >=0 # Probability transported from observation i to observation j between times i and i+1. 
                                z[t=1:T] <= 0 # Hypographical-reformulation variables. Upper bound enforces log probability constraints.

                        end)

    for t in 1:2
        @constraint(Problem, sum(p[i,t] for i in 1:T) == 1) # Initial and terminal probabilities sum to 1.

    end

    for i in 1:T
        for j in 1:i
            JuMP.fix(γ[i,j], 0; force = true) # Only transport probability from an earlier observation to a later observation (diagonality constraints).

        end
    end

    for t in 1:T # Conservation of probability flow.
        @constraint(Problem, p[t,1] + sum(γ[i,t] for i in 1:T) == p_diag[t]) # Entering.
        @constraint(Problem, p_diag[t] == p[t,2] + sum(γ[t,j] for j in 1:T)) # Exiting.
        
    end

    for t in 1:T # Hypographical-reformulation constraints.
        @constraint(Problem, [z[t]; 1; p_diag[t]] in MathOptInterface.ExponentialCone()) # Enforces the constraint z[t] <= log(p_diag[t])  
        
    end

    @objective(Problem, Max,
        sum(z[t] for t in 1:T) - λ*sum(d(observations[i],observations[j])*γ[i,j] for j in 1:T for i in 1:T))

    optimize!(Problem)

    try #is_solved_and_feasible(Problem) # Awkwardness to deal with "LP termination status 11 is not wrapped by COPT.jl."
        weights = [max(value(p[i, 2]),0) for i in 1:T] # Maximum likelihood terminal probability distribution.
        weights = weights/sum(weights)
		return weights

	catch
        return Ipopt_WPF_weights(observations, λ, d) # Use Ipopt Interior-Point Method if COPT Barrier Method fails.

    end

end

function Ipopt_WPF_weights(observations, λ, d)

    Problem = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    T = length(observations)
	
    @variables(Problem, begin
                            1>= p[i=1:T, t=1:2] >=0 # Probability allocated to observation i at time 1, Probability allocated to observation i at time T.
                            1>= p_diag[1:T] >= 0   # Diagonal probabilities.
                            1>= γ[i=1:T, j=1:T] >=0 # Probability transported from observation i to observation j between times i and i+1. 

                        end)

    for t in 1:2
        @constraint(Problem, sum(p[i,t] for i in 1:T) == 1) # Initial and terminal probabilities sum to 1.

    end

    for i in 1:T
        for j in 1:i
            JuMP.fix(γ[i,j], 0; force = true) # Only transport probability from an earlier observation to a later observation (diagonality constraints).

        end
    end

    for t in 1:T # Conservation of probability flow.
        @constraint(Problem, p[t,1] + sum(γ[i,t] for i in 1:T) == p_diag[t]) # Entering.
        @constraint(Problem, p_diag[t] == p[t,2] + sum(γ[t,j] for j in 1:T)) # Exiting.

    end

    @objective(Problem, Max,
        sum(ifelse(p_diag[t] > 0, log(p_diag[t]), -Inf) for t in 1:T) - # Defining log(0) = -Inf seems to result in faster solves here.
            λ*sum(d(observations[i],observations[j])*γ[i,j] for j in 1:T for i in 1:T))

    optimize!(Problem)

    weights = [max(value(p[i, 2]),0) for i in 1:T] # Maximum likelihood terminal probability distribution.
    weights = weights/sum(weights)
	    
    return weights

end

function smoothing_weights(observations, α, d)

    T = length(observations)

    if α == 0; weights = zeros(T); weights .= 1/T; return weights; end


    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights = weights/sum(weights)

    return weights
end


function windowing_weights(observations, s, d)

    T = length(observations)

    weights = zeros(T)

    if s >= T
        weights .= 1
    else
        for t in T:-1:T-(s-1)
            weights[t] = 1
        end
    end

    weights = weights/sum(weights)

    return weights
end


#using LinearAlgebra
#d(ξ, ζ) = norm(ξ - ζ, 2)
#@assert sum(abs.(WPF_weights([6.13, 7.85, 6.47, 4.91, 5.54, 7.13], 4.0, d) - [0.0, 0.275, 0.021, 0.0, 0.325, 0.379])) <= 1e-3

