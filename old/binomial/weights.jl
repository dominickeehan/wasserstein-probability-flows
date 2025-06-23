# Dominic Keehan : 2025

using JuMP, MathOptInterface
using COPT, Ipopt

WPF_optimizer = optimizer_with_attributes(COPT.ConeOptimizer, "Logging" => 0, "LogToConsole" => 0, "BarIterLimit" => 50)  

"""
    WPF_weights

history_of_observations [vector of vectors] : Historical observations. ([1] being the oldest and [end] being the most recent.)
d [function] : Metric defining distances between observations.
λ [scalar] : Regularisation parameter.

Solve for the terminal probability distribution under the Wasserstein probability flow as represented with a hypographical formulation using the exponential cone. 
Occasionaly this gets stuck; see catch_WPF_weights for a slower but reliable alternative.
"""
function WPF_weights(history_of_observations, λ)

    Problem = Model(WPF_optimizer)

    T = length(history_of_observations)
    
	distances = zeros((T,T))
    for i in 1:T
        for j in 1:T
            distances[i,j] = d(i,j,history_of_observations[i],history_of_observations[j])
        end
    end
	
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
        #@constraint(Problem, p[t,1] + sum(γ[i,t] for i in 1:T) == p[t,2] + sum(γ[t,j] for j in 1:T))
    end

    for t in 1:T # Hypographical-reformulation constraints.
        @constraint(Problem, [z[t]; 1; p_diag[t]] in MathOptInterface.ExponentialCone()) # Enforces the constraint z[t] <= log(p_diag[t])  
        #@constraint(Problem, [z[t]; 1; p[t,1] + sum(γ[i,t] for i in 1:T)] in MathOptInterface.ExponentialCone()) # Enforces the constraint z[t] <= log(p_diag[t])     
    end

    @objective(Problem, Max,
        sum(z[t] for t in 1:T) - λ*sum(distances[i,j]*γ[i,j] for j in 1:T for i in 1:T))

    optimize!(Problem)

    try #is_solved_and_feasible(Problem) # Awkwardness to deal with "LP termination status 11 is not wrapped by COPT.jl."
        
        weights = [max(value(p[i, 2]),0) for i in 1:T] # Maximum likelihood terminal probability distribution.
        weights = weights/sum(weights)
		return weights

	catch

	end

    return catch_WPF_weights(history_of_observations, λ) # Use Ipopt Interior-Point Method if COPT Barrier Method fails.

end


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
            λ*sum(distances[i,j]*γ[i,j] for j in 1:T for i in 1:T))

    optimize!(Problem)

    #if is_solved_and_feasible(Problem)
        weights = [max(value(p[i, 2]),0) for i in 1:T] # Maximum likelihood terminal probability distribution.
        weights = weights/sum(weights)
	    
        return weights

    #else
    #    return 1/T*ones(T)

    #end

end


function smoothing_weights(history_of_observations, α)

    T = length(history_of_observations)

    if α == 0; weights = zeros(T); weights .= 1/T; return weights; end


    weights = [α*(1-α)^(t-1) for t in T:-1:1]
    weights = weights/sum(weights)

    return weights
end


function windowing_weights(history_of_observations, window_size)

    T = length(history_of_observations)

    weights = zeros(T)

    if window_size >= T
        weights .= 1
    else
        for t in T:-1:T-(window_size-1)
            weights[t] = 1
        end
    end

    weights = weights/sum(weights)

    return weights
end

