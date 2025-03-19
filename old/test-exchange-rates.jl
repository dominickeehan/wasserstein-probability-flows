include("extract-exchange-rates.jl")

using LinearAlgebra

function fit_weighted_AR1_model(time_series, weights)
    m = length(time_series[1])

    lagged_time_series = time_series[2:end]
    time_series = time_series[1:end-1]

    X = hcat(ones(length(time_series)), stack(time_series)') # Leading column of ones for the additive term.
    weights = Diagonal(weights)
    least_squares_solution = (X'*weights*X) \ (X'*weights*stack(lagged_time_series)')
    
    μ = least_squares_solution[1, :] # First row is the intercept vector.
    A = least_squares_solution[2:end, :]' # Due to transposing these are ordered differently.
    return μ, A
end

function fit_weighted_AR2_model(time_series, weights)
    m = length(time_series[1])
    lagged_lagged_time_series = time_series[3:end]
    lagged_time_series = time_series[2:end-1]
    time_series = time_series[1:end-2]

    X = hcat(ones(length(time_series)), stack(time_series)', stack(lagged_time_series)') # Leading column of ones for the additive term.
    weights = Diagonal(weights)
    least_squares_solution = (X'*weights*X) \ (X'*weights*stack(lagged_lagged_time_series)')
    
    μ = least_squares_solution[1, :] # First row is the intercept vector.
    B = least_squares_solution[2:m+1, :]' # Remaining rows transposed are the coefficient matrices.
    A = least_squares_solution[m+2:end, :]' # Due to transposing these are ordered differently.
    return μ, A, B
end

loss_function(x,ξ) = (norm(x-ξ, 2))^2

training_testing_split = ceil(Int,0.7*length(extracted_data))
warm_up_period = ceil(Int,1.0*training_testing_split)-1
warm_up_data = extracted_data[1:warm_up_period]
training_data = extracted_data[warm_up_period+1:training_testing_split]
training_T = length(training_data)
testing_data = extracted_data[training_testing_split+1:end]
testing_T = length(testing_data)

windowing_parameters = [1000]
exponential_smoothing_parameters = LinRange(0.001,0.2,101)
probability_flow_parameters = LinRange(50,300,5) #LinRange(75,1000,6) #LinRange(100,3000,60)

using ProgressBars, IterTools
using Statistics, StatsBase
using Plots
function train_and_test_out_of_sample(parameters, solve_for_weights; catch_solve_for_weights = nothing)
    
    parameter_costs_in_training_stages = zeros((training_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(training_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data[1:t-1]]
        local paired_samples = [[samples[i], samples[i+1]] for i in 1:length(samples)-1]
        local sample_weights = nothing
        try; sample_weights = solve_for_weights(paired_samples, parameters[i]); catch; sample_weights = catch_solve_for_weights(paired_samples, parameters[i]); end
        local μ, A = fit_weighted_AR1_model(samples, sample_weights)
        local x = μ + A*samples[end]
        parameter_costs_in_training_stages[t,i] = loss_function(x, training_data[t])
    end

    parameter_costs_in_testing_stages = zeros((testing_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(testing_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data; testing_data[1:t-1]]
        local paired_samples = [[samples[i], samples[i+1]] for i in 1:length(samples)-1]
        local sample_weights = nothing
        try; sample_weights = solve_for_weights(paired_samples, parameters[i]); catch; sample_weights = catch_solve_for_weights(paired_samples, parameters[i]); end
        local μ, A = fit_weighted_AR1_model(samples, sample_weights)
        local x = μ + A*samples[end]
        parameter_costs_in_testing_stages[t,i] = loss_function(x, testing_data[t])
    end

    total_parameter_costs_in_previous_stages = [zeros(length(parameters)) for _ in 1:testing_T]
    total_parameter_costs_in_previous_stages[1] = vec(sum(parameter_costs_in_training_stages, dims=1))
    for t in 2:testing_T
        total_parameter_costs_in_previous_stages[t] = vec(total_parameter_costs_in_previous_stages[t-1] + parameter_costs_in_testing_stages[t-1,:])
    end

    realised_costs = [parameter_costs_in_testing_stages[t,argmin(total_parameter_costs_in_previous_stages[t])] for t in 1:testing_T] 
    μ = mean(realised_costs)
    s = sem(realised_costs)
    display("Realised out-of-sample cost: $μ ± $s")
    display(plot(parameters, vec(total_parameter_costs_in_previous_stages[end]+parameter_costs_in_testing_stages[end,:])/(training_T+testing_T)))

    return realised_costs

end

d(i,j,ξ_i,ξ_j) = 0
include("weight-solvers.jl")
SAA_costs = train_and_test_out_of_sample(windowing_parameters, solve_for_moving_average_weights)
train_and_test_out_of_sample(exponential_smoothing_parameters, solve_for_exponential_smoothing_weights)

d(i,j,ξ_i,ξ_j) = norm(ξ_i[1] - ξ_j[1], 1) + norm(ξ_i[2] - ξ_j[2], 1)
include("weight-solvers.jl")
WPF_costs = train_and_test_out_of_sample(probability_flow_parameters, solve_for_probability_flow; catch_solve_for_weights = nonconic_solve_for_probability_flow)

μ = mean(WPF_costs - SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")

d(i,j,ξ_i,ξ_j) = ifelse(i == j, 0, norm(ξ_i[1] - ξ_j[1], 1) + norm(ξ_i[2] - ξ_j[2], 1) + 0.1)
include("weight-solvers.jl")
#train_and_test_out_of_sample(probability_flow_parameters, solve_for_probability_flow; catch_solve_for_weights = nonconic_solve_for_probability_flow)

d(i,j,ξ_i,ξ_j) = sqrt(norm(ξ_i[1] - ξ_j[1], 2)^2 + norm(ξ_i[2] - ξ_j[2], 2)^2)
include("weight-solvers.jl")
#train_and_test_out_of_sample(probability_flow_parameters, solve_for_probability_flow; catch_solve_for_weights = nonconic_solve_for_probability_flow)

d(i,j,ξ_i,ξ_j) = ifelse(i == j, 0, sqrt(norm(ξ_i[1] - ξ_j[1], 2)^2 + norm(ξ_i[2] - ξ_j[2], 2)^2) + 0.01)
include("weight-solvers.jl")
#train_and_test_out_of_sample(probability_flow_parameters, solve_for_probability_flow; catch_solve_for_weights = nonconic_solve_for_probability_flow)

d(i,j,ξ_i,ξ_j) = max(norm(ξ_i[1] - ξ_j[1], Inf), norm(ξ_i[2] - ξ_j[2], Inf))
include("weight-solvers.jl")
#train_and_test_out_of_sample(probability_flow_parameters, solve_for_probability_flow; catch_solve_for_weights = nonconic_solve_for_probability_flow)

d(i,j,ξ_i,ξ_j) = ifelse(i == j, 0, max(norm(ξ_i[1] - ξ_j[1], Inf), norm(ξ_i[2] - ξ_j[2], Inf)) + 0.01)
include("weight-solvers.jl")
#train_and_test_out_of_sample(probability_flow_parameters, solve_for_probability_flow; catch_solve_for_weights = nonconic_solve_for_probability_flow)