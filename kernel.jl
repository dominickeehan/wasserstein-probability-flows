function median_pairwise_distance(samples)
    length(samples) <= 1 && return 1.0

    distances = Float64[]
    for i in 1:length(samples)-1
        for j in i+1:length(samples)
            push!(distances, norm(samples[i] - samples[j], 2))
        end
    end

    scale = median(distances)
    return scale > 0 ? scale : 1.0
end

function rbf_kernel(x, y, bandwidth::Float64)
    return exp(-norm(x-y, 2)^2 / (2*bandwidth^2))
end

function kernel_analog_forecast(samples, h_multiplier)
    inputs = samples[1:end-1]
    outputs = [samples[i+1] - samples[i] for i in 1:length(samples)-1]
    bandwidth = h_multiplier * median_pairwise_distance(inputs)

    scaled_distances = [-norm(samples[end] - input, 2)^2 / (2*bandwidth^2) for input in inputs]
    kernel_weights = exp.(scaled_distances .- maximum(scaled_distances))
    kernel_weights ./= sum(kernel_weights)

    predicted_change = zeros(length(samples[1]))
    for i in eachindex(outputs)
        predicted_change .+= kernel_weights[i] .* outputs[i]
    end

    return samples[end] + predicted_change
end

function train_and_test_kernel_out_of_sample(parameters)

    parameter_costs_in_training_stages = zeros((training_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(training_T:-1:1, eachindex(parameters))))

        local samples = [warm_up_data; training_data[1:t-1]]
        local x = kernel_analog_forecast(samples, parameters[i])

        parameter_costs_in_training_stages[t,i] = loss_function(x, training_data[t])
    end

    parameter_costs_in_testing_stages = zeros((testing_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(testing_T:-1:1, eachindex(parameters))))

        local samples = [warm_up_data; training_data; testing_data[1:t-1]]
        local x = kernel_analog_forecast(samples, parameters[i])

        parameter_costs_in_testing_stages[t,i] = loss_function(x, testing_data[t])
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]

    total_parameter_costs_in_previous_stages = [zeros(length(parameters)) for _ in 1:testing_T]
    for t in 1:testing_T

        total_parameter_costs_in_previous_stages[t] = vec(sum(parameter_costs[training_T+(t-1)-(parameter_tuning_window-1):training_T+(t-1),:], dims=1))
    end

    realised_costs = [parameter_costs_in_testing_stages[t,argmin(total_parameter_costs_in_previous_stages[t])] for t in 1:testing_T]
    μ = mean(realised_costs)
    s = sem(realised_costs)

    display("Cost: $μ ± $s")

    return realised_costs, parameters[argmin(vec(sum(parameter_costs[end-(parameter_tuning_window-1):end,:], dims=1))/(parameter_tuning_window))]
end

function extract_kernel_results(parameters)

    realised_costs, optimal_parameter = train_and_test_kernel_out_of_sample(parameters)

    average_cost = mean(realised_costs)
    percentage_average_difference = round((average_cost - SAA_average_cost) / SAA_average_cost * 100, digits = 1)
    percentage_sem_difference = round(sem(realised_costs - SAA_realised_costs) / SAA_average_cost * 100, digits = 1)

    display("difference to SAA: $percentage_average_difference ± $percentage_sem_difference")

    return round(average_cost, digits=digits), percentage_average_difference, percentage_sem_difference, optimal_parameter
end
