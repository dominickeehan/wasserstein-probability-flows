include("extract-dairy-prices.jl")

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
warm_up_period = ceil(Int,0.5*training_testing_split)-1 # Needs to be small enough to allow parameter_tuning_window after.
warm_up_data = extracted_data[1:warm_up_period]
training_data = extracted_data[warm_up_period+1:training_testing_split]
training_T = length(training_data)
testing_data = extracted_data[training_testing_split+1:end]
testing_T = length(testing_data)

parameter_tuning_window = 3*12

windowing_parameters = round.(Int, LinRange(10,training_testing_split-parameter_tuning_window,51))
SES_parameters = LinRange(0.001,0.9,51)
WPF_parameters = LinRange(0,150,21) #LinRange(0,150,51) #LinRange(75,1000,6) #LinRange(100,3000,60)

using ProgressBars, IterTools
using Statistics, StatsBase
using Plots, Measures
function train_and_test_out_of_sample(parameters, solve_for_weights; save_cost_plot_as = nothing)
    
    parameter_costs_in_training_stages = zeros((training_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(training_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data[1:t-1]]
        local paired_samples = [[samples[i], samples[i+1]] for i in 1:length(samples)-1]
        local sample_weights = solve_for_weights(paired_samples, parameters[i])
        local μ, A = fit_weighted_AR1_model(samples, sample_weights)
        local x = μ + A*samples[end]
        parameter_costs_in_training_stages[t,i] = loss_function(x, training_data[t])
    end

    parameter_costs_in_testing_stages = zeros((testing_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(testing_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data; testing_data[1:t-1]]
        local paired_samples = [[samples[i], samples[i+1]] for i in 1:length(samples)-1]
        local sample_weights = solve_for_weights(paired_samples, parameters[i])
        local μ, A = fit_weighted_AR1_model(samples, sample_weights)
        local x = μ + A*samples[end]
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
    display("Realised out-of-sample cost: $μ ± $s")

    default() # Reset plot defaults.

    gr(size = (600,400))
    
    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 17)
    secondary_font = Plots.font(font_family, pointsize = 11)
    legend_font = Plots.font(font_family, pointsize = 16)
    
    default(framestyle = :box,
            grid = true,
            #gridlinewidth = 1.0,
            gridalpha = 0.075,
            #minorgrid = true,
            #minorgridlinewidth = 1.0, 
            #minorgridalpha = 0.075,
            #minorgridlinestyle = :dash,
            tick_direction = :in,
            xminorticks = 0, 
            yminorticks = 0,
            fontfamily = font_family,
            guidefont = primary_font,
            tickfont = secondary_font,
            legendfont = legend_font)
    
    plt = plot(parameters, 
            vec(sum(parameter_costs[end-(parameter_tuning_window-1):end,:], dims=1))/(parameter_tuning_window),
            ribbon = sem.([parameter_costs[end-(parameter_tuning_window-1):end,parameter] for parameter in eachindex(parameters)]),
            xlabel = "\$λ\$", 
            ylabel = "Expected cost",
            legend = nothing,
            legendfonthalign = :center,
            color = palette(:tab10)[1],
            alpha = 1,
            linestyle = :dash,
            linewidth = 1,
            fillalpha = .1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)
    
    display(plt);

    if !(save_cost_plot_as === nothing); savefig(plt, save_cost_plot_as); end

    return realised_costs, parameters[argmin(vec(sum(parameter_costs[end-(parameter_tuning_window-1):end,:], dims=1))/(parameter_tuning_window))]

end

d(i,j,ξ_i,ξ_j) = 0
include("weights.jl")
SAA_costs, _ = train_and_test_out_of_sample(length(extracted_data), windowing_weights)
windowing_costs, _ = train_and_test_out_of_sample(windowing_parameters, windowing_weights)
μ = mean(windowing_costs - SAA_costs)
s = sem(windowing_costs - SAA_costs)
display("Windowing - SAA: $μ ± $s")

SES_costs, _ = train_and_test_out_of_sample(SES_parameters, SES_weights) # ceil(Int,0.7*training_testing_split)-1

μ = mean(SES_costs - SAA_costs)
s = sem(SES_costs - SAA_costs)
display("SES - SAA: $μ ± $s")


#=
d(i,j,ξ_i,ξ_j) = norm(ξ_i[1] - ξ_j[1], 1) + norm(ξ_i[2] - ξ_j[2], 1)
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights)

μ = mean(WPF_costs - SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")
=#

d(i,j,ξ_i,ξ_j) = ifelse(i == j, 0, norm(ξ_i[1] - ξ_j[1], 1) + norm(ξ_i[2] - ξ_j[2], 1))# + 0.001) # 0.001 ?
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights; save_cost_plot_as = "figures/dairy-prices-WPF_{1+s}-parameter-costs.pdf")

μ = mean(WPF_costs - SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")

weights = WPF_weights([[extracted_data[i], extracted_data[i+1]] for i in 1:length(extracted_data)-1], WPF_parameter)

default() # Reset plot defaults.

gr(size = (700,515+3.5))

font_family = "Computer Modern"
primary_font = Plots.font(font_family, pointsize = 17)
secondary_font = Plots.font(font_family, pointsize = 11)
legend_font = Plots.font(font_family, pointsize = 11)

default(framestyle = :box,
        grid = true,
        #gridlinewidth = 1.0,
        gridalpha = 0.075,
        #minorgrid = true,
        #minorgridlinewidth = 1.0, 
        #minorgridalpha = 0.075,
        #minorgridlinestyle = :dash,
        tick_direction = :in,
        xminorticks = 0, 
        yminorticks = 0,
        fontfamily = font_family,
        guidefont = primary_font,
        tickfont = secondary_font,
        legendfont = legend_font)

colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]]

plt_extracted_data = plot(1:14*1*12, 
        stack(log_dairy_prices)',
        xticks = (1*6+1:1*12:14*1*12),
        xlims = (-5,14*1*12+6),
        xformatter = :none,
        ylabel = "Log price (\$/t)",
        labels = nothing, 
        color = colors,
        linewidth = 1,
        topmargin = 0pt, 
        rightmargin = 0pt,
        bottommargin = 0pt, 
        leftmargin = 5pt)

A = 2:14*1*12
WPF_parameter = round(Int,WPF_parameter)
println("\$λ\$ = $WPF_parameter")

plt_probabilities = plot(A[weights .>= 1e-3], 
                weights[weights .>= 1e-3],
                xlabel = "Time (year)",
                xticks = (1*6+1:1*12:14*1*12, ["2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]),
                xlims = (-5,14*1*12+6),
                ylabel = "Probability", # at \$λ=$WPF_parameter\$",
                seriestype=:sticks,
                linestyle=:solid,
                linewidth = 1,
                seriescolor = palette(:tab10)[8],
                marker = nothing,
                markersize = 2.0,
                markercolor = palette(:tab10)[8],
                markerstrokecolor = :black,
                markerstrokewidth = 0.5,
                label = nothing,
                topmargin = 0pt, 
                rightmargin = 0pt,
                bottommargin = 3.5pt, 
                leftmargin = 0pt)

figure = plot(plt_extracted_data, plt_probabilities, layout=@layout([a; b]))
display(figure)
savefig(figure, "figures/dairy-prices-WPF_{1+s}-assigned-probability-to-historical-observations.pdf")

#=
d(i,j,ξ_i,ξ_j) = sqrt(norm(ξ_i[1] - ξ_j[1], 2)^2 + norm(ξ_i[2] - ξ_j[2], 2)^2)
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights)

μ = mean(WPF_costs - SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")

d(i,j,ξ_i,ξ_j) = max(norm(ξ_i[1] - ξ_j[1], Inf), norm(ξ_i[2] - ξ_j[2], Inf))
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights)

μ = mean(WPF_costs - SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")
=#
