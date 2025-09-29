include("extract-dairy-prices.jl")


using LinearAlgebra
using Statistics, StatsBase
using ProgressBars, IterTools
using Plots, Measures


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

loss_function(x,ξ) = (norm(x-ξ, 2))^2


training_testing_split = ceil(Int,0.7*length(extracted_data))

warm_up_period = ceil(Int,0.7*training_testing_split)-1 # Needs to be small enough to allow parameter_tuning_window after.
warm_up_data = extracted_data[1:warm_up_period]

training_data = extracted_data[warm_up_period+1:training_testing_split]
training_T = length(training_data)

testing_data = extracted_data[training_testing_split+1:end]
testing_T = length(testing_data)

parameter_tuning_window = 2*12 # 2*12


windowing_parameters = round.(Int, LinRange(12,length(extracted_data),27))
smoothing_parameters = [LinRange(.001,.01,10); LinRange(.01,.1,10); LinRange(.1,.9,9)]
WPF_parameters = [LinRange(10,100,10); LinRange(200,1000,9); LinRange(2000,10000,9)] #[LinRange(10,100,100); LinRange(100,1000,100)]

#throw=thrwo

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

    display("Cost: $μ ± $s")


    default() # Reset plot defaults.

    gr(size = (275+6,183+6).*sqrt(3))

    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 12)
    secondary_font = Plots.font(font_family, pointsize = 10)
    legend_font = Plots.font(font_family, pointsize = 11)
    
    default(framestyle = :box,
            grid = true,
            #gridlinewidth = 1.0,
            gridalpha = 0.075,
            minorgrid = true,
            #minorgridlinewidth = 1.0, 
            minorgridalpha = 0.075,
            minorgridlinestyle = :dash,
            tick_direction = :in,
            xminorticks = 9, 
            yminorticks = 0,
            fontfamily = font_family,
            guidefont = primary_font,
            tickfont = secondary_font,
            legendfont = legend_font)
    
    plt = plot([float.(parameters)], 
            vec(sum(parameter_costs[end-(parameter_tuning_window-1):end,:], dims=1))/(parameter_tuning_window),
            ribbon = sem.([parameter_costs[end-(parameter_tuning_window-1):end,parameter] for parameter in eachindex(parameters)]),
            xscale = :log10,
            xticks = [10, 100, 1000, 10000],
            xlabel = "Penalty parameter, \$λ\$", 
            ylabel = "Average cost",
            legend = nothing,
            legendfonthalign = :center,
            color = :black,#palette(:tab10)[1],
            alpha = 1,
            linestyle = :solid,
            linewidth = 1,
            fillalpha = .1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 6pt, 
            leftmargin = 6pt)
    
    display(plt);

    if !(save_cost_plot_as === nothing); savefig(plt, save_cost_plot_as); end

    return realised_costs, parameters[argmin(vec(sum(parameter_costs[end-(parameter_tuning_window-1):end,:], dims=1))/(parameter_tuning_window))]

end

d(i,j,ξ_i,ξ_j) = 0
include("weights.jl")
SAA_realised_costs, _ = train_and_test_out_of_sample(length(extracted_data), windowing_weights)
SAA_average_cost = mean(SAA_realised_costs)


digits=4
function extract_results(parameters, weights; save_cost_plot_as = nothing)

    if save_cost_plot_as === nothing
    
        realised_costs, optimal_parameter = train_and_test_out_of_sample(parameters, weights)
    else
    
        realised_costs, optimal_parameter = train_and_test_out_of_sample(parameters, weights; save_cost_plot_as = save_cost_plot_as)
    end
    
    average_cost = round(mean(realised_costs), digits=digits)
    difference = round(average_cost - SAA_average_cost, digits=digits)
    difference_pairwise_se = round(sem(realised_costs - SAA_realised_costs), digits=digits)
    
    display("difference to SAA: $difference ± $difference_pairwise_se")
    
    return average_cost, difference, difference_pairwise_se, optimal_parameter
end

windowing_average_cost, windowing_difference, windowing_difference_pairwise_se, _ = 
    extract_results(windowing_parameters, windowing_weights)

smoothing_average_cost, smoothing_difference, smoothing_difference_pairwise_se, _ = 
    extract_results(smoothing_parameters, smoothing_weights)


d(i,j,ξ_i,ξ_j) = norm(ξ_i[1] - ξ_j[1], 1) + norm(ξ_i[2] - ξ_j[2], 1)
include("weights.jl")
WPF_L1_average_cost, WPF_L1_difference, WPF_L1_difference_pairwise_se, WPF_L1_parameter = 
    extract_results(WPF_parameters, WPF_weights; save_cost_plot_as = "figures/dairy-prices-WPF-L1-parameter-costs.pdf")

WPF_L1_sample_weights = WPF_weights([[extracted_data[i], extracted_data[i+1]] for i in 1:length(extracted_data)-1], WPF_L1_parameter)


default() # Reset plot defaults.


gr(size = (317+6+6,159+7).*sqrt(3))

font_family = "Computer Modern"
primary_font = Plots.font(font_family, pointsize = 12)
secondary_font = Plots.font(font_family, pointsize = 10)
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
        xminorticks = 3, 
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

sample_indices = 2:14*1*12
WPF_L1_parameter = round(Int,WPF_L1_parameter)
println("\$λ\$ = $WPF_L1_parameter")

plt_probabilities = plot(sample_indices[WPF_L1_sample_weights .>= 1e-3], 
                WPF_L1_sample_weights[WPF_L1_sample_weights .>= 1e-3],
                xlabel = "Time (year)",
        xticks = ([-5,3*12-5,6*12-5,9*12-5,12*12-5,14*12+6], ["2010","2013","2016","2019","2022","2025"]),
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
                rightmargin = 6pt,
                #bottommargin = 2.5pt, 
                #leftmargin = 2.5pt
                bottommargin = 7pt, 
                leftmargin = 6pt,
                )

yl = ylims(plt_probabilities)
ylims!((0,yl[2]))

#figure = plot(plt_extracted_data, plt_probabilities, layout=@layout([a; b]))
#display(figure)
#savefig(figure, "figures/dairy-prices-WPF-L1-assigned-probability-to-historical-observations.pdf")
display(plt_probabilities)
savefig(plt_probabilities, "figures/dairy-prices-WPF-L1-probability-assigned.pdf")



#=
d(i,j,ξ_i,ξ_j) = sqrt(norm(ξ_i[1] - ξ_j[1], 2)^2 + norm(ξ_i[2] - ξ_j[2], 2)^2)
include("weights.jl")
WPF_L2_average_cost, WPF_L2_difference, WPF_L2_difference_pairwise_se, _ = 
    extract_results(WPF_parameters, WPF_weights)

d(i,j,ξ_i,ξ_j) = max(norm(ξ_i[1] - ξ_j[1], Inf), norm(ξ_i[2] - ξ_j[2], Inf))
include("weights.jl")
WPF_LInfty_average_cost, WPF_LInfty_difference, WPF_LInfty_difference_pairwise_se, _ = 
    extract_results(WPF_parameters, WPF_weights)

SAA_average_cost = round(SAA_average_cost, digits=digits)
println("& \$$SAA_average_cost\$ & \$$windowing_average_cost\$ & \$$smoothing_average_cost\$ & \$$WPF_L1_average_cost\$ & \$$WPF_L2_average_cost\$ & \$$WPF_LInfty_average_cost\$")
println("& \$\$ & \\makecell{\$$windowing_difference\$\\\\\\small\$\\pm$windowing_difference_pairwise_se\$} & \\makecell{\$\\kern8.5167pt$smoothing_difference\$\\\\\\small{\$\\pm$smoothing_difference_pairwise_se\$}} & \\makecell{\$$WPF_L1_difference\$\\\\\\small{\$\\pm$WPF_L1_difference_pairwise_se\$}} & \\makecell{\$\\kern8.5167pt$WPF_L2_difference\$\\\\\\small{\$\\pm$WPF_L2_difference_pairwise_se\$}} & \\makecell{\$$WPF_LInfty_difference\$\\\\\\small{\$\\pm$WPF_LInfty_difference_pairwise_se\$}}")
=#

