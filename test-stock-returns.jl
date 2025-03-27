include("extract-stock-returns.jl")

using JuMP, LinearAlgebra, COPT, HiGHS

portfolio_optimizer = optimizer_with_attributes(COPT.Optimizer, "Logging" => 0, "LogToConsole" => 0,)
#portfolio_optimizer = optimizer_with_attributes(HiGHS.Optimizer, "log_to_console" => false,)

#using Gurobi
#env = Gurobi.Env()  
#portfolio_optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0)

ρ = 0.1 # 1 - risk aversion parameter.
α = 0.05 # CVaR (dis)-confidence level (in (0, 1]). 1 = Expectation.

function solve_risk_averse_portfolio(sample_returns, sample_weights)
    """
    Solve the risk-averse portfolio optimization problem.

    Inputs:
    - 
    - (1-ρ): Risk aversion parameter (non-negative scalar).
    - α: CVaR confidence level (in (0, 1]).
    Outputs:
    - x: Optimal portfolio weights (vector of size m).
    """

    N = length(sample_returns) # Number of return samples.
    m = length(sample_returns[1]) # Number of assets.

    # Create the model
    model = Model(portfolio_optimizer)  

    # Decision variables
    @variables(model, begin
                            x[i=1:m] >= 0 # Portfolio weights (non-negative)     
                            τ             # CVaR threshold
                            z[i=1:N] >= 0 # Slack variables for CVaR
                      end)

    # Constraints
    @constraint(model, sum(x) == 1)     # Portfolio weights sum to 1

    # Objective function
    @objective(model, Min,
        - ρ*sum(sample_weights[i]*dot(x, sample_returns[i]) for i in 1:N) + (1-ρ) * τ + (1-ρ) * sum(sample_weights[i]*(1/α)*z[i] for i in 1:N)
    )

    

    # CVaR constraints
    for i in 1:N
        @constraint(model, z[i] >= -dot(x, sample_returns[i]) - τ)
        #@constraint(model, z[i] >= 0)
    end

    # Solve the model
    optimize!(model)

    return value.(x)
end

function unweighted_cvar(costs)

    N = length(costs)

    # Create the model
    model = Model(portfolio_optimizer)  

    # Decision variables
    @variables(model, begin  
                            τ             # CVaR threshold
                            z[i=1:N] >= 0 # Slack variables for CVaR
                      end)

    # Objective function
    @objective(model, Min,
        τ + sum((1/N)*(1/α)*z[i] for i in 1:N)
    )

    # CVaR constraints
    for i in 1:N
        @constraint(model, z[i] >= costs[i] - τ)
    end

    # Solve the model
    optimize!(model)

    return objective_value(model)
end

portfolio_return(portfolio, realised_return) = dot(portfolio, realised_return)

training_testing_split = ceil(Int,0.7*length(extracted_data))
warm_up_period = ceil(Int,0.5*training_testing_split)-1 # Needs to be small enough to allow parameter_tuning_window after.
warm_up_data = extracted_data[1:warm_up_period]
training_data = extracted_data[warm_up_period+1:training_testing_split]
training_T = length(training_data)
testing_data = extracted_data[training_testing_split+1:end]
testing_T = length(testing_data)

parameter_tuning_window = 3*12

windowing_parameters = round.(Int, LinRange(1,length(extracted_data),11))
SES_parameters = LinRange(0.0001,0.2,11)
WPF_parameters = LinRange(0,500,11)

using ProgressBars, IterTools
using Statistics, StatsBase
using Plots, Measures
function train_and_test_out_of_sample(parameters, weights; save_cost_plot_as = nothing)

    parameter_costs_in_training_stages = zeros((training_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(training_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data[1:t-1]]
        local sample_weights = weights(samples, parameters[i])
        local x = solve_risk_averse_portfolio(samples, sample_weights)
        parameter_costs_in_training_stages[t,i] = -portfolio_return(x, training_data[t])
    end

    parameter_costs_in_testing_stages = zeros((testing_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(testing_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data; testing_data[1:t-1]]
        local sample_weights = weights(samples, parameters[i])
        local x = solve_risk_averse_portfolio(samples, sample_weights)
        parameter_costs_in_testing_stages[t,i] = -portfolio_return(x, testing_data[t])
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]

    average_parameter_costs_in_previous_stages = [zeros(length(parameters)) for _ in 1:testing_T]
    for t in 1:testing_T
        average_parameter_costs_in_previous_stages[t] = ρ*vec(mean(parameter_costs[training_T+(t-1)-(parameter_tuning_window-1):training_T+(t-1),:], dims=1)) + (1-ρ)*[unweighted_cvar(parameter_costs[training_T+(t-1)-(parameter_tuning_window-1):training_T+(t-1),i]) for i in eachindex(parameters)]
    end

    realised_costs = [parameter_costs_in_testing_stages[t,argmin(average_parameter_costs_in_previous_stages[t])] for t in 1:testing_T] 
    μ = ρ*mean(realised_costs) + (1-ρ)*unweighted_cvar(realised_costs)
    s = sem(realised_costs)

    wealth = 1
    for i in eachindex(realised_costs); wealth *= 1-realised_costs[i]; end
    display("Cost: $μ ± $s, Wealth: $wealth")
    
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
            average_parameter_costs_in_previous_stages[end], #vec(sum(parameter_costs[end-(parameter_tuning_window-1):end,:], dims=1))/(parameter_tuning_window),
            ribbon = sem.([parameter_costs[end-(parameter_tuning_window-1):end,parameter_index] for parameter_index in eachindex(parameters)]),
            xlabel = "\$λ\$", 
            ylabel = "Risk-adjusted expected cost",
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

    return realised_costs, parameters[argmin(average_parameter_costs_in_previous_stages[end])]

end

d(i,j,ξ_i,ξ_j) = 0
include("weights.jl")
SAA_realised_costs, _ = train_and_test_out_of_sample(length(extracted_data), windowing_weights)
SAA_risk_adjusted_expected_cost = ρ*mean(SAA_realised_costs) + (1-ρ)*unweighted_cvar(SAA_realised_costs)

digits=6
function extract_results(parameters, weights; save_cost_plot_as = nothing)
    if save_cost_plot_as === nothing
        realised_costs, optimal_parameter = train_and_test_out_of_sample(parameters, weights)
    else
        realised_costs, optimal_parameter = train_and_test_out_of_sample(parameters, weights; save_cost_plot_as = save_cost_plot_as)
    end
    risk_adjusted_expected_cost = round(ρ*mean(realised_costs) + (1-ρ)*unweighted_cvar(realised_costs), digits=digits-1)
    difference = round(risk_adjusted_expected_cost - SAA_risk_adjusted_expected_cost, digits=digits)
    difference_pairwise_se = round(sem(realised_costs - SAA_realised_costs), digits=digits)
    display("difference to SAA: $difference ± $difference_pairwise_se")
    
    return risk_adjusted_expected_cost, difference, difference_pairwise_se, optimal_parameter
end


windowing_risk_adjusted_expected_cost, windowing_difference, windowing_difference_pairwise_se, _ = 
    extract_results(windowing_parameters, windowing_weights)

SES_risk_adjusted_expected_cost, SES_difference, SES_difference_pairwise_se, _ = 
    extract_results(SES_parameters, SES_weights)


d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
include("weights.jl")
WPF1_risk_adjusted_expected_cost, WPF1_difference, WPF1_difference_pairwise_se, _ = 
    extract_results(WPF_parameters, WPF_weights)



d(i,j,ξ_i,ξ_j) = ifelse(i == j, 0, 1.0*norm(ξ_i - ξ_j, 1)+0.0001)
include("weights.jl")
WPF1s_risk_adjusted_expected_cost, WPF1s_difference, WPF1s_difference_pairwise_se, WPF1s_parameter = 
    extract_results(WPF_parameters, WPF_weights; save_cost_plot_as = "figures/stock-returns-WPF1s-parameter-costs.pdf")

WPF1s_sample_weights = WPF_weights(extracted_data, WPF1s_parameter)

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

colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5] palette(:tab10)[6] palette(:tab10)[7] palette(:tab10)[8] palette(:tab10)[9] palette(:tab10)[10] palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4]]
linestyles = [:solid :solid :solid :solid :solid :solid :solid :solid :solid :solid :dash :dash :dash :dash]

plt_extracted_data = plot(1:10*12, 
                        100*stack(extracted_data)'[:,[1,2,11]], 
                        xformatter = :none,
                        #xlims = (1-6,10*12+6),
                        xlims = (1-4,10*12+4),
                        #ylims = (0,11.5),
                        ylabel = "Return (%)",
                        labels = nothing,
                        legend = nothing,
                        linetype = :stepmid,
                        #legendfonthalign = :center,
                        color = permutedims(colors[[1,2,11]]),
                        linestyle = permutedims(linestyles[[1,2,11]]),
                        linewidth = 1,
                        topmargin = 0pt, 
                        rightmargin = 0pt,
                        bottommargin = 0pt, 
                        leftmargin = 5pt) 

sample_indices = 1:10*12
WPF1s_parameter = round(Int,WPF1s_parameter)
println("\$λ\$ = $WPF_parameter")

plt_probabilities = plot(sample_indices[WPF1s_sample_weights .>= 1e-3], 
                WPF1s_sample_weights[WPF1s_sample_weights .>= 1e-3],
                xlabel = "Time (year)",
                xticks = (1:12:10*12, ["2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]),
                xlims = (1-4,10*12+4),
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
                leftmargin = 2.5pt)

figure = plot(plt_extracted_data, plt_probabilities, layout=@layout([a; b]))
display(figure)
savefig(figure, "figures/stock-returns-WPF1s-assigned-probability-to-historical-observations.pdf")




d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 2)
include("weights.jl")
WPF2_risk_adjusted_expected_cost, WPF2_difference, WPF2_difference_pairwise_se, _ = 
    extract_results(WPF_parameters, WPF_weights)

d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, Inf)
include("weights.jl")
WPFInfty_risk_adjusted_expected_cost, WPFInfty_difference, WPFInfty_difference_pairwise_se, _ = 
    extract_results(WPF_parameters, WPF_weights)

SAA_risk_adjusted_expected_cost = round(SAA_risk_adjusted_expected_cost, digits=digits)
println("& \$$SAA_risk_adjusted_expected_cost\$ & \$$windowing_risk_adjusted_expected_cost\$ & \$$SES_risk_adjusted_expected_cost\$ & \$$WPF1_risk_adjusted_expected_cost\$ & \$$WPF1s_risk_adjusted_expected_cost\$ & \$$WPF2_risk_adjusted_expected_cost\$ & \$$WPFInfty_risk_adjusted_expected_cost\$")
println("& \$\$ & \\makecell{\$\\kern8.5167pt $windowing_difference\$\\\\\\small\$\\pm$windowing_difference_pairwise_se\$} & \\makecell{\$\\kern8.5167pt$SES_difference\$\\\\\\small{\$\\pm$SES_difference_pairwise_se\$}} & \\makecell{\$\\kern8.5167pt$WPF1_difference\$\\\\\\small{\$\\pm$WPF1_difference_pairwise_se\$}} & \\makecell{\$$WPF1s_difference\$\\\\\\small{\$\\pm$WPF1s_difference_pairwise_se\$}} & \\makecell{\$$WPF2_difference\$\\\\\\small{\$\\pm$WPF2_difference_pairwise_se\$}} & \\makecell{\$$WPFInfty_difference\$\\\\\\small{\$\\pm$WPFInfty_difference_pairwise_se\$}}")

