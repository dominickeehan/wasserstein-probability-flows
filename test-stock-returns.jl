include("extract-stock-returns.jl")

using JuMP, LinearAlgebra, COPT, HiGHS

portfolio_optimizer = optimizer_with_attributes(COPT.Optimizer, "Logging" => 0, "LogToConsole" => 0,)
#portfolio_optimizer = optimizer_with_attributes(HiGHS.Optimizer, "log_to_console" => false,)

#using Gurobi
#env = Gurobi.Env()  
#portfolio_optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0)

ρ = 0.9 # Risk aversion parameter.
α = 0.05 # CVaR (dis)-confidence level (in (0, 1]). 1 = Expectation.

function solve_risk_averse_portfolio(sample_returns, sample_weights)
    """
    Solve the risk-averse portfolio optimization problem.

    Inputs:
    - 
    - ρ: Risk aversion parameter (non-negative scalar).
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
        - (1-ρ)*sum(sample_weights[i]*dot(x, sample_returns[i]) for i in 1:N) + ρ * τ + ρ * sum(sample_weights[i]*(1/α)*z[i] for i in 1:N)
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

parameter_tuning_window = 1*12

windowing_parameters = round.(Int, LinRange(1,training_testing_split-parameter_tuning_window,51))
SES_parameters = LinRange(0.001,0.9,51)
WPF_parameters = LinRange(0,750,51)

using ProgressBars, IterTools
using Statistics, StatsBase
using Plots, Measures
function train_and_test_out_of_sample(parameters, solve_for_weights; save_cost_plot_as = nothing)

    parameter_costs_in_training_stages = zeros((training_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(training_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data[1:t-1]]
        local sample_weights = solve_for_weights(samples, parameters[i])
        local x = solve_risk_averse_portfolio(samples, sample_weights)
        parameter_costs_in_training_stages[t,i] = -portfolio_return(x, training_data[t])
    end

    parameter_costs_in_testing_stages = zeros((testing_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(testing_T:-1:1, eachindex(parameters))))  
        local samples = [warm_up_data; training_data; testing_data[1:t-1]]
        local sample_weights = solve_for_weights(samples, parameters[i])
        local x = solve_risk_averse_portfolio(samples, sample_weights)
        parameter_costs_in_testing_stages[t,i] = -portfolio_return(x, testing_data[t])
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]

    average_parameter_costs_in_previous_stages = [zeros(length(parameters)) for _ in 1:testing_T]
    for t in 1:testing_T
        average_parameter_costs_in_previous_stages[t] = (1-ρ)*vec(mean(parameter_costs[training_T+(t-1)-(parameter_tuning_window-1):training_T+(t-1),:], dims=1)) + ρ*[unweighted_cvar(parameter_costs[training_T+(t-1)-(parameter_tuning_window-1):training_T+(t-1),i]) for i in eachindex(parameters)]
    end

    realised_costs = [parameter_costs_in_testing_stages[t,argmin(average_parameter_costs_in_previous_stages[t])] for t in 1:testing_T] 
    μ = (1-ρ)*mean(realised_costs) + ρ*unweighted_cvar(realised_costs)
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
SAA_costs, _ = train_and_test_out_of_sample(length(extracted_data), windowing_weights)

#=
windowing_costs, _ = train_and_test_out_of_sample(windowing_parameters, windowing_weights)
μ = mean(windowing_costs) + ρ*unweighted_cvar(windowing_costs) - mean(SAA_costs) - ρ*unweighted_cvar(SAA_costs)
s = sem(windowing_costs - SAA_costs)
display("Windowing - SAA: $μ ± $s")

SES_costs, _ = train_and_test_out_of_sample(SES_parameters, SES_weights) # ceil(Int,0.7*training_testing_split)-1

μ = mean(SES_costs) + ρ*unweighted_cvar(SES_costs) - mean(SAA_costs) - ρ*unweighted_cvar(SAA_costs)
s = sem(SES_costs - SAA_costs)
display("SES - SAA: $μ ± $s")
=#

#=
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights)

μ = mean(WPF_costs) + ρ*unweighted_cvar(WPF_costs) - mean(SAA_costs) - ρ*unweighted_cvar(SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")
=#


d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1) #ifelse(i == j, 0, norm(ξ_i - ξ_j, 1) + 0.001) # 0.001
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights; save_cost_plot_as = "figures/stock-returns-WPF_{1+s}-parameter-costs.pdf")

μ = (1-ρ)*mean(WPF_costs) + ρ*unweighted_cvar(WPF_costs) - (1-ρ)*mean(SAA_costs) - ρ*unweighted_cvar(SAA_costs)
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

colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5] palette(:tab10)[6] palette(:tab10)[7] palette(:tab10)[8] palette(:tab10)[9] palette(:tab10)[10] palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4]]
linestyles = [:solid :solid :solid :solid :solid :solid :solid :solid :solid :solid :dash :dash :dash :dash]

plt_extracted_data = plot(1:10*12+1, 
                        wealth, 
                        xformatter = :none,
                        #xlims = (1-6,10*12+6),
                        ylims = (0,11.5),
                        ylabel = "Net wealth",
                        labels = nothing,
                        legend = nothing,
                        #legendfonthalign = :center,
                        color = colors,
                        linestyle = linestyles,
                        linewidth = 1,
                        topmargin = 0pt, 
                        rightmargin = 0pt,
                        bottommargin = 0pt, 
                        leftmargin = 5pt) 

A = 2:10*12
WPF_parameter = round(Int,WPF_parameter)
println("\$λ\$ = $WPF_parameter")

plt_probabilities = plot(A[weights .>= 1e-3], 
                weights[weights .>= 1e-3],
                xlabel = "Time (year)",
                xticks = (1:12:10*12+1, ["2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]),
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
                leftmargin = 0pt)

figure = plot(plt_extracted_data, plt_probabilities, layout=@layout([a; b]))
display(figure)
savefig(figure, "figures/stock-returns-WPF_{1+s}-assigned-probability-to-historical-observations.pdf")

#=
d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, 2)
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights)

μ = mean(WPF_costs) + ρ*unweighted_cvar(WPF_costs) - mean(SAA_costs) - ρ*unweighted_cvar(SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")


d(i,j,ξ_i,ξ_j) = norm(ξ_i - ξ_j, Inf)
include("weights.jl")
WPF_costs, WPF_parameter = train_and_test_out_of_sample(WPF_parameters, WPF_weights)

μ = mean(WPF_costs) + ρ*unweighted_cvar(WPF_costs) - mean(SAA_costs) - ρ*unweighted_cvar(SAA_costs)
s = sem(WPF_costs - SAA_costs)
display("WPF - SAA: $μ ± $s")
=#
