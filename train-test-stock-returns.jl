using JuMP, LinearAlgebra, COPT
using Statistics, StatsBase
using ProgressBars, IterTools
using Plots, Measures

include("extract-stock-returns.jl")
include("weights.jl")

portfolio_optimizer = optimizer_with_attributes(COPT.Optimizer, "Logging" => 0, "LogToConsole" => 0,)

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

    model = Model(portfolio_optimizer)  

    @variables(model, begin
                            x[i=1:m] >= 0 # Portfolio weights (non-negative).
                            τ             # CVaR threshold.
                            z[i=1:N] >= 0 # Slack variables for CVaR.
                      end)

    @constraint(model, sum(x) == 1) # Portfolio weights sum to 1

    @objective(model, Min,
        - ρ*sum(sample_weights[i]*dot(x, sample_returns[i]) for i in 1:N) + (1-ρ) * τ + (1-ρ) * sum(sample_weights[i]*(1/α)*z[i] for i in 1:N)
    )

    for i in 1:N; @constraint(model, z[i] >= -dot(x, sample_returns[i]) - τ); end # CVaR constraints.

    optimize!(model)

    return value.(x)
end

function unweighted_cvar(costs)

    N = length(costs) # Number of cost samples.

    model = Model(portfolio_optimizer)  

    @variables(model, begin  
                            τ             # CVaR threshold.
                            z[i=1:N] >= 0 # Slack variables for CVaR.
                      end)

    @objective(model, Min, τ + sum((1/N)*(1/α)*z[i] for i in 1:N))

    for i in 1:N; @constraint(model, z[i] >= costs[i] - τ); end # CVaR constraints.

    optimize!(model)

    return objective_value(model)
end

portfolio_return(portfolio, realised_return) = dot(portfolio, realised_return)


training_testing_split = ceil(Int,0.7*length(extracted_data))

warm_up_period = ceil(Int,0.7*training_testing_split)-1 # Needs to be small enough to allow parameter_tuning_window after.
warm_up_data = extracted_data[1:warm_up_period]

training_data = extracted_data[warm_up_period+1:training_testing_split]
training_T = length(training_data)

testing_data = extracted_data[training_testing_split+1:end]
testing_T = length(testing_data)

parameter_tuning_window = 2*12

LogRange(start, stop, len) = exp.(LinRange(log(start), log(stop), len))

windowing_parameters = unique(ceil.(Int, LogRange(1,length(extracted_data),30)))
smoothing_parameters = [0; LogRange(1e-4,1e0,30)]
WPF_parameters = [0; LinRange(1,10,10); LinRange(20,100,9); LinRange(200,1000,9); Inf] 


function train_and_test_out_of_sample(parameters, weights, d; save_cost_plot_as = nothing)

    parameter_costs_in_training_stages = zeros((training_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(training_T:-1:1, eachindex(parameters))))

        local samples = [warm_up_data; training_data[1:t-1]]
        local sample_weights = weights(samples, parameters[i], d)
        local x = solve_risk_averse_portfolio(samples, sample_weights)

        parameter_costs_in_training_stages[t,i] = -portfolio_return(x, training_data[t])
    end

    parameter_costs_in_testing_stages = zeros((testing_T,length(parameters)))
    Threads.@threads for (t,i) in ProgressBar(collect(IterTools.product(testing_T:-1:1, eachindex(parameters))))  

        local samples = [warm_up_data; training_data; testing_data[1:t-1]]
        local sample_weights = weights(samples, parameters[i], d)
        local x = solve_risk_averse_portfolio(samples, sample_weights)

        parameter_costs_in_testing_stages[t,i] = -portfolio_return(x, testing_data[t])
    end

    parameter_costs = [parameter_costs_in_training_stages; parameter_costs_in_testing_stages]

    average_parameter_costs_in_previous_stages = [zeros(length(parameters)) for _ in 1:testing_T]
    for t in 1:testing_T

        average_parameter_costs_in_previous_stages[t] = 
            ρ*vec(mean(parameter_costs[training_T+(t-1)-(parameter_tuning_window-1):training_T+(t-1),:], dims=1)) + 
                (1-ρ)*[unweighted_cvar(parameter_costs[training_T+(t-1)-(parameter_tuning_window-1):training_T+(t-1),i]) for i in eachindex(parameters)]
    end

    realised_costs = [parameter_costs_in_testing_stages[t,argmin(average_parameter_costs_in_previous_stages[t])] for t in 1:testing_T]
    μ = ρ*mean(realised_costs) + (1-ρ)*unweighted_cvar(realised_costs)
    s = sem(realised_costs)

    wealth = 1; for i in eachindex(realised_costs); wealth *= 1-realised_costs[i]; end

    display("Cost: $μ ± $s, Wealth: $wealth")

    
    default() # Reset plot defaults.

    gr(size = (275+6,183+6+6).*sqrt(3))

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

    if weights == WPF_weights

        plt = plot([1e-1; parameters[2:end-1]; 1e4], 
                average_parameter_costs_in_previous_stages[end],
                ribbon = sem.([parameter_costs[end-(parameter_tuning_window-1):end,parameter_index] for parameter_index in eachindex(parameters)]),
                xscale = :log10,
                xticks = [1, 10, 100, 1000],
                xlabel = "Shift penalty, \$λ\$", 
                ylabel = "Risk-adjusted average training cost",
                legend = nothing,
                legendfonthalign = :center,
                color = :black,#palette(:tab10)[1],
                alpha = 1,
                linestyle = :solid,
                linewidth = 1,
                fillalpha = .1,
                topmargin = 6pt, 
                rightmargin = 0pt,
                bottommargin = 6pt, 
                leftmargin = 6pt)
        
        xlims!((parameters[2]-2e-1, parameters[end-1]+0.25e3))

        display(plt);
    end

    if !(save_cost_plot_as === nothing); savefig(plt, save_cost_plot_as); end

    return realised_costs, parameters[argmin(average_parameter_costs_in_previous_stages[end])]

end

d(ξ_i,ξ_j) = 0

SAA_realised_costs, _ = train_and_test_out_of_sample(length(extracted_data), windowing_weights, d)
SAA_risk_adjusted_average_cost = ρ*mean(SAA_realised_costs) + (1-ρ)*unweighted_cvar(SAA_realised_costs)


digits=4
function extract_results(parameters, weights, d; save_cost_plot_as = nothing)

    if save_cost_plot_as === nothing

        realised_costs, optimal_parameter = train_and_test_out_of_sample(parameters, weights, d)
    else

        realised_costs, optimal_parameter = train_and_test_out_of_sample(parameters, weights, d; save_cost_plot_as = save_cost_plot_as)
    end

    risk_adjusted_average_cost = ρ*mean(realised_costs) + (1-ρ)*unweighted_cvar(realised_costs)

    percentage_average_difference = 
        round((risk_adjusted_average_cost - SAA_risk_adjusted_average_cost) / SAA_risk_adjusted_average_cost * 100, digits=1)
    percentage_sem_difference = round(sem(realised_costs - SAA_realised_costs) / SAA_risk_adjusted_average_cost * 100, digits=1)

    display("difference to SAA: $percentage_average_difference ± $percentage_sem_difference")
    
    return round(risk_adjusted_average_cost, digits=digits), percentage_average_difference, percentage_sem_difference, optimal_parameter
end

windowing_risk_adjusted_average_cost, windowing_percentage_average_difference, windowing_percentage_sem_difference, _ = 
    extract_results(windowing_parameters, windowing_weights, d)

smoothing_risk_adjusted_average_cost, smoothing_percentage_average_difference, smoothing_percentage_sem_difference, _ = 
    extract_results(smoothing_parameters, smoothing_weights, d)

d(ξ_i,ξ_j) = norm(ξ_i - ξ_j, 1)

WPF_L1_risk_adjusted_average_cost, WPF_L1_percentage_average_difference, WPF_L1_percentage_sem_difference, WPF_L1_parameter = 
    extract_results(WPF_parameters, WPF_weights, d; save_cost_plot_as = "figures/stock-returns-WPF-L1-parameter-costs.pdf")

WPF_L1_sample_weights = WPF_weights(extracted_data, WPF_L1_parameter, d)


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
        xminorticks = 2, 
        yminorticks = 0,
        fontfamily = font_family,
        guidefont = primary_font,
        tickfont = secondary_font,
        legendfont = legend_font)

colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5] palette(:tab10)[6] palette(:tab10)[7] palette(:tab10)[8] palette(:tab10)[9] palette(:tab10)[10]]
linestyles = [:solid :solid :solid :solid :solid :solid :solid :solid :solid :solid]

plt_extracted_data = plot(1:10*12, 
                        100*stack(extracted_data)', 
                        xformatter = :none,
                        xticks = (1:12:11*12),
                        xlims = (1-4,10*12+4),
                        #ylims = (0,11.5),
                        ylabel = "Return (%)",
                        labels = nothing,
                        legend = nothing,
                        #markershape = :circle,
                        #markersize = 2,
                        #markeroutlinecolor = :black,
                        #markeroutlinewidth = 1pt,
                        linetype = :stepmid,
                        #legendfonthalign = :center,
                        color = colors,
                        #linestyle = palette(:tab10)[8], #permutedims(linestyles[:]),
                        #alpha = 0.2,
                        linewidth = 1,
                        topmargin = 0pt, 
                        rightmargin = 0pt,
                        bottommargin = 0pt, 
                        leftmargin = 5pt)

sample_indices = 1:10*12
WPF_L1_parameter = round(Int,WPF_L1_parameter)
println("\$λ\$ = $WPF_L1_parameter")

plt_probabilities = plot(sample_indices[WPF_L1_sample_weights .>= 1e-3], 
                WPF_L1_sample_weights[WPF_L1_sample_weights .>= 1e-3],
                xlabel = "Time (year)",
            xticks = (1:2*12:10*12+1, ["2014","2016","2018","2020","2022","2024"]),
            xlims = (1-6,10*12+6+1),
                ylabel = "Probability mass", # at \$λ=$WPF_parameter\$",
                seriestype=:sticks,
                linestyle=:solid,
                linewidth = 1,
                seriescolor = :black,#palette(:tab10)[8],
                marker = nothing,
                markersize = 2.0,
                markercolor = :black,#palette(:tab10)[8],
                markerstrokecolor = :black,
                markerstrokewidth = 0.5,
                label = nothing,
                topmargin = 0pt, 
                rightmargin = 0pt,
                #bottommargin = 2.5pt, 
                #leftmargin = 2.5pt
                bottommargin = 7pt, 
                leftmargin = 6pt,
                )

yl = ylims(plt_probabilities)
ylims!((0,yl[2]))

#figure = plot(plt_extracted_data, plt_probabilities, layout=@layout([a; b]))
#display(figure)
#savefig(figure, "figures/stock-returns-WPF-L1-assigned-probability-to-historical-observations.pdf")
display(plt_probabilities)
savefig(plt_probabilities, "figures/stock-returns-WPF-L1-probability-assigned.pdf")

5

#=
d(ξ_i,ξ_j) = norm(ξ_i - ξ_j, 2)
WPF_L2_risk_adjusted_average_cost, WPF_L2_percentage_average_difference, WPF_L2_percentage_sem_difference, _ = 
    extract_results(WPF_parameters, WPF_weights, d)

d(ξ_i,ξ_j) = norm(ξ_i - ξ_j, Inf)
WPF_LInf_risk_adjusted_average_cost, WPF_LInf_percentage_average_difference, WPF_LInf_percentage_sem_difference, _ = 
    extract_results(WPF_parameters, WPF_weights, d)

SAA_risk_adjusted_average_cost = round(SAA_risk_adjusted_average_cost, digits=digits)

println("\\makecell[r]{Risk-adjusted\\\\average testing cost} 
        & \$\\textcolor{white}{+}$SAA_risk_adjusted_average_cost\$ 
        & \$\\textcolor{white}{+}$windowing_risk_adjusted_average_cost\$ 
        & \$\\textcolor{white}{+}$smoothing_risk_adjusted_average_cost\$ 
        & \$\\textcolor{white}{+}$WPF_L1_risk_adjusted_average_cost\$ 
        & \$\\textcolor{white}{+}$WPF_L2_risk_adjusted_average_cost\$ 
        & \$\\textcolor{white}{+}$WPF_LInf_risk_adjusted_average_cost\$ \\\\")
println("\\addlinespace")
println("\\midrule")
println("\\addlinespace")
println("\\makecell[r]{Difference\\\\from SAA (\\%)} 
        & \$ \$ 
        & \$\\textcolor{white}{+}$windowing_percentage_average_difference\\pm $windowing_percentage_sem_difference\$ 
        & \$\\textcolor{white}{+}$smoothing_percentage_average_difference\\pm $smoothing_percentage_sem_difference\$
        & \$$WPF_L1_percentage_average_difference\\pm $WPF_L1_percentage_sem_difference\$
        & \$\\textcolor{white}{+}$WPF_L2_percentage_average_difference\\pm $WPF_L2_percentage_sem_difference\$
        & \$$WPF_LInf_percentage_average_difference\\pm $WPF_LInf_percentage_sem_difference\$ \\\\")


=#