using Random, Statistics, StatsBase, Distributions

Cu = 4 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)
newsvendor_order(ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

Random.seed!(42)

using LinearAlgebra


Q = 10

shift_distribution = Normal(0, 1)

repetitions = 1000
history_length = 30

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
demand_distributions = [[Gamma(1,1) for _ in 1:history_length+1] for _ in 1:repetitions]

for repetition in 1:repetitions
    shape = 30
    scale = 30

    for t in 1:history_length+1
        demand_distributions[repetition][t] = Gamma(shape, scale)
        demand_sequences[repetition][t] = rand(demand_distributions[repetition][t])

        shape = max.(shape + rand(shift_distribution), 0.01)     
    end
end

using Plots
using ProgressBars, IterTools
function parameter_fit(solve_for_weights, weight_parameters)

    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        local demand_samples = demand_sequences[repetition][1:history_length]
        local demand_sample_weights = solve_for_weights(demand_samples, weight_parameters[weight_parameter_index])
        local order = newsvendor_order(demand_samples, demand_sample_weights)
        costs[repetition][weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])

    end

    weight_parameter_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][weight_parameter_index] for repetition in 1:repetitions]

    digits = 4

    display([round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits), round(weight_parameters[weight_parameter_index], digits=digits)])

    display(plot(weight_parameters, mean(costs), xscale=:log10))
    #display(plot(weight_parameters, mean(costs)))

    if true

        plt_1 = plot()
        for repetition in 1:repetitions
        
            ξ_range = LinRange(0,2000,1000)
            plot!(ξ_range, [pdf(demand_distributions[repetition][end-1], ξ) for ξ in ξ_range], labels = nothing, xlims = (0,2000), alpha = 0.5)
        end

        #display(plt)

        plt_2 = plot()
        for repetition in 1:repetitions

            demand_samples = demand_sequences[repetition][1:history_length]
            demand_sample_weights = solve_for_weights(demand_samples, weight_parameters[weight_parameter_index])

            stephist!(demand_samples, weights=demand_sample_weights, normalize=:pdf, labels=nothing, xlims=(0,2000), bins = round(Int, history_length/10), alpha = 0.5)
        end

        display(plot(plt_1, plt_2, layout=@layout([a;b])))
    end

    return minimal_costs
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i[1] - ξ_j[1], 1) #ifelse(i == j, 0, norm(ξ_i[1] - ξ_j[1], 1)+0)
include("weights.jl")


parameter_fit(windowing_weights, history_length)
#parameter_fit(windowing_weights, round.(Int, LinRange(1,history_length,history_length)))
SES_costs = parameter_fit(SES_weights, [LinRange(0.00001,0.0001,10); LinRange(0.0001,0.001,9); LinRange(0.002,0.01,9); LinRange(0.02,1.0,99)])

#WPF_costs = parameter_fit(WPF_weights, [LinRange(.02,.1,5); LinRange(.2,1,5); LinRange(2,10,5); LinRange(20,100,5)])

#WPF_costs = parameter_fit(WPF_weights, [LinRange(.001,.03,30); LinRange(.04,0.1,7); LinRange(.2,1,9);])
#WPF_costs = parameter_fit(WPF_weights, [LinRange(.002,.01,9); LinRange(.02,.1,9); LinRange(.2,1,9);])
#WPF_costs = parameter_fit(WPF_weights, [LinRange(.01,.1,10); LinRange(.2,1,9); LinRange(2,10,9);])
#WPF_costs = parameter_fit(WPF_weights, LinRange(.1,1,Q))


#WPF_costs = parameter_fit(WPF_weights, LinRange(.001,.01,Q))
#WPF_costs = parameter_fit(WPF_weights, [LinRange(.001,.01,Q); LinRange(.01,.1,Q)])
#WPF_costs = parameter_fit(WPF_weights, [LinRange(.01,.1,Q); LinRange(.1,1,Q); LinRange(1,10,Q)])
#WPF_costs = parameter_fit(WPF_weights, LinRange(.1,1,Q))

#WPF_costs = parameter_fit(WPF_weights, [LinRange(.0000001,.000001,Q); LinRange(.000001,.00001,Q); LinRange(.00001,.0001,Q); LinRange(.0001,.001,Q); LinRange(.001,.01,Q); LinRange(.01,.1,Q); LinRange(.1,1,Q); LinRange(1,10,Q); LinRange(10,100,Q); LinRange(100,1000,Q);])

#WPF_costs = parameter_fit(WPF_weights, [LinRange(.001,.01,Q); LinRange(.01,.1,Q); LinRange(.1,1,Q); LinRange(1,10,Q); LinRange(10,100,Q)])

#WPF_costs = parameter_fit(WPF_weights, [LinRange(.01,.1,Q); LinRange(.1,1,Q); LinRange(1,10,Q)])
#WPF_costs = parameter_fit(WPF_weights, [LinRange(.02,.1,Q); LinRange(.2,1,Q); LinRange(2,10,Q);])
#WPF_costs = parameter_fit(WPF_weights, LinRange(.1,1,Q))
#WPF_costs = parameter_fit(WPF_weights, LinRange(1,10,Q))


#WPF_costs = parameter_fit(WPF_weights, [LinRange(.001,.01,Q); LinRange(.01,.1,Q); LinRange(.1,1,Q); LinRange(1,10,Q); LinRange(10,100,Q)])
WPF_costs = parameter_fit(WPF_weights, LinRange(0.01,0.1,30))


#WPF_costs = parameter_fit(WPF_weights, [LinRange(.01,.1,Q); LinRange(.1,1,Q)])
#WPF_costs = parameter_fit(WPF_weights, LinRange(.01,.1,Q))

#WPF_costs = parameter_fit(WPF_weights, [LinRange(.01,.2,20); LinRange(.3,1,8); 10])
#WPF_costs = parameter_fit(WPF_weights, LinRange(.01,1,100))
#WPF_costs = parameter_fit(WPF_weights, 0.24)

display(sem(WPF_costs - SES_costs))

#display([parameter_fit(WPF_weights, 0.7)])







if false

    using Plots, Measures

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

    plt = plot(1:history_length, 
            stack(demand_sequences[2:end])[1:end-1,:], 
            xlims = (0,history_length+1),
            xlabel = "Time", 
            ylabel = "Demand",
            labels = nothing, 
            #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
            #markercolor = palette(:tab10)[1],
            #markershape = :circle,
            color = palette(:tab10)[1],
            alpha = 0.01,
            #linestyle = :auto,
            #markersize = 4, 
            #markerstrokewidth = 1,
            #markerstrokecolor = :black,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt,
            )

    plot!(1:history_length, 
            stack(demand_sequences[1])[1:end-1,:], 
            labels = nothing, 
            #linecolor = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]],
            markercolor = palette(:tab10)[1],
            markershape = :circle,
            color = palette(:tab10)[1],
            alpha = 1.0,
            #linestyle = :auto,
            markersize = 4, 
            markerstrokewidth = 1,
            markerstrokecolor = :black,
            )

    display(plt)

    #savefig(plt, "figures/demand_sequences.pdf")

end