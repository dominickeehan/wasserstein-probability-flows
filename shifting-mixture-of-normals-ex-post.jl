using Random, Statistics, StatsBase, Distributions

Cu = 4 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)
newsvendor_order(ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

Random.seed!(42)

using LinearAlgebra

N = 2

weight_shift_distribution = Normal(0, 0.0)
mean_shift_distribution = MvNormal(zeros(N), [300 0; 0 100]) # I
sd_shift_distribution = MvNormal(zeros(N), [1 0; 0 1])

repetitions = 1000
history_length = 10

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
demand_distributions = [[MixtureModel(Normal[Normal(0, 0) for _ in 1:N]) for _ in 1:history_length+1] for _ in 1:repetitions]


for repetition in 1:repetitions
    means = [1000, 2000] # [i*1000 for i in 1:N]
    sds = [100, 300] #100*ones(N)
    weight = 0.5 #rand(Uniform(0,1))

    for t in 1:history_length+1
        demand_distributions[repetition][t] = MixtureModel(Normal[Normal(means[i], sds[i]) for i in 1:N], [weight, 1-weight])
        #demand_sequences[repetition][t] = max(rand(demand_distributions[repetition][t]), 0)
        demand_sequences[repetition][t] = rand(demand_distributions[repetition][t])

        means = means + rand(mean_shift_distribution)
        sds = max.(sds + rand(sd_shift_distribution), zeros(N))
        weight = min(max(weight + rand(weight_shift_distribution), 0), 1)        
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

    display(plot(weight_parameters, mean(costs)))

    if true

        plt_1 = plot()
        for repetition in 1:repetitions
        
            ξ_range = LinRange(0,(N+1)*1000,1000)
            plot!(ξ_range, [pdf(demand_distributions[repetition][end-1], ξ) for ξ in ξ_range], labels = nothing, xlims = (0,(N+1)*1000), alpha = 0.1)
        end

        #display(plt)

        plt_2 = plot()
        for repetition in 1:repetitions

            demand_samples = demand_sequences[repetition][1:history_length]
            demand_sample_weights = solve_for_weights(demand_samples, weight_parameters[weight_parameter_index])

#            plot!(demand_samples, demand_sample_weights, seriestype = :sticks, labels = nothing, xlims = (0,3000))
            stephist!(demand_samples, weights=demand_sample_weights, normalize=:pdf, labels=nothing, bins=10, xlims=(0,(N+1)*1000), alpha = 0.1, fill=true, fillalpha=0.01)
        end

        display(plot(plt_1, plt_2, layout=@layout([a;b])))
    end

    return minimal_costs
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i[1] - ξ_j[1], 1) #ifelse(i == j, 0, norm(ξ_i[1] - ξ_j[1], 1)+0)
include("weights.jl")


SES_costs = parameter_fit(SES_weights, [LinRange(0.0001,0.001,10); LinRange(0.002,0.01,9); LinRange(0.02,0.2,19)])



#WPF_costs = parameter_fit(WPF_weights, [LinRange(.02,.1,5); LinRange(.2,1,5); LinRange(2,10,5); LinRange(20,100,5)])

#WPF_costs = parameter_fit(WPF_weights, [LinRange(.001,.03,30); LinRange(.04,0.1,7); LinRange(.2,1,9);])
WPF_costs = parameter_fit(WPF_weights, [LinRange(.01,.2,20); LinRange(.3,1,8)])
#WPF_costs = parameter_fit(WPF_weights, LinRange(.01,1,100))
#WPF_costs = parameter_fit(WPF_weights, 0.24)

display(sem(WPF_costs - SES_costs))

#display([parameter_fit(WPF_weights, 0.7)])







if true

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