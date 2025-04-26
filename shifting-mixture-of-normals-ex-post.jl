using Random, Statistics, StatsBase, Distributions

Cu = 3 # Cost of underage.
Co = 1 # Cost of overage.

newsvendor_loss(x,ξ) = Cu*max(ξ-x,0) + Co*max(x-ξ,0)
newsvendor_order(ξ, weights) = quantile(ξ, Weights(weights), Cu/(Co+Cu))

Random.seed!(42)

weight_shift_distribution = Normal(0,0.0)
mean_shift_distribution = MvNormal([0, 0], [1000 0.1; 0.1 0.1])
sd_shift_distribution = MvNormal([0, 0], [100 0.1; 0.1 0.1])

repetitions = 1000
history_length = 100

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
for repetition in 1:repetitions
    means = [1000, 2000] #[rand(Uniform(500,1500)), rand(Uniform(1000,3000))]
    sds = [100, 141] #[rand(Uniform(0,100)), rand(Uniform(0,141))]
    weight = 0.9 #rand(Uniform(0,1))

    for t in 1:history_length+1
        demand_sequences[repetition][t] = max(rand(MixtureModel(Normal[
                                                                   Normal(means[1], sds[1]),
                                                                   Normal(means[2], sds[2])], [weight, 1-weight])), 0)

        means = means + rand(mean_shift_distribution)
        sds = max.(sds + rand(sd_shift_distribution), [0, 0])
        weight = min(max(weight + rand(weight_shift_distribution), 0), 1)        
    end
end


using ProgressBars, IterTools
function parameter_fit(solve_for_weights, weight_parameters)

    costs = [zeros(length(weight_parameters)) for _ in 1:repetitions]

    Threads.@threads for (weight_parameter_index, repetition) in ProgressBar(collect(IterTools.product(eachindex(weight_parameters), 1:repetitions)))
        demand_samples = demand_sequences[repetition][1:history_length]
        demand_sample_weights = solve_for_weights(demand_samples, weight_parameters[weight_parameter_index])
        order = newsvendor_order(demand_samples, demand_sample_weights)
        costs[repetition][weight_parameter_index] = newsvendor_loss(order, demand_sequences[repetition][history_length+1])
    end

    weight_parameter_index = argmin(mean(costs))
    minimal_costs = [costs[repetition][weight_parameter_index] for repetition in 1:repetitions]

    println(mean(costs))

    digits = 4

    return round(mean(minimal_costs), digits=digits), round(sem(minimal_costs), digits=digits), round(weight_parameters[weight_parameter_index], digits=digits)
end

using LinearAlgebra
d(i,j,ξ_i,ξ_j) = norm(ξ_i[1] - ξ_j[1], 1) #ifelse(i == j, 0, norm(ξ_i[1] - ξ_j[1], 1)+0)
include("weights.jl")

display([parameter_fit(windowing_weights, history_length)])

display([parameter_fit(windowing_weights, 1)])

display([parameter_fit(SES_weights, LinRange(0.01,0.3,30))])

display([parameter_fit(WPF_weights, LinRange(0.01,0.1,3))])
#display([parameter_fit(WPF_weights, 0.04)])







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
            alpha = 0.03,
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