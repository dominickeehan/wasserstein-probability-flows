using  Random, Statistics, StatsBase, Distributions

Random.seed!(42)

m = 2
modes = 2

weight_shift_distribution = Product(fill(Uniform(-0.01,0.01), m))

initial_modes = [[1,1], [2,2]]

repetitions = 30
history_length = 100

demand_sequences = [zeros(history_length+1) for _ in 1:repetitions]
for repetition in 1:repetitions
    demand_probability = initial_demand_probability
    for t in 1:history_length+1
        demand_sequences[repetition][t] = rand(Binomial(number_of_consumers, demand_probability))
        demand_probability = min(max(demand_probability + rand(shift_distribution), 0), 1.0)
    end
end









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
            stack(demand_sequences[2:100])[1:end-1,:], 
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



    default() # Reset plot defaults.

    gr(size = (600,400))

    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 17)
    secondary_font = Plots.font(font_family, pointsize = 11)
    legend_font = Plots.font(font_family, pointsize = 15)

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

    W₁_windowing_weights = reverse(windowing_weights(history_length, [0], round(Int, W₁_windowing_t)))
    W₁_smoothing_weights = reverse(smoothing_weights(history_length, [0], W₁_smoothing_α))
    W₁_weights = reverse(W₁_concentration_weights(history_length, W₁_concentration_ε, W₁_concentration_ϱ))

    W₁_windowing_t = round(Int, W₁_windowing_t)

    plt = plot(1:history_length, stack([W₁_windowing_weights, W₁_smoothing_weights, W₁_weights]), 
            xlabel = "Time", 
            ylabel = "Probability",
            xlims = (0,history_length+1),
            #legend = nothing,
            labels = ["\$t=$W₁_windowing_t\$" "\$α=$W₁_smoothing_α\$" "\$ε=$W₁_concentration_ε\$, \$ϱ=$W₁_concentration_ϱ\$"],
            colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3]],
            #markershapes = [:circle :diamond :hexagon],
            seriestypes = [:steppre :line :line],
            alpha = 1,
            #markersize = 2,
            #linestyles = :auto,
            linewidth = 1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)

    display(plt);

    #savefig(plt, "figures/W1-weights.pdf")




    default() # Reset plot defaults.

    gr(size = (600,400))

    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 17)
    secondary_font = Plots.font(font_family, pointsize = 11)
    legend_font = Plots.font(font_family, pointsize = 15)

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

    W₂_windowing_weights = reverse(windowing_weights(history_length, [0], round(Int, W₂_windowing_t)))
    W₂_smoothing_weights = reverse(smoothing_weights(history_length, [0], W₂_smoothing_α))
    W₂_weights = reverse(W₂_concentration_weights(history_length, W₂_concentration_ε, W₂_concentration_ϱ))

    W₂_windowing_t = round(Int, W₂_windowing_t)

    plt = plot(1:history_length, stack([W₂_windowing_weights, W₂_smoothing_weights, W₂_weights]), 
            xlabel = "Time", 
            ylabel = "Probability",
            xlims = (0,history_length+1),
            #legend = nothing,
            labels = ["\$ε=$W₂_windowing_ε\$, \$t=$W₂_windowing_t\$" "\$ε=$W₂_smoothing_ε\$, \$α=$W₂_smoothing_α\$" "\$ε=$W₂_concentration_ε\$, \$ϱ=$W₂_concentration_ϱ\$"],
            colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3]],
            #markershapes = [:circle :diamond :hexagon],
            seriestypes = [:steppre :line :line],
            alpha = 1,
            #markersize = 2,
            #linestyles = :auto,
            linewidth = 1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 3pt, 
            leftmargin = 3pt)

    display(plt);

    #savefig(plt, "figures/W2-weights.pdf")

end