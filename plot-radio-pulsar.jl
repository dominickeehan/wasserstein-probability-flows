using Random, Distributions, Statistics, StatsBase
using LinearAlgebra
using IterTools, ProgressBars

include("weights.jl")

Cu = 4  # Underage cost.
Co = 1  # Overage cost.

#q = Cu / (Cu + Co)   
#quantile(demand[i, :], Weights(weights), q)

Random.seed!(42)

dimensions = 1
modes = 2

history_length = 100

# Initial demand-distribution parameters. Mixture of axis-aligned normals.
μ = [i*100 for i in 1:modes]
σ = 20

# Demand-mode shift-distribution parameters.
shift_distribution = [MvNormal(zeros(dimensions), (4^2) * I) for _ in 1:modes]

demands = [zeros(dimensions) for _ in 1:history_length]
μs_history = [[ones(dimensions) * μ[i] for i in 1:modes] for _ in 1:history_length]

μs = [ones(dimensions) * μ[i] for i in 1:modes]
for t in 1:history_length
    for i in eachindex(μs); μs_history[t][i] = μs[i]; end
    demands[t] = rand(MixtureModel(MvNormal, [(μs[i], Diagonal(fill((σ)^2, dimensions))) for i in 1:modes]))
    
    for i in eachindex(μs); μs[i] += rand(shift_distribution[1]); end

end

d(ξ, ζ) = norm(ξ - ζ, 2)
λ = 10
weights = WPF_weights(demands, λ, d)


using Plots, Measures

default() # Reset plot defaults.

gr(size = (337+6,224+6).*sqrt(3))

fontfamily = "Computer Modern"

default(framestyle = :box,
        grid = false,
        #gridlinewidth = 1.0,
        #gridalpha = 0.075,
        minorgrid = false,
        #minorgridlinewidth = 1.0, 
        #minorgridalpha = 0.075,
        #minorgridlinestyle = :dash,
        ytick_direction = :none,
        xtick_direction = :in,
        xminorticks = 0, 
        yminorticks = 0,
        fontfamily = fontfamily,
        guidefont = Plots.font(fontfamily, pointsize = 12),
        tickfont = Plots.font(fontfamily, pointsize = 10),
        legendfont = Plots.font(fontfamily, pointsize = 11))

Ξ = LinRange(-0, 300, 1000)
yl = (Ξ[1],Ξ[end])

horizontal_increment = 0.001

plt = plot(xlabel = "Time, t",
                        ylabel = "Outcome space, \$Ξ\$", 
                        yformatter=_->"",
                        topmargin = 0pt, 
                        rightmargin = 0pt,
                        bottommargin = 6pt, 
                        leftmargin = 6pt,
                        color = cgrad([palette(:tab10)[2], palette(:tab10)[1]], [34/50]),
                        colorbar_title = "\nOut-of-sample performance difference")

for t in 1:history_length
        alpha = weights[t]+0.01

        plot!([-pdf(MixtureModel(Normal, [(μs_history[t][i][1], σ) for i in 1:modes]), ξ) for ξ in Ξ].+horizontal_increment*t, 
                Ξ,
                color = palette(:tab10)[1],
                linewidth = 1,
                alpha = 10*alpha,
                fill = (0, alpha, palette(:tab10)[1]),
                label = nothing)

        scatter!([-pdf(MixtureModel(Normal, [(μs_history[t][i][1], σ) for i in 1:modes]), demands[t][1])+horizontal_increment*t],
                [demands[t][1]],
                markershape = :circle,
                markersize = 3,
                markerstrokecolor = palette(:tab10)[1],
                #markerstrokealpha = 1, #10*weights[t],
                markerstrokewidth = 0.0,
                markercolor = palette(:tab10)[1],
                alpha = 20*alpha,
                label = nothing)

end

outer_increment = 11

x = [demands[t][1] for t in 1:history_length]
w = weights
# Weighted histogram
h = fit(Histogram, x, Weights(w), nbins=10)
# Extract bin edges and weighted counts
edges = h.edges[1]
counts = h.weights
widths = diff(edges)
total_weight = sum(w)
# Density heights (constant within each bin)
densities = counts ./ (total_weight .* widths)
# Build stepwise coordinates
xcoords = repeat(edges, inner=2)[2:end-1]   # repeat edges so each appears twice, trim first/last
ycoords = repeat(densities, inner=2)        # repeat each density twice
plot!([horizontal_increment*(history_length+1+outer_increment); horizontal_increment*(history_length+1+outer_increment); -ycoords.+horizontal_increment*(history_length+1+outer_increment); horizontal_increment*(history_length+1+outer_increment); horizontal_increment*(history_length+1+outer_increment);], 
        [Ξ[1]; xcoords[1]; xcoords; xcoords[end]; Ξ[end];],
        color = palette(:tab10)[2],
        linewidth = 1,
        linestyle = :dash,
        alpha = 1,
        fill = (0, 0.5, palette(:tab10)[2]),
        label = nothing)

xticks!([0+horizontal_increment*t for t in 0:10:history_length],["$i" for i in 0:10:history_length])

ylims!(yl)
xlims!((-outer_increment-1,history_length+1+outer_increment).*horizontal_increment)

plot!([-100,-99], 
        [100, 100], 
        label = "Underlying density",
        color = palette(:tab10)[1],
        linewidth = 1,
        alpha = 1,
        fill = (0, 0.5, palette(:tab10)[1]))
scatter!([-100,-99], 
        [100, 100], 
        label = "Observation",#, \$ξ_t\$",
        markershape = :circle,
        markersize = 3,
        markerstrokewidth = 0.0,
        markercolor = palette(:tab10)[1],
        markeralpha = 0.9,
        legend = :topleft)
plot!([-100,-99], 
        [100, 100], 
        label = "Estimated density",
        color = palette(:tab10)[2],
        linewidth = 1,
        linestyle = :dash,
        alpha = 1,
        fill = (0, 0.5, palette(:tab10)[2]))

display(plt)
savefig(plt,"figures\\radio-pulsar.pdf")










