using CSV

csv_file_path = "stock-returns-30.csv"
stock_tickers = Symbol.(split(readlines(csv_file_path)[1], ",")[2:end])
stock_labels = replace.(String.(stock_tickers), "BRKB" => "BRK.B")

data = CSV.File(csv_file_path)

stock_returns = [(Float64.([getproperty(row, ticker) for ticker in stock_tickers])) for row in Iterators.take(data, size(data)[1])]
extracted_data = stock_returns

m = length(stock_tickers)
wealth = zeros((length(stock_returns)+1, m))
wealth[1,:] .= 1
for t in 1:length(stock_returns)+1 - 1
    wealth[t+1,:] .= wealth[t,:].*(ones(m)+stock_returns[t])
end
n_periods = length(stock_returns)

do_plots = true
if do_plots == true
    using Plots, Measures

    default() # Reset plot defaults.


        gr(size = (317+7+75,159+10).*sqrt(3))

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
    
    show_legend = m <= 20
    labels = show_legend ? permutedims(stock_labels) : nothing
    colors = permutedims([palette(:tab10)[mod1(i, 10)] for i in 1:m])
    linestyles = permutedims(fill(:solid, m))

    plt = plot(1:n_periods+1, 
            wealth, 
            xlabel = "Time (year)", 
            xticks = (1:2*12:n_periods+1, ["2014","2016","2018","2020","2022","2024"]),
            xlims = (1-6,n_periods+6+1),
            ylims = (0,max(4,1.05*maximum(wealth))),
            ylabel = "Stock value (normalised)",
            labels = labels,
            legend = show_legend ? :outerright : nothing,
            #legendfonthalign = :center,
            linetype = :steppost,
            color = colors,
            linestyle = linestyles,
            linewidth = 1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 10pt, 
            leftmargin = 7pt)
    
    display(plt)
    savefig(plt, "figures/stock-returns-30.pdf")


end
