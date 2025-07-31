using CSV

csv_file_path = "stock-returns.csv"

data = CSV.File(csv_file_path)

stock_returns = [(Float64.([row.BA, row.BRKB, row.GS, row.JNJ, row.JPM, row.KO, row.MCD, row.PFE, row.WMT, row.XOM])) for row in Iterators.take(data, size(data)[1])]
extracted_data = stock_returns

m = 10
wealth = zeros((length(stock_returns)+1, m))
wealth[1,:] .= 1
for t in 1:length(stock_returns)+1 - 1
    wealth[t+1,:] .= wealth[t,:].*(ones(m)+stock_returns[t])
end

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
    
    labels = ["BA" "BRK.B" "GS" "JNJ" "JPM" "KO" "MCD" "PFE" "WMT" "XOM"]
    colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5] palette(:tab10)[6] palette(:tab10)[7] palette(:tab10)[8] palette(:tab10)[9] palette(:tab10)[10]]
    linestyles = [:solid :solid :solid :solid :solid :solid :solid :solid :solid :solid]

    plt = plot(1:10*12+1, 
            wealth, 
            xlabel = "Time (year)", 
            xticks = (1:2*12:10*12+1, ["2014","2016","2018","2020","2022","2024"]),
            xlims = (1-6,10*12+6+1),
            ylims = (0,4),
            ylabel = "Stock value (normalised)",
            labels = labels,
            legend = :outerright,
            #legendfonthalign = :center,
            color = colors,
            linestyle = linestyles,
            linewidth = 1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 10pt, 
            leftmargin = 7pt)
    
    display(plt)
    savefig(plt, "figures/stock-returns.pdf")


end