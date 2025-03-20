using CSV

csv_file_path = "stock-returns.csv"

data = CSV.File(csv_file_path)

stock_returns = [(Float64.([row.AAPL, row.AMZN, row.BA, row.BRKB, row.GE, row.GS, row.JNJ, row.JPM, row.KO, row.MCD, row.MSFT, row.PFE, row.WMT, row.XOM])) for row in Iterators.take(data, size(data)[1])]
stock_returns = stock_returns[120+1:end] # 10 years of monthly data.
extracted_data = stock_returns

m = 14
wealth = zeros((length(stock_returns)+1, m))
wealth[1,:] .= 1
for t in 1:length(stock_returns)+1 - 1
    wealth[t+1,:] .= wealth[t,:].*(ones(m)+stock_returns[t])
end

do_plots = true
if do_plots == true
    using Plots, Measures

    default() # Reset plot defaults.

    gr(size = (830,343))
    
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
    
    labels = ["AAPL" "AMZN" "BA" "BRK.B" "GE" "GS" "JNJ" "JPM" "KO" "MCD" "MSFT" "PFE" "WMT" "XOM"]
    colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5] palette(:tab10)[6] palette(:tab10)[7] palette(:tab10)[8] palette(:tab10)[9] palette(:tab10)[10] palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4]]
    linestyles = [:solid :solid :solid :solid :solid :solid :solid :solid :solid :solid :dash :dash :dash :dash]

    plt = plot(1:10*12+1, 
            wealth, 
            xlabel = "Time (year)", 
            xticks = (1:12:10*12+1, ["2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]),
            xlims = (1-4,10*12+4),
            ylims = (0,11.5),
            ylabel = "Net wealth",
            labels = labels,
            legend = :outerright,
            #legendfonthalign = :center,
            color = colors,
            linestyle = linestyles,
            linewidth = 1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 15.5pt, 
            leftmargin = 12.5pt)
    
    display(plt)
    savefig(plt, "figures/stock-returns.pdf")


end