using CSV

csv_file_path = "dairy-prices.csv"

data = CSV.File(csv_file_path) # Units are in dollars per tonne.

dairy_prices = [(Float64.([row.AMF, row.BUT, row.BMP, row.SMP, row.WMP])) for row in Iterators.take(data, size(data)[1])]
log_dairy_prices = [log.(dairy_prices[t]) for t in eachindex(dairy_prices)]
extracted_data = log_dairy_prices

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
            xminorticks = 3, 
            yminorticks = 0,
            fontfamily = font_family,
            guidefont = primary_font,
            tickfont = secondary_font,
            legendfont = legend_font)
    
    labels = ["AMF" "BUT" "BMP" "SMP" "WMP"]
    colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]]
    
    plt = plot(1:14*1*12, 
            stack(dairy_prices)'./1000, 
            xlabel = "Time (year)", 
        xticks = ([-5,3*12-5,6*12-5,9*12-5,12*12-5,14*12+6], ["2010","2013","2016","2019","2022","2025"]),
        xlims = (-5,14*1*12+6),
            ylabel = "Price (k\$/t)",
            labels = labels,
            legend = :outerright,
            legendfonthalign = :center,
            color = colors,
            linewidth = 1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 10pt, 
            leftmargin = 7pt)
    
    display(plt)
    savefig(plt, "figures/dairy-prices.pdf")

end