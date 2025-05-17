using CSV

csv_file_path = "dairy-prices.csv"

data = CSV.File(csv_file_path) # Units are in dollars per tonne.

dairy_prices = [(Float64.([row.amf, row.but, row.bmp, row.smp, row.wmp])) for row in Iterators.take(data, size(data)[1])]
#dairy_prices = [(Float64.([row.wmp])) for row in Iterators.take(data, size(data)[1])]
dairy_prices = dairy_prices[1:2:end] # Convert to per month.
log_dairy_prices = [log.(dairy_prices[t]) for t in eachindex(dairy_prices)]
extracted_data = log_dairy_prices

do_plots = true
if do_plots == true
    using Plots, Measures

    default() # Reset plot defaults.

    gr(size = (850,343))
    
    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 15)
    secondary_font = Plots.font(font_family, pointsize = 11)
    legend_font = Plots.font(font_family, pointsize = 13)
    
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
    
    labels = ["AMF" "BUT" "BMP" "SMP" "WMP"]
    colors = [palette(:tab10)[1] palette(:tab10)[2] palette(:tab10)[3] palette(:tab10)[4] palette(:tab10)[5]]
    
    plt = plot(1:14*1*12, 
            stack(dairy_prices)'./1000, 
            xlabel = "Time (year)", 
            xticks = (1*6+1:1*12:14*1*12, ["2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]),
            xlims = (-5,14*1*12+6),
            ylabel = "Price (k\$/t)",
            labels = labels,
            legend = :outerright,
            legendfonthalign = :center,
            color = colors,
            linewidth = 1,
            topmargin = 0pt, 
            rightmargin = 0pt,
            bottommargin = 16pt, 
            leftmargin = 12.5pt)
    
    display(plt)
    savefig(plt, "figures/dairy-prices.pdf")

end