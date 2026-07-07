using CSV, Random

csv_file_path = "stock-returns-S&P-500-2014.csv"

data = CSV.File(csv_file_path)
rows = collect(data)

5

stock_return_column_names = collect(propertynames(data))[2:end]
stock_return_tickers = String.(stock_return_column_names)
stock_return_dates = [String(row[:Date]) for row in rows]

stock_returns = [
    Float64[row[column_name] for column_name in stock_return_column_names]
    for row in rows
]
#stock_return_indices = sort(randperm(length(stock_return_tickers))[1:10]); stock_return_tickers = stock_return_tickers[stock_return_indices]; stock_returns = [returns[stock_return_indices] for returns in stock_returns]
extracted_data = stock_returns

m = length(stock_return_tickers)
wealth = zeros((length(stock_returns) + 1, m))
wealth[1, :] .= 1
for t in 1:length(stock_returns)
    wealth[t + 1, :] .= wealth[t, :] .* (ones(m) + stock_returns[t])
end

do_plots = true
if do_plots == true
    using Plots, Measures

    default() # Reset plot defaults.

    gr(size = (317 + 7 + 75, 159 + 10) .* sqrt(3))

    font_family = "Computer Modern"
    primary_font = Plots.font(font_family, pointsize = 12)
    secondary_font = Plots.font(font_family, pointsize = 10)
    legend_font = Plots.font(font_family, pointsize = 11)

    default(framestyle = :box,
            grid = true,
            gridalpha = 0.075,
            tick_direction = :in,
            xminorticks = 2,
            yminorticks = 0,
            fontfamily = font_family,
            guidefont = primary_font,
            tickfont = secondary_font,
            legendfont = legend_font)

    tick_positions = 1:(2 * 12):(length(stock_returns) + 1)
    tick_labels = string.(2014:2:(2014 + 2 * (length(tick_positions) - 1)))

    plt = plot(1:(length(stock_returns) + 1),
            wealth,
            xlabel = "Time (year)",
            xticks = (tick_positions, tick_labels),
            xlims = (1 - 6, length(stock_returns) + 6 + 1),
            ylabel = "Stock value (normalised)",
            labels = nothing,
            legend = false,
            linetype = :steppost,
            linewidth = m > 50 ? 0.35 : 1,
            linealpha = m > 50 ? 0.25 : 1,
            topmargin = 0pt,
            rightmargin = 0pt,
            bottommargin = 10pt,
            leftmargin = 7pt)

    display(plt)
    savefig(plt, "figures/stock-returns-S&P-500-2014.pdf")
end
