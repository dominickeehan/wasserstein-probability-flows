using CSV

csv_file_path = "metal-prices.csv"

data = CSV.File(csv_file_path) # Units are in dollars per tonne.

metal_prices = [(Float64.([row.copper, row.nickel, row.lead, row.zinc])) for row in Iterators.take(data, size(data)[1])]
log_metal_prices = [log.(metal_prices[t]) for t in eachindex(metal_prices)]
extracted_data = log_metal_prices

do_plots = true
if do_plots == true
    using Plots

    display(plot(1:length(log_metal_prices), stack(log_metal_prices)', labels = nothing,))

end