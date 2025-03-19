using CSV

csv_file_path = "exchange-rates.csv"

data = CSV.File(csv_file_path) # Units are in dollars per tonne.

exchange_rates = [(Float64.([row.Rate])) for row in Iterators.take(data, size(data)[1])]
exchange_rates = exchange_rates[9*12+15*12:end]
log_exchange_rates = [log.(exchange_rates[t]) for t in eachindex(exchange_rates)]
extracted_data = log_exchange_rates

do_plots = true
if do_plots == true
    using Plots

    display(plot(1:length(exchange_rates), stack(log_exchange_rates)', labels = nothing,))

end