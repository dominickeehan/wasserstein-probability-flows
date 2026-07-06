using Dates
using Downloads
using JSON3
using Printf

constituents_csv = "sp500-constituents-2014-01-01.csv"
date_source_csv = "stock-returns.csv"
output_csv = "stock-returns-sp500-2014.csv"
min_stocks = 100
ticker_limit = 0

period_start = floor(Int, datetime2unix(DateTime(2014, 1, 1)))
period_stop = floor(Int, datetime2unix(DateTime(2024, 2, 1)))

function load_constituent_tickers(path)
    lines = filter(!isempty, strip.(readlines(path)))

    tickers = String[]
    for line in lines[2:end]
        ticker = strip(split(line, ","; limit = 2)[1])
        isempty(ticker) || push!(tickers, ticker)
    end

    if ticker_limit > 0
        tickers = tickers[1:min(ticker_limit, length(tickers))]
    end

    return unique(tickers)
end

function load_target_dates(path)
    lines = filter(!isempty, strip.(readlines(path)))

    return [split(line, ","; limit = 2)[1] for line in lines[2:end]]
end

yahoo_ticker(ticker) = replace(ticker, "." => "-")

function yahoo_chart_url(ticker)
    symbol = yahoo_ticker(ticker)
    return "https://query2.finance.yahoo.com/v8/finance/chart/$symbol?period1=$period_start&period2=$period_stop&interval=1mo&events=history&includeAdjustedClose=true"
end

function yahoo_chart_response(ticker)
    url = yahoo_chart_url(ticker)
    for attempt in 1:5
        path = tempname()
        try
            Downloads.download(url, path; headers = ["User-Agent" => "Mozilla/5.0"])
            return JSON3.read(read(path, String))
        catch err
            attempt == 5 && rethrow()
            message = sprint(showerror, err)
            if occursin("HTTP/2 400", message) || occursin("HTTP/2 404", message)
                rethrow()
            end
            println("Retrying $ticker after download failure")
            sleep(2.0^(attempt - 1))
        finally
            rm(path; force = true)
        end
    end
end

function yahoo_monthly_adjusted_closes(ticker)
    response = yahoo_chart_response(ticker)
    if response.chart.error !== nothing
        error("Yahoo Finance returned an error for $ticker: $(response.chart.error)")
    end

    result = response.chart.result[1]
    closes = result.indicators.adjclose[1].adjclose
    if any(value -> value === nothing, closes)
        error("Yahoo Finance returned a missing adjusted close for $ticker")
    end
    return [Float64(value) for value in closes]
end

function close_to_close_returns(closes)
    return [(closes[i + 1] / closes[i]) - 1 for i in 1:(length(closes) - 1)]
end

function format_return(value)
    text = @sprintf("%.9f", value)
    text = replace(text, r"0+$" => "")
    text = replace(text, r"\.$" => "")
    return text == "-0" ? "0" : text
end

dates = load_target_dates(date_source_csv)
tickers = load_constituent_tickers(constituents_csv)
println("Candidate tickers: $(length(tickers))")

returns_by_ticker = Dict{String, Vector{String}}()
failures = Pair{String, String}[]
for ticker in tickers
    try
        closes = yahoo_monthly_adjusted_closes(ticker)
        returns = close_to_close_returns(closes)
        if length(returns) == length(dates)
            returns_by_ticker[ticker] = format_return.(returns)
            println("Fetched $ticker")
        else
            reason = "expected $(length(dates)) returns but got $(length(returns))"
            push!(failures, ticker => reason)
            println("Skipping $ticker: $reason")
        end
    catch err
        reason = sprint(showerror, err)
        push!(failures, ticker => reason)
        println("Skipping $ticker: $reason")
    end
end

kept_tickers = [ticker for ticker in tickers if haskey(returns_by_ticker, ticker)]
if length(kept_tickers) < min_stocks
    println("Only $(length(kept_tickers)) complete tickers, below the target of $min_stocks")
end

open(output_csv, "w") do io
    println(io, join(["Date"; kept_tickers], ","))
    for i in eachindex(dates)
        println(io, join([dates[i]; [returns_by_ticker[ticker][i] for ticker in kept_tickers]], ","))
    end
end

println("Wrote $output_csv with $(length(kept_tickers)) tickers")
println("Skipped $(length(failures)) tickers")
