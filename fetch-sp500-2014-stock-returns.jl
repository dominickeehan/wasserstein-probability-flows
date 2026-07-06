using Dates
using Downloads
using JSON3
using Printf

const CONSTITUENTS_CSV = "sp500-constituents-2014-01-01.csv"
const DEFAULT_DATE_SOURCE_CSV = "stock-returns.csv"
const DEFAULT_OUTPUT_CSV = "stock-returns-sp500-2014.csv"

const PERIOD1 = floor(Int, datetime2unix(DateTime(2014, 1, 1)))
const PERIOD2 = floor(Int, datetime2unix(DateTime(2024, 2, 1)))

function env_int(name, default)
    value = get(ENV, name, "")
    isempty(value) && return default
    return parse(Int, value)
end

function load_constituent_tickers(path)
    lines = filter(!isempty, strip.(readlines(path)))
    length(lines) >= 2 || error("No constituent rows found in $path")

    tickers = String[]
    for line in lines[2:end]
        ticker = strip(split(line, ","; limit = 2)[1])
        isempty(ticker) || push!(tickers, ticker)
    end

    limit = env_int("SP500_LIMIT", 0)
    if limit > 0
        tickers = tickers[1:min(limit, length(tickers))]
    end
    return unique(tickers)
end

function load_target_dates(path)
    lines = filter(!isempty, strip.(readlines(path)))
    length(lines) >= 2 || error("No date rows found in $path")
    return [split(line, ","; limit = 2)[1] for line in lines[2:end]]
end

yahoo_ticker(ticker) = replace(ticker, "." => "-")

function yahoo_chart_url(ticker)
    symbol = yahoo_ticker(ticker)
    return "https://query2.finance.yahoo.com/v8/finance/chart/$symbol?period1=$PERIOD1&period2=$PERIOD2&interval=1mo&events=history&includeAdjustedClose=true"
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
            @warn "Retrying Yahoo Finance download" ticker attempt exception = (err, catch_backtrace())
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
    length(closes) >= 2 || error("Need at least two closes")
    return [(closes[i + 1] / closes[i]) - 1 for i in 1:(length(closes) - 1)]
end

function format_return(value)
    text = @sprintf("%.9f", value)
    text = replace(text, r"0+$" => "")
    text = replace(text, r"\.$" => "")
    return text == "-0" ? "0" : text
end

function main()
    output_path = get(ENV, "SP500_OUTPUT_CSV", DEFAULT_OUTPUT_CSV)
    date_source_path = get(ENV, "SP500_DATE_SOURCE_CSV", DEFAULT_DATE_SOURCE_CSV)
    min_stocks = env_int("SP500_MIN_STOCKS", 100)

    dates = load_target_dates(date_source_path)
    tickers = load_constituent_tickers(CONSTITUENTS_CSV)
    println("candidate_tickers=$(length(tickers)) output=$output_path")

    returns_by_ticker = Dict{String, Vector{String}}()
    failures = Pair{String, String}[]
    for ticker in tickers
        try
            closes = yahoo_monthly_adjusted_closes(ticker)
            returns = close_to_close_returns(closes)
            if length(returns) != length(dates)
                error("expected $(length(dates)) returns but got $(length(returns))")
            end
            returns_by_ticker[ticker] = format_return.(returns)
            println("Fetched $ticker")
        catch err
            push!(failures, ticker => sprint(showerror, err))
            @warn "Skipping ticker" ticker reason = sprint(showerror, err)
        end
    end

    kept_tickers = [ticker for ticker in tickers if haskey(returns_by_ticker, ticker)]
    length(kept_tickers) >= min_stocks ||
        error("Only $(length(kept_tickers)) complete tickers, below SP500_MIN_STOCKS=$min_stocks")

    open(output_path, "w") do io
        println(io, join(["Date"; kept_tickers], ","))
        for i in eachindex(dates)
            println(io, join([dates[i]; [returns_by_ticker[ticker][i] for ticker in kept_tickers]], ","))
        end
    end

    println("Wrote $output_path with $(length(kept_tickers)) tickers")
    if !isempty(failures)
        println("Skipped $(length(failures)) tickers")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
