using Downloads
using JSON3
using Printf

const ORIGINAL_STOCK_RETURNS_PATH = "stock-returns.csv"
const ADDITIONAL_STOCK_RETURNS_PATH = "stock-returns-additional.csv"
const COMBINED_STOCK_RETURNS_PATH = "stock-returns-30.csv"

const REQUESTED_ADDITIONAL_STOCK_TICKERS = ["LLY", "V", "CAT", "BAC", "GE", "PG", "ABBV", "HD", "PM", "MRK"]
const EXPANDED_ADDITIONAL_STOCK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "CSCO", "QCOM", "UNH", "COST", "CVX", "NEE",
]
const ADDITIONAL_STOCK_TICKERS = [REQUESTED_ADDITIONAL_STOCK_TICKERS; EXPANDED_ADDITIONAL_STOCK_TICKERS]

const PERIOD1 = 1388534400 # 2014-01-01 00:00:00 UTC
const PERIOD2 = 1706745600 # 2024-02-01 00:00:00 UTC, includes Jan 2024 month-end close

function yahoo_chart_url(ticker)
    return "https://query2.finance.yahoo.com/v8/finance/chart/$ticker?period1=$PERIOD1&period2=$PERIOD2&interval=1mo&events=history&includeAdjustedClose=true"
end

function yahoo_chart_response(ticker)
    url = yahoo_chart_url(ticker)

    for attempt in 1:5
        download_path = tempname()
        try
            Downloads.download(
                url,
                download_path;
                headers = ["User-Agent" => "Mozilla/5.0"],
            )

            response_text = read(download_path, String)
            return JSON3.read(response_text)
        catch err
            if attempt == 5
                rethrow()
            end

            @warn "Retrying Yahoo Finance download" ticker attempt exception = (err, catch_backtrace())
            sleep(2.0^(attempt - 1))
        finally
            rm(download_path; force = true)
        end
    end
end

function yahoo_monthly_closes(ticker)
    response = yahoo_chart_response(ticker)
    if response.chart.error !== nothing
        error("Yahoo Finance returned an error for $ticker: $(response.chart.error)")
    end

    close_values = response.chart.result[1].indicators.quote[1].close
    if any(value -> value === nothing, close_values)
        error("Yahoo Finance returned a missing monthly close for $ticker")
    end

    return [Float64(value) for value in close_values]
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

function load_original_rows(path)
    lines = filter(!isempty, strip.(readlines(path)))
    length(lines) >= 2 || error("No stock-return rows found in $path")
    header = split(lines[1], ",")
    rows = [split(line, ",") for line in lines[2:end]]
    return header, rows
end

function validate_stock_universe(original_header)
    original_tickers = original_header[2:end]
    additional_tickers = ADDITIONAL_STOCK_TICKERS
    combined_tickers = [original_tickers; additional_tickers]

    length(unique(additional_tickers)) == length(additional_tickers) ||
        error("Additional stock ticker list contains duplicates")
    length(unique(combined_tickers)) == length(combined_tickers) ||
        error("Combined stock ticker list contains duplicates")
    length(combined_tickers) == 30 ||
        error("Expected 30 total stock tickers but found $(length(combined_tickers))")
end

function main()
    original_header, original_rows = load_original_rows(ORIGINAL_STOCK_RETURNS_PATH)
    validate_stock_universe(original_header)
    dates = [row[1] for row in original_rows]

    returns_by_ticker = Dict{String, Vector{String}}()
    for ticker in ADDITIONAL_STOCK_TICKERS
        closes = yahoo_monthly_closes(ticker)
        returns = close_to_close_returns(closes)
        length(returns) == length(dates) || error(
            "Expected $(length(dates)) returns for $ticker but got $(length(returns))",
        )
        returns_by_ticker[ticker] = format_return.(returns)
        println("Fetched $ticker")
    end

    open(ADDITIONAL_STOCK_RETURNS_PATH, "w") do io
        println(io, join(["Date"; ADDITIONAL_STOCK_TICKERS], ","))
        for i in eachindex(dates)
            println(io, join([dates[i]; [returns_by_ticker[ticker][i] for ticker in ADDITIONAL_STOCK_TICKERS]], ","))
        end
    end

    open(COMBINED_STOCK_RETURNS_PATH, "w") do io
        println(io, join([original_header; ADDITIONAL_STOCK_TICKERS], ","))
        for (i, original_row) in enumerate(original_rows)
            println(io, join([original_row; [returns_by_ticker[ticker][i] for ticker in ADDITIONAL_STOCK_TICKERS]], ","))
        end
    end

    println("Wrote $ADDITIONAL_STOCK_RETURNS_PATH")
    println("Wrote $COMBINED_STOCK_RETURNS_PATH")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
