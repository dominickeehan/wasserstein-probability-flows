#!/usr/bin/env python3
"""Download a small, reproducible daily adjusted-return universe for the WPF pilot."""

import csv
import datetime as dt
import json
import math
import os
import random
import time
import urllib.parse
import urllib.request


AUDIT_PATH = os.environ.get("PILOT_AUDIT_CSV", "data/sp500-2014-survivor-audit.csv")
OUTPUT_PATH = os.environ.get("PILOT_DAILY_CSV", "data/stock-returns-daily-pilot.csv")
TARGET_ASSETS = int(os.environ.get("PILOT_UNIVERSE_SIZE", "50"))
SEED = int(os.environ.get("PILOT_UNIVERSE_SEED", "20260810"))
START_DATE = dt.date.fromisoformat(os.environ.get("PILOT_START_DATE", "2019-01-01"))
END_DATE = dt.date.fromisoformat(os.environ.get("PILOT_END_DATE", "2024-01-01"))


def unix_seconds(day):
    instant = dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc)
    return int(instant.timestamp())


def fetch_adjusted_closes(symbol):
    encoded_symbol = urllib.parse.quote(symbol, safe="")
    query = urllib.parse.urlencode(
        {
            "period1": unix_seconds(START_DATE),
            "period2": unix_seconds(END_DATE),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    prices = {}
    for timestamp, price in zip(timestamps, adjusted):
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            continue
        day = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).date()
        prices[day] = float(price)
    return prices


def load_candidates():
    with open(AUDIT_PATH, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    candidates = [
        (row["Ticker2014"], row["YahooPriceTicker"])
        for row in rows
        if row["Included"].strip().lower() == "yes" and row["YahooPriceTicker"].strip()
    ]
    random.Random(SEED).shuffle(candidates)
    return candidates


def main():
    if TARGET_ASSETS < 2:
        raise ValueError("PILOT_UNIVERSE_SIZE must be at least 2")
    if END_DATE <= START_DATE:
        raise ValueError("PILOT_END_DATE must be after PILOT_START_DATE")

    series = []
    for ticker, yahoo_ticker in load_candidates():
        if len(series) >= TARGET_ASSETS:
            break
        try:
            prices = fetch_adjusted_closes(yahoo_ticker)
            if len(prices) < 1000:
                print(f"skip {ticker} ({yahoo_ticker}): only {len(prices)} prices", flush=True)
                continue
            series.append((ticker, yahoo_ticker, prices))
            print(f"accepted {ticker} ({yahoo_ticker}): {len(prices)} prices", flush=True)
        except Exception as error:
            print(f"skip {ticker} ({yahoo_ticker}): {error}", flush=True)
        time.sleep(0.05)

    if len(series) < TARGET_ASSETS:
        raise RuntimeError(f"downloaded only {len(series)} of {TARGET_ASSETS} requested assets")

    common_dates = set(series[0][2])
    for _, _, prices in series[1:]:
        common_dates.intersection_update(prices)
    dates = sorted(common_dates)
    if len(dates) < 1000:
        raise RuntimeError(f"only {len(dates)} common trading dates remain")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["Date"] + [ticker for ticker, _, _ in series])
        for date_index in range(1, len(dates)):
            previous_date = dates[date_index - 1]
            date = dates[date_index]
            returns = []
            for _, _, prices in series:
                value = prices[date] / prices[previous_date] - 1.0
                if not math.isfinite(value) or value <= -1.0:
                    raise RuntimeError(f"invalid return on {date}: {value}")
                returns.append(value)
            writer.writerow([date.isoformat()] + [f"{value:.12g}" for value in returns])

    print(
        f"wrote {len(dates) - 1} daily returns for {len(series)} assets to {OUTPUT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
