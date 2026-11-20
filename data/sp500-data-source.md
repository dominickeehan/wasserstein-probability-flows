# S&P 500 January 2014 survivor dataset

## Files

- `sp500-constituents-2014-01-01.csv` is the exact 500-row constituent snapshot used as the candidate universe.
- `sp500-2014-survivor-audit.csv` records the 2014 ticker, company, Yahoo price ticker/lineage, inclusion decision, and reason for every candidate.
- `stock-returns-S&P-500-2014-survivors.csv` contains 120 monthly return observations for the 379 candidates with a complete history.

The existing `stock-returns-S&P-500-2014.csv` is retained unchanged for comparison. It should not be treated as the authoritative January 2014 survivor file.

## Constituent source

The candidate universe is the constituent table in the last Wikipedia revision before 1 January 2014:

- Page: [List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- Revision: [oldid 588150360](https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=588150360)
- Revision timestamp: `2013-12-29T01:53:50Z`
- Extracted constituent rows: `500`

This point-in-time revision is used instead of reconstructing the past from the current Wikipedia table. S&P Dow Jones Indices independently stated that the index would move from 500 to 501 trading lines only when Google's Class C shares entered on 3 April 2014: [Inside the S&P 500: Multiple Share Classes](https://www.indexologyblog.com/2014/03/30/inside-the-sp-500-multiple-share-classes/).

Wikipedia is a secondary source. The exact revision identifier and extracted snapshot are retained so the candidate universe is reproducible and auditable.

## Price and return source

The monthly adjusted closing prices underlying the returns came from the Yahoo Finance chart endpoint used to generate the repository's original return file:

```text
https://query2.finance.yahoo.com/v8/finance/chart/{symbol}
  ?period1=1388534400
  &period2=1706745600
  &interval=1mo
  &events=history
  &includeAdjustedClose=true
```

- Price window requested: 1 January 2014 through 1 February 2024.
- Return window written: January 2014 through December 2023.
- Monthly return definition: `adjusted_close[t] / adjusted_close[t-1] - 1`.
- Required completeness: 121 consecutive non-missing monthly adjusted closes, producing 120 returns.
- The corrected file was derived from the already-downloaded Yahoo return series in `stock-returns-S&P-500-2014.csv`; it was not independently re-downloaded on 14 July 2026.

## Definition of “survivor”

For this dataset, a survivor is a company/security that:

1. appears in the 500-row point-in-time constituent table above; and
2. has a complete Yahoo-derived monthly adjusted-close return series from January 2014 through December 2023 under the same security lineage.

This does **not** mean that the security necessarily remained an S&P 500 constituent for the entire ten-year period. It is a complete-price-history filter applied to the January 2014 constituent universe.

Output columns use the ticker shown in the 2014 constituent table. Where Yahoo stores the historical series under a later ticker, the series is mapped back to its 2014 label. Examples include `COR -> ABC`, `BNY -> BK`, `META -> FB`, `MRSH -> MMC`, and `RTX -> UTX`. All mappings and inclusion decisions are listed in `sp500-2014-survivor-audit.csv`.

`ACT` and `AGN` were separate constituents in the source table. Actavis later acquired Allergan and adopted the `AGN` identity, so `ACT` cannot be represented as a distinct complete ten-year series and is excluded.

## Correction to the earlier 505-row candidate file

The earlier generated candidate list contained 505 rows because it added six securities that were not in the point-in-time source table—`CPAY`, `DAY`, `ECHO`, `IQV`, `UAA`, and `WTW`—while collapsing the separate `ACT` and `AGN` constituents into one lineage.

Of those six erroneous candidates, five (`CPAY`, `ECHO`, `IQV`, `UAA`, and `WTW`) had passed the original complete-history filter and have been removed from the corrected return file. `DAY` had already failed the filter.

## Validation totals

- Point-in-time candidates: `500`
- Included complete-history survivors: `379`
- Excluded for incomplete/unavailable distinct history: `121`
- Return months: `120`
- Date coverage: `Jan-14` through `Dec-23`
- Missing or non-numeric return cells in the output: `0`
- Duplicate output tickers: `0`
- Output tickers outside the point-in-time 500: `0`
