# Dense-grid daily-return WPF pilot

## Purpose

This is a bounded computational pilot of daily risk measurement with monthly
portfolio decisions. It uses 20 investment subsets, 30 finite lambda parameters
plus the `0` and `Inf` endpoints, and separate tuning specifications for 5%, 10%,
and 20% tail risk.

It remains an exploratory development experiment, not final statistical
evidence.

## Design

- Data: 1,257 common daily adjusted-return observations from 3 January 2019 to
  29 December 2023 for a reproducibly sampled 50-stock universe.
- Replications: 20 seeded 20-stock investment subsets.
- Portfolio decisions: first trading day of each month; holdings drift without
  rebalancing during the month.
- WPF/SAA estimation: previous 126 trading days.
- Test: the 250 trading days in 2023.
- Objective: 10% mean loss plus 90% empirical CVaR.
- Portfolio constraints: long only, fully invested, and at most `5/m = 25%` in
  one stock.
- Distance: per-stock volatility-standardised mean L1 distance, rescaled so the
  median pairwise distance in each estimation window is one.
- Dimensionless penalty: `lambda = eta * 126`.
- SAA: the `eta = Inf` candidate, solved with the same data and constraints.
- Runtime: 185.5 seconds with eight Julia threads; the runner has a default
  540-second computation deadline and checkpoints every completed subset.

### Shared-signal implementation

The subset-specific dense version did not complete one parallel wave within
seven minutes and was stopped without retaining partial results. The completed
experiment therefore estimates WPF scenario weights from the common 50-stock
return signal once per month, then solves the portfolio separately inside each
20-stock subset. All subsets have the same information signal but different
feasible investment universes.

This reduces the number of dense WPF sweeps from `20 * 37` to `37`. It is not
the same experiment as estimating distances separately inside every subset.

### Tail-specific specifications

Every tail problem has 30 finite eta candidates plus `0` and `Inf`. Its finite
grid is a slice of a shared 34-point logarithmic master grid, avoiding 90
distinct WPF solves per month.

| Tail probability | Tuning days | Effective tuning-tail observations | Finite eta range |
|---:|---:|---:|---:|
| 5% | 504 | 25.2 | 0.01000 to 1.50267 |
| 10% | 252 | 25.2 | 0.01413 to 2.12321 |
| 20% | 126 | 25.2 | 0.01996 to 3.00000 |

The 126-day estimation window is a runtime concession. It contains only 6.3,
12.6, and 25.2 tail observations at the three risk levels.

## Results

Negative differences mean the rolling WPF/SAA selector improves on always
using SAA.

| Tail probability | Wins / losses | Mean difference from SAA | Median difference | Mean SAA selection share |
|---:|---:|---:|---:|---:|
| 5% | 4 / 16 | +6.22% | +4.59% | 15.4% |
| 10% | 5 / 15 | +4.42% | +5.87% | 12.1% |
| 20% | 3 / 17 | +7.05% | +5.92% | 0.0% |

The exact subset percentage differences were:

```text
5%:  +3.56, -1.22, +5.10, +9.95, +5.97, -5.35, +2.65, +17.31, -6.44, -12.47,
     +8.68, +27.59, +4.09, +13.37, +0.56, +7.66, +5.96, +33.05, +1.65, +2.71

10%: -1.51, -0.59, -4.81, +0.70, +6.68, +0.89, +10.75, +7.94, +8.47, +10.64,
     +5.30, +7.69, +12.07, +1.31, -11.80, +6.44, +8.86, +16.58, -0.73, +3.57

20%: +8.68, +24.48, +3.98, +2.37, +3.42, -1.01, +2.97, -3.15, +20.76, +7.20,
     +14.00, +9.80, +8.48, -1.38, +5.43, +12.22, +4.79, +6.41, +3.81, +7.73
```

Finite grid-edge selections accounted for 12.3% of finite choices at 5%, 3.8%
at 10%, and 2.5% at 20%. Thus the poor aggregate result is not generally caused
by all selected penalties piling up at a numerical grid boundary.

## Interpretation and limitations

The dense grid removes the apparent positive signal from the original
three-subset coarse-grid pilot. Comparing 32 candidates using approximately 25
effective tail observations still produces substantial selection overfitting.
At 20%, the 126-day selector never chose SAA and changed finite eta frequently;
equalising the number of tail observations does not give equal selection
stability when the total tuning history becomes very short.

The experiment suggests that the next design change should concern the
selection rule, not further grid density. Candidates include exact SAA, but
candidate containment cannot prevent an estimated rolling criterion from
choosing an inferior finite penalty. A longer common tuning window, a
one-standard-error SAA fallback, or persistence/hysteresis should be fixed on a
development period before another test.

Other limitations are the short 126-day estimation window, survivor-only
50-stock signal universe, overlapping 20-stock subsets, only one test year, no
transaction costs, and no block-bootstrap uncertainty interval.

## Artifacts

- `fetch-daily-stock-pilot.py`: reproducible Yahoo adjusted-close downloader.
- `stock-returns-daily-pilot.csv`: aligned daily pilot returns.
- `test-daily-stock-return-subsets-pilot.jl`: bounded dense-grid walk-forward
  runner.
- `stock-daily-pilot-results.csv`: 60 detailed seed-by-tail results.
