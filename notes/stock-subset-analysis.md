# WPF versus SAA on reproducible 100-stock subsets

## Design

- Data: 120 monthly returns from January 2014 through December 2023 for the
  379 complete-history securities in `stock-returns-S&P-500-2014-survivors.csv`.
- Test period: January 2019 through December 2023 (60 months).
- Subsets: ten reproducible 100-stock samples generated with Julia
  `MersenneTwister` seeds 1 through 10.
- Portfolio objective: 10% mean loss plus 90% 5%-CVaR.
- WPF tuning: the original 30-point finite grid from 0.1 through 100, plus
  0 and `Inf`, selected through the original rolling 24-month procedure.
- Uncertainty per row: paired six-month circular block bootstrap with 1,000
  repetitions.

Positive cost differences mean WPF is worse than SAA. Positive wealth
differences mean WPF produces greater terminal wealth.

## Main results

| Experiment | WPF wins | Median relative cost difference | Mean relative cost difference | Mean absolute cost difference | Mean-loss difference | CVaR difference | Mean wealth difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 379 stocks | 1/1 | -13.09% | -13.09% | -0.009996 | +0.001761 | -0.011303 | -0.1160 |
| Ten 100-stock subsets; subset L1 distances | 4/10 | +18.62% | +19.72% | +0.013979 | -0.001228 | +0.015669 | +0.1068 |
| Ten 100-stock subsets; all-379-stock L1 distances | 2/10 | +25.84% | +24.77% | +0.017072 | +0.000628 | +0.018899 | -0.0496 |

The ten subset-distance percentage differences were:

```text
+18.15, -3.09, +25.39, +32.82, +19.08,
+51.83, -18.48, +91.13, -16.46, -3.22
```

The full-signal percentage differences were:

```text
+12.60, +44.82, +21.20, +35.61, -7.95,
+53.85, +7.18, +82.00, -32.06, +30.47
```

Across the ten subset-distance experiments, a subset bootstrap gave a 95%
interval of `[+0.0010, +0.0282]` for the mean absolute WPF-minus-SAA cost.
The analogous full-signal interval was `[+0.0027, +0.0304]`. These are
descriptive intervals: the subsets overlap and all use the same test months.

The full-universe -13.09% estimate had a six-month block-bootstrap interval of
`[-27.73%, +14.13%]`, so the observed advantage is not precise with only 60
test months and three observations in the empirical 5% tail.

## Interpretation

The full-universe result is a tail-risk result, not a return result. With all
379 stocks, WPF increased average loss by 0.001761 and reduced terminal wealth,
but reduced CVaR by 0.011303. Because CVaR receives 90% of the objective weight,
that tail improvement produces the reported 13.09% lower risk-adjusted cost.

On the 100-stock subsets, WPF did the opposite on average: it improved mean
loss by 0.001228 and increased terminal wealth by 0.1068, but worsened CVaR by
0.015669. The smaller feasible universe does not provide enough breadth for
WPF's probability reweighting to retain the full-universe tail-risk benefit.
Meanwhile SAA has much less opportunity to overfit when the asset dimension
falls from 379 to 100.

Using all 379 stocks to construct WPF distances while restricting investments
to each subset made results worse, not better. Loss of market-regime information
therefore does not explain the subset failure. The benefit requires the broad
portfolio feasible set itself.

## Grid sensitivity

The rolling selection is extremely sensitive to the lambda grid:

- All 379 stocks: eight finite grid points gave `+49.31%`; the original 30
  points gave `-13.09%`.
- 100-stock seed 1: eight finite grid points gave `-11.81%`; the original 30
  points gave `+18.15%`.

This is consistent with tuning a 5%-CVaR criterion over only 24 months: the
tuning tail contains 1.2 effective observations. Small changes to the candidate
grid can change which historical loss becomes decisive.

### Full-universe grids denser than 30

The full 379-stock experiment was also run on strictly nested logarithmic
grids, `30 ⊂ 59 ⊂ 117`. Every lambda in a coarser grid is retained by the
next grid.

| Finite grid points | Difference from SAA | WPF cost | Final lambda |
|---:|---:|---:|---:|
| 30 | -13.09% | 0.066374 | 1.37382 |
| 59 | -12.35% | 0.066937 | 1.37382 |
| 117 | -12.36% | 0.066932 | 1.37382 |

Increasing density beyond 30 therefore does not improve the full-universe
result. The 59- and 117-point costs differ by only 0.0000052, indicating that
the risk-adjusted result has converged by 59 points. The 30-point grid is
slightly better out of sample even though all three grids select the same final
lambda; denser grids sometimes choose added candidates during earlier rolling
stages.

### 100-stock subsets with 59 points

The ten seeded 100-stock subsets were rerun with the strictly nested 59-point
grid. Six individual subsets improved relative to the 30-point grid, but the
aggregate result became slightly worse because seed 7 changed from a WPF win to
a large loss.

| Seed | 30 points | 59 points | Change from added density |
|---:|---:|---:|---:|
| 1 | +18.15% | +12.72% | -5.43 pp |
| 2 | -3.09% | -6.76% | -3.66 pp |
| 3 | +25.39% | +32.51% | +7.12 pp |
| 4 | +32.82% | +26.51% | -6.31 pp |
| 5 | +19.08% | +10.37% | -8.71 pp |
| 6 | +51.83% | +39.87% | -11.96 pp |
| 7 | -18.48% | +29.67% | +48.15 pp |
| 8 | +91.13% | +91.20% | +0.07 pp |
| 9 | -16.46% | -20.32% | -3.86 pp |
| 10 | -3.22% | -1.09% | +2.13 pp |

| Aggregate | 30 points | 59 points |
|---|---:|---:|
| WPF wins | 4/10 | 3/10 |
| Mean difference from SAA | +19.72% | +21.47% |
| Median difference from SAA | +18.62% | +19.62% |
| Mean absolute cost difference | +0.013979 | +0.015135 |
| Mean CVaR difference | +0.015669 | +0.016842 |
| Mean wealth difference | +0.1068 | +0.0007 |

Thus the additional lambda choices reduce several losses but do not improve
the overall 100-stock result. The large seed-7 reversal is consistent with
hyperparameter-selection overfitting in the 24-month rolling tuning window.

## Artifacts

- `test-stock-return-subsets.jl`: reproducible experiment runner.
- `stock-subset-results-grid30.csv`: primary ten-subset results.
- `stock-subset-results-full-signal-grid30.csv`: full-signal diagnostic.
- `stock-subset-results.csv`: exploratory eight-point grid sensitivity runs.
- `stock-full-grid-density-summary.csv`: nested 30/59/117-point comparison.
- `stock-full-grid59.csv` and `stock-full-grid117.csv`: detailed dense-grid runs.
- `stock-subset-results-grid59.csv`: ten detailed 59-point subset runs.
