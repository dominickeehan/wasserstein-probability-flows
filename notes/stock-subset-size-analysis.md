# WPF performance across random subset sizes

## Design

- Original 30-point finite lambda grid, plus the 0 and `Inf` endpoints.
- Reproducible random permutations using seeds 1 through 5 at every size.
- Subsets are nested within each seed, so the size comparisons are paired.
- Sizes 350 and 150 were extended to seeds 1 through 10 because their first
  five results were ambiguous. Size 100 uses the ten runs already completed.
- Positive WPF-minus-SAA differences mean WPF is worse.

## Average results

The pooled difference compares the mean WPF cost with the mean SAA cost at
each size. The median is included because some subsets produce very large
tail-loss outliers.

| Stocks | Draws | WPF wins | Pooled average difference | Median subset difference | Mean absolute cost difference |
|---:|---:|---:|---:|---:|---:|
| 350 | 10 | 4 | +23.40% | +3.53% | +0.01780 |
| 300 | 5 | 1 | +22.87% | +16.79% | +0.01527 |
| 250 | 5 | 0 | +87.43% | +65.07% | +0.06024 |
| 200 | 5 | 0 | +25.57% | +13.12% | +0.01803 |
| 150 | 10 | 4 | +41.75% | +23.50% | +0.03126 |
| 100 | 10 | 4 | +17.92% | +18.62% | +0.01398 |

For comparison, the single all-379-stock experiment gives `-13.09%`, meaning
WPF is better. None of the random-subset sizes reproduces that result on
average.

## Paired first-five results

These are the exact percentage differences for the same five seeded
permutations at every requested size.

| Seed | 350 | 300 | 250 | 200 | 150 | 100 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -10.62% | +32.61% | +53.11% | +26.20% | +10.96% | +18.15% |
| 2 | -23.27% | +56.84% | +6.94% | +4.11% | -0.07% | -3.09% |
| 3 | +109.38% | +12.72% | +143.94% | +81.19% | -9.79% | +25.39% |
| 4 | +6.93% | +16.79% | +65.07% | +7.62% | +59.06% | +32.82% |
| 5 | -10.53% | -2.55% | +182.93% | +13.12% | +69.30% | +19.08% |

## Interpretation

Size 350 is the transition region. A typical result is close to neutral (the
ten-draw median is +3.53%), and WPF wins four times, but rare tail failures of
+109% and +146% make its average result poor. Therefore, removing only 29
stocks can already destroy the apparent robustness for some compositions.

At 300 stocks and below, WPF generally loses. The relationship is not monotone:
250 stocks performs particularly badly, while 100 and 200 are less bad. This
non-monotonicity is consistent with composition-sensitive portfolio
opportunities and noisy rolling 5%-CVaR tuning, rather than a smooth
dimension-only effect.

The average CVaR difference is positive at every subset size, meaning WPF has
worse tail loss than SAA. At 100 and 150 stocks, WPF improves mean loss on
average, but that return benefit is overwhelmed by worse CVaR. This is the same
mechanism found in the earlier 100-stock investigation.

The effect is not explained only by SJM: every one of the first five 350-,
300-, and 250-stock subsets contains SJM, yet most 300-stock subsets and all
250-stock subsets lose to SAA. The full-universe result depends on broader
portfolio composition and diversification, not merely the presence of one
defensive stock.

## Diagnosis of the 350-stock catastrophic failures

The +109% (seed 3) and +146% (seed 9) results are tuning failures rather than
failures of the candidate family to contain SAA. The `Inf`-lambda testing costs
match separately solved SAA costs with maximum numerical error exactly zero in
both experiments.

The current selector nevertheless chooses a non-SAA lambda in 100% of testing
months. Seed 3 switches eight times among seven parameters; seed 9 switches six
times among five parameters. The selector is minimizing a 5%-CVaR criterion
over 24 observations, so each rolling tuning decision depends on only 1.2
effective tail observations while comparing 32 candidates.

Simple validation-improvement margins of 2%, 5%, 10%, and 20% do not prevent
the failures. The noisy winners look strongly favourable in the tuning window.
A diagnostic persistence rule that requires the same lambda to win two
consecutive windows gives:

| Failure seed | Current selector | 5% margin + persistence 2 | 10% margin + persistence 2 |
|---:|---:|---:|---:|
| 3 | +109.38% | -3.64% | -8.75% |
| 9 | +145.79% | -24.92% | -24.92% |

This is evidence that unstable month-to-month model selection causes the
catastrophes. It is not yet evidence that persistence is the final solution:
the rule was examined after observing these failures and must be evaluated on
all subsets using a predeclared specification. Candidate containment alone
cannot guarantee out-of-sample dominance over SAA; only selecting SAA itself
provides that deterministic fallback.

## Caveats

The subsets overlap, and all runs share the same 60 testing months, so the
subset bootstrap intervals are descriptive rather than independent-sample
confidence intervals. Large variation across seeds is itself an important
finding: the WPF result is highly sensitive to the exact investable universe.

## Artifacts

- `stock-subset-size-summary.csv`: aggregate size comparison.
- `stock-subset-results-size350-grid30.csv` and
  `stock-subset-results-size350-grid30-seeds6-10.csv`.
- `stock-subset-results-size300-grid30.csv`.
- `stock-subset-results-size250-grid30.csv`.
- `stock-subset-results-size200-grid30.csv`.
- `stock-subset-results-size150-grid30.csv` and
  `stock-subset-results-size150-grid30-seeds6-10.csv`.
- `stock-subset-results-grid30.csv`: size-100 results.
- `stock-tuning-guardrail-diagnostics.csv`: exact tuning diagnostics for the
  two 350-stock catastrophic failures.
- `diagnose-tuning-failures.jl`: reproducible guardrail diagnostic.
