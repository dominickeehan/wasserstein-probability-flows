# Single-stock removal sensitivity of the 379-stock result

## Design

The original 30-point WPF experiment was rerun exactly after removing one
stock. Six removals were deliberately selected from the full-universe testing
portfolios to stress persistent concentration, one-month concentration, and
WPF-versus-SAA allocation disagreement. Four additional removals were selected
with a fixed random seed as controls.

The baseline uses all 379 complete-history stocks and gives WPF a 13.09% lower
risk-adjusted cost than SAA.

## Results

| Removed stock | Selection reason | WPF difference from SAA | Change from baseline |
|---|---|---:|---:|
| None | Baseline | -13.09% | 0.00 pp |
| SJM | Largest mean WPF weight | +16.56% | +29.65 pp |
| NEE | Largest maximum WPF weight | -14.30% | -1.21 pp |
| CME | Largest mean SAA weight | -13.31% | -0.22 pp |
| PGR | Largest maximum SAA weight | -7.70% | +5.39 pp |
| AZO | Largest mean allocation disagreement | -16.78% | -3.69 pp |
| GIS | Largest maximum allocation disagreement | -9.43% | +3.65 pp |
| HON | Random control; unused by both | -13.16% | -0.07 pp |
| PHM | Random control; unused by both | -12.88% | +0.21 pp |
| AIG | Random control; unused by both | -9.94% | +3.15 pp |
| EIX | Random control | -7.91% | +5.18 pp |

Nine of the ten tested removals leave WPF ahead of SAA. Excluding SJM, the
largest absolute change is 5.39 percentage points and the median absolute
change is 3.40 percentage points.

SJM is a material exception: removing it reverses the conclusion from WPF
being 13.09% better to 16.56% worse.

## Why SJM matters

In the full-universe testing portfolios:

- WPF assigns SJM 11.11% on average and as much as 36.89% in one month.
- SAA assigns SJM only 1.45% on average and at most 6.58%.
- Removing SJM changes SAA cost from 0.076370 to 0.076358, essentially zero.
- Removing SJM changes WPF cost from 0.066374 to 0.089006, an increase of
  0.022632.
- WPF's CVaR difference from SAA changes from -0.011303 (better) to +0.013453
  (worse).
- WPF terminal wealth falls from 2.0324 to 1.5939 after the removal.

SJM behaves as a defensive tail hedge in this test period. For example, its
returns were +7.78% in February 2020, +2.11% in May 2022, and +3.50% in
December 2021. These months overlap the adverse months that determine WPF's
5%-CVaR. The full-universe 13% advantage is therefore partly dependent on WPF
identifying and heavily using this single defensive stock.

The final selected lambda remains 1.37382 after every tested removal, so the
SJM reversal is not caused by moving the final lambda.

An additional isolation test removed SJM from the investable portfolio while
retaining the original 379-stock WPF distances and probability weights for
every candidate lambda. WPF was still 16.08% worse than SAA, compared with
16.56% when SJM was also removed from the distance calculation. The difference
is only 0.48 percentage points. The reversal is therefore caused almost
entirely by losing SJM from the portfolio opportunity set, not by perturbing
the Wasserstein geometry.

## Conclusion

The result is stable to ordinary or unused-stock removals, but it is not robust
to the worst allocation-based single-stock removal. The concern is valid: the
headline 13% improvement should not be presented as constituent-insensitive.

The next useful robustness test is to impose a maximum holding such as 5% or
10% and repeat the full-universe experiment. That directly tests whether WPF's
tail-risk benefit survives without relying on a concentrated SJM position.

## Artifacts

- `stock-removal-sensitivity.csv`: exact baseline and ten removal results.
- `stock-removal-influence-ranking.csv`: allocation influence ranking for all
  379 stocks.
- `investigate-stock-removal-sensitivity.jl`: reproducible investigation.
- `stock-removal-full-distance.csv`: SJM opportunity-set isolation result.
- `test-stock-removal-full-distance.jl`: reproducible isolation test.
