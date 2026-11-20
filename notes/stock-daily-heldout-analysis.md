# Search for a daily WPF setting that beats SAA

## Protocol

The previously inspected 2023 pilot cannot support a convincing parameter
search. A longer daily dataset was therefore downloaded before defining this
search:

- Data: 2,010 common daily adjusted returns from January 2018 through December
  2025 for a reproducibly sampled 50-stock signal universe.
- Development period: January 2022 through December 2023.
- Untouched time holdout: January 2024 through December 2025.
- Calibration subset seeds: 1 through 10.
- Held-out subset seeds: 11 through 20.
- Estimation: 126 daily observations.
- Portfolio decisions: monthly, with buy-and-hold weight drift within each
  month.
- Candidate grids: 30 finite tail-specific eta values plus `0` and exact SAA at
  `Inf`.
- WPF signal: common 50-stock distance signal; portfolio optimization is
  performed separately inside each investment subset.

For each tail probability, the development-only search varied:

- rolling tuning-window length;
- the required historical percentage improvement over SAA before selecting a
  finite eta; and
- a persistence requirement of one, two, or three consecutive qualifying
  months.

Lambda decisions were estimated by pooling the prior candidate losses from
calibration seeds 1 through 10. The frozen rule was then applied to seeds 11
through 20 in the untouched 2024--2025 period. Confidence intervals use 2,000
common-time circular block bootstrap repetitions with 63-day blocks, preserving
cross-subset market dependence.

## Held-out results

Positive differences mean the frozen WPF/SAA selector is worse than always
using SAA.

| Investment subset | Tail | Frozen setting `(days, margin, persistence)` | Wins | Mean difference | 95% time-block interval |
|---:|---:|---:|---:|---:|---:|
| 20 stocks | 5% | `(756, 15%, 3)` | 0/10 | 0.00% | exact SAA |
| 20 stocks | 10% | `(126, 0%, 1)` | 1/10 | +1.58% | `[+0.000022, +0.000487]` |
| 20 stocks | 20% | `(504, 15%, 3)` | 0/10 | 0.00% | exact SAA |
| 40 stocks | 5% | `(252, 1%, 1)` | 1/10 | +5.70% | `[+0.000312, +0.001445]` |
| 40 stocks | 10% | `(252, 0%, 3)` | 2/10 | +2.52% | `[+0.000040, +0.000572]` |
| 40 stocks | 20% | `(126, 1%, 3)` | 0/10 | +3.33% | `[+0.000076, +0.000554]` |

Every active frozen rule loses on average. All four nonzero aggregate
differences have bootstrap intervals strictly above zero.

The reversal is particularly clear for the 40-stock 10% problem. Its selected
development setting improved the objective on all 10 calibration subsets, with
an average absolute improvement of `-0.000465`. The same frozen rule then lost
on 8 of 10 held-out subsets, with an average deterioration of `+0.000295`.

## Candidate-family diagnostic

The failure is not only a poor rolling selection rule. For each tail and subset
size, every fixed eta was evaluated retrospectively on the holdout as a
diagnostic:

- 20 stocks, 5%: exact SAA is the hindsight-best fixed candidate.
- 20 stocks, 10%: the hindsight-best finite candidate improves the average
  objective by only `0.0000033`.
- 20 stocks, 20%: the hindsight-best finite candidate improves it by only
  `0.0000047`.
- 40 stocks, 5% and 10%: exact SAA is the hindsight-best fixed candidate.
- 40 stocks, 20%: the hindsight-best finite candidate improves it by only
  `0.0000015`.

These finite-candidate improvements are economically negligible and were found
with future information. No feasible lambda selector could turn this candidate
family into a convincing held-out improvement.

## Conclusion

No honest setting was found in this specification that convincingly beats SAA.
Continuing to adjust tuning windows, margins, or persistence after observing
2024--2025 would turn the exercise into in-sample selection and could not
support the requested claim.

A further attempt must change the candidate model rather than only its selector,
for example by using a longer-horizon weekly estimation distribution, a stronger
diversification constraint, or a materially different investable universe. Such
a change requires a new untouched time period or stock universe for validation.

## Artifacts

- `stock-returns-daily-extended.csv`: 2018--2025 daily data.
- `build-daily-wpf-candidate-cache.jl`: bounded candidate-loss generator.
- `select-and-test-daily-wpf-setting.jl`: development search and frozen holdout
  evaluation.
- `stock-daily-heldout-setting-results.csv`: 20-stock results.
- `stock-daily-heldout-setting-results-size40.csv`: 40-stock results.
