# Objective O4 Report: Joint End-to-End Neural CQR vs Post-Hoc CQR (§3 O4 & §4.2)

## Research question
Does genuine joint end-to-end Neural CQR provide an advantage over the intended post-hoc approach?

## What actually differs between the two arms
| | Post-hoc (Condition A) | Joint (Condition B) |
|---|---|---|
| Training paradigm | `post_hoc_two_stage` | `joint_end_to_end` |
| Optimization stages | 2 | 1 |
| Stage 1 loss terms | `['huber']` | `['huber', 'pinball_05', 'pinball_50', 'pinball_95', 'crossing', 'width']` |
| Stage 2 loss terms | `['pinball_05', 'pinball_50', 'pinball_95', 'crossing']` | `None` |
| Quantile gradients reach backbone | False | True |
| Backbone frozen for interval fitting | True | False |

**Distinctness check**: PASS — max absolute difference in test point
predictions between arms = `1.215924`.

## Held constant across both arms
- identical FIT/DEV/CAL/TEST partitions
- identical feature set and fitted scaler
- identical architecture and capacity
- identical optimizer, lr, weight decay, batch size
- identical epoch budget and early-stopping patience on identical DEV data
- identical random seed
- identical conformal calibration applied to both arms

Only difference: **training paradigm (joint end-to-end vs post-hoc two-stage)**.

## Point prediction
| Metric | Post-hoc | Joint | Absolute Δ (joint − post-hoc) | Relative Δ | Joint better? |
|---|---:|---:|---:|---:|---|
| RMSE | 0.9629 | 0.963 | 0.0001 | 0.01% | False |
| MAE | 0.7397 | 0.7445 | 0.0048 | 0.65% | False |
| R² | 0.6773 | 0.6773 | 0.0 | 0.0% | False |

- Squared-error difference: mean difference (joint − post-hoc) = `0.000118`, 95% CI `[-0.05389, 0.053183]` (year-block bootstrap, 5 years, p = `0.959`); year-level paired t p = `0.969886`
- Absolute-error difference: mean difference (joint − post-hoc) = `0.004776`, 95% CI `[-0.017567, 0.024813]` (year-block bootstrap, 5 years, p = `0.66`); year-level paired t p = `0.77901`

## Uncertainty — static split conformal (identical calibration on both arms)
| Metric | Post-hoc | Joint | Absolute Δ | Relative Δ | Joint better? |
|---|---:|---:|---:|---:|---|
| PICP | 0.965 | 0.9539 | -0.0111 | -1.15% | False |
| MPIW | 4.7056 | 3.7938 | -0.9118 | -19.38% | True |
| ACE | 0.065 | 0.0539 | -0.0111 | -17.08% | True |
| Winkler | 5.1018 | 4.3066 | -0.7952 | -15.59% | True |

- Winkler difference: mean difference (joint − post-hoc) = `-0.79516`, 95% CI `[-0.925282, -0.6586]` (year-block bootstrap, 5 years, p = `0.0005`); year-level paired t p = `0.000411`
- Interval-width difference: mean difference (joint − post-hoc) = `-0.911852`, 95% CI `[-1.008953, -0.825603]` (year-block bootstrap, 5 years, p = `0.0005`); year-level paired t p = `5.9e-05`

## Uncertainty — ACI (identical calibration on both arms)
| Metric | Post-hoc | Joint | Absolute Δ | Relative Δ | Joint better? |
|---|---:|---:|---:|---:|---|
| PICP | 0.9501 | 0.9369 | -0.0132 | -1.39% | False |
| MPIW | 4.3963 | 3.5756 | -0.8207 | -18.67% | True |
| ACE | 0.0501 | 0.0369 | -0.0132 | -26.35% | True |
| Winkler | 4.9417 | 4.2239 | -0.7178 | -14.53% | True |

- ACI Winkler difference: mean difference (joint − post-hoc) = `-0.7178`, 95% CI `[-0.862223, -0.582722]` (year-block bootstrap, 5 years, p = `0.0005`); year-level paired t p = `0.001045`
- ACI width difference: mean difference (joint − post-hoc) = `-0.820766`, 95% CI `[-0.946123, -0.713005]` (year-block bootstrap, 5 years, p = `0.0005`); year-level paired t p = `0.000286`

## Verdict
**Partially supported — joint training improves interval quality only**

- Point-prediction improvement supported by the data: **False**
- Interval-quality improvement supported by the data: **True**

A metric counts as improved only when the joint arm is better *and* the
dependence-aware 95% confidence interval for the paired difference excludes
zero. With 5 test
years the unit of independence is the year, so this comparison has limited
power; a non-significant result is not evidence of equivalence.
