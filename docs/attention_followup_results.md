# Independent attention follow-up

The [protocol](attention_followup_protocol.md) uses the existing three frozen 2018-cutoff backbones and 2021-01-08 to 2026-10-09 historical test. All comparisons share 283 test weeks. The execution rule sweep uses 133 eligible validation weeks, 123 of which also have all stock-level labels for model selection and accuracy. The scores and original test curve reproduce the saved attention run.

| Test arm | Annualized net | Sharpe | Max drawdown | Direction accuracy, complete weeks |
|---|---:|---:|---:|---:|
| Frozen Top-20 / Bottom-10 | 21.47% | 1.53 | 21.83% | |
| Frozen Top-40 / Bottom-10 | 21.57% | 1.59 | 19.80% | |
| Attention candidates held 50/50 | 13.70% | 1.07 | 17.36% | |
| Original single-query attention | 34.75% | 1.19 | 30.01% | 56.88% |
| Return-gap-weighted BCE | 20.80% | 0.79 | 34.35% | 54.71% |
| Previous-target cost-aware attention | 22.76% | 0.86 | 39.01% | 55.80% |
| Mean pooling by leg | 35.96% | 1.25 | 24.88% | 56.88% |
| Two queries by leg | 28.52% | 1.00 | 37.24% | 57.61% |

The frozen Top-40 curve changes only the long count in the historical comparator. It keeps the buffered short leg, liquidity pool, historical borrow proxy and fee assumption. Original attention averages +26.14 bps/week versus the Top-40 comparator. Its 8-week paired block-bootstrap 95% interval is −13.19 to +61.63 bps/week. Mean pooling averages +1.21 bps/week versus the original, interval −18.57 to +24.14. Validation Sharpe is 1.31 for mean pooling and 1.32 for the original. The cost-aware arm uses estimated prior target overlap for the decision, then charges drift-adjusted turnover in the replay.

| Validation-selected rule | Outer long allocations | Neutral half-width | Val Sharpe | Test annualized net | Test Sharpe | Test max drawdown |
|---|---|---:|---:|---:|---:|---:|
| Original policy | 100%, 0% | 0 | 1.32 | 34.75% | 1.19 | 30.01% |
| Threshold only | 100%, 0% | 0.13088 | 1.54 | 28.47% | 1.18 | 23.58% |
| Weights only | 70%, 40% | 0 | 1.84 | 21.97% | 1.47 | 13.33% |
| Joint selection | 70%, 20% | 0.13088 | 1.92 | 21.35% | 1.33 | 14.00% |

The symmetric band chosen on validation covers 24.8% of validation weeks and 35.0% of test weeks. Threshold-only Sharpe rank correlation between validation and test is −0.05, and validation bootstrap selection frequency for the exact chosen threshold is 12.5%. At zero threshold, the 70% and 40% long weights win the validation and retrospective test Sharpe rankings. The exact cell wins 21.6% of validation block resamples. The neighboring 60% and 40% pair wins 44.1%, and the 80% and 40% pair yields a test Sharpe of 1.46 versus 1.47 at 70% and 40%. The weights-only winner averages +1.01 bps/week versus frozen Top-40, interval −15.62 to +19.54. The joint winner repeats in 1.3% of validation resamples. The frozen test period has been previously explored, and the reported intervals apply to the fixed paired policies rather than correcting for the grid search.

Validation and test logits have means 0.152 and 0.157 with standard deviations 0.262 and 0.258. Validation skew is −0.81 and test skew is −0.09. Across low, medium, and high absolute score bins defined on validation, test direction accuracy is 54.1%, 60.5%, and 57.0%, respectively, with 111, 86, and 79 full-label weeks. A symmetric threshold centered on the zero decision boundary is a policy choice. The observed logits do not give a transferable monotone confidence scale.

Outputs are `outputs/directional_2018/three_seed/followup/aligned_test.csv`, `allocation_grid.csv`, `ablation_weekly.csv`, and `report.json`. Historical execution, borrow availability, and capacity remain proxies.
