# Full validation and staged freezing results

[Protocol](attention_full_validation_protocol.md). Train on 473 complete-label weeks before 2018. Select checkpoints on 123 complete-label weeks in 2018–2020. Replay 133 validation weeks and 283 test weeks from 2021 to 2026. All portfolio arms share the same candidates, fees and fixed 70% or 40% long weights. The original BCE and MSE weekly net curves reproduce the previous full-validation run.

| Portfolio arm | Validation Sharpe | Test annualized net | Test Sharpe | Test max drawdown |
|---|---:|---:|---:|---:|
| Joint mean-pool BCE | 1.85 | 22.28% | 1.54 | 12.97% |
| 32-weight cross-leg interaction | 1.95 | 21.33% | 1.48 | 12.46% |
| BCE plus quantile auxiliary | 2.09 | 21.10% | 1.44 | 12.97% |
| Gap MSE | 2.10 | 16.44% | 1.07 | 21.14% |
| Freeze stock projection after step 10 | 2.16 | 24.92% | 1.69 | 12.16% |
| Freeze representation after step 10 | 2.07 | 24.56% | 1.68 | 12.97% |

Projection freezing is the validation Sharpe winner. Its paired test mean against joint BCE is +4.13 bps/week, with an 8-week block-bootstrap 95% interval of [−0.52, +8.23]. Representation freezing is +3.54 bps/week with interval [−1.18, +8.08]. The four other candidates are reported for comparison. These are intervals for the fixed paired policies on the reused historical window.

The first ten training and validation loss values are identical across the joint BCE and both freeze schedules for each seed. Frozen parameter tensors remain bitwise unchanged after step 10. Joint BCE selects steps 38, 11 and 23. Projection freezing selects 100, 14 and 67. Representation freezing selects 100, 17 and 100. A best checkpoint at step 100 is at the end of the fixed budget. For seed 42, joint BCE training loss falls from 0.671 at step 38 to 0.598 at step 100, while validation loss rises from 0.686 to 0.757. With the projection frozen, seed 42 validation loss at step 100 is 0.689. The schedules compare jointly fitted models. The global losses cannot identify a universal overfitting onset for an individual block.

| Stock score | Validation RankIC | Validation Top-40 Sharpe | Test RankIC | Test Top-40 long annualized net | Test Top-40 Sharpe |
|---|---:|---:|---:|---:|---:|
| Frozen score | 0.0795 | 1.18 | 0.0866 | 23.75% | 0.93 |
| Own-stock ridge residual | 0.0828 | 1.15 | 0.0881 | 25.16% | 0.97 |
| Same-week sector and market residual | 0.0834 | 1.18 | 0.0890 | 24.57% | 0.95 |

Context adds −0.86 bps/week versus the own-stock ridge on test, with interval [−3.95, +1.88]. This is a frozen-embedding scoring-head comparison. The stock table uses a long-only book, and the portfolio table uses long and short legs. Execution and borrow costs remain historical proxies.

Reproduce with `python scripts/attention_context_ablation.py`. Outputs in `outputs/directional_2018/three_seed/followup/context_ablation_full_val/` include all weekly curves, per-step losses, checkpoints, and summary statistics.

## Projection freeze timing

Each step is one full-batch Adam update on all 473 training weeks. After step `k`, freeze the allocation head's shared 129→32 linear projection and keep the FFN, normalization and output trainable through step 100. Step 0 freezes the randomly initialized projection. Use the same 2018–2020 validation and 2021–2026 test replays, with 10 bps per traded notional and 8% annual borrow. Select each seed's checkpoint by validation BCE and the schedule by validation net Sharpe of the averaged three-seed logits.

| Freeze after step | Validation Sharpe | Test Sharpe | Test annualized net | Selected steps for seeds 42 / 43 / 44 |
|---|---:|---:|---:|---|
| Joint BCE | 1.85 | 1.54 | 22.28% | 38 / 11 / 23 |
| 0 | 1.88 | 1.35 | 20.00% | 100 / 16 / 72 |
| 2 | 1.81 | 1.33 | 19.15% | 100 / 15 / 92 |
| 5 | 2.03 | 1.53 | 22.38% | 100 / 22 / 87 |
| 10 | 2.16 | 1.69 | 24.92% | 100 / 14 / 67 |
| 20 | 2.09 | 1.44 | 21.19% | 100 / 11 / 28 |
| 40 | 1.85 | 1.54 | 22.28% | 38 / 11 / 23 |
| 60 | 1.85 | 1.54 | 22.28% | 38 / 11 / 23 |

Step 10 has the highest validation Sharpe. Its test allocation differs from joint BCE in 14 of 283 weeks and adds 4.13 bps/week with an 8-week paired block-bootstrap 95% interval of [−0.52, +8.23]. The step-40 and step-60 schedules select the same pre-freeze checkpoints as joint BCE and reproduce its weekly returns exactly. Seed 42 selects the step-100 training limit in each schedule from step 0 through step 20. These results use the same historical validation and test periods as the earlier studies.

Reproduce with `python scripts/leg_projection_lr.py --freeze-sweep`. The `projection_freeze_sweep/` subdirectory contains validation schedule selection, weekly returns, loss curves and selected checkpoints.
