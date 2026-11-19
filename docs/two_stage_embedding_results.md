# Projection learning rate and two-stage stock encoder results

[Protocol](two_stage_embedding_protocol.md). The allocation replay uses the same 133 validation and 283 historical test weeks, long and short baskets, and fixed allocations as the previous frozen-projection study.

| Allocation head | Validation Sharpe | Test annualized net | Test Sharpe | Test max drawdown |
|---|---:|---:|---:|---:|
| Joint BCE | 1.85 | 22.28% | 1.54 | 12.97% |
| Freeze stock projection after step 10 | 2.16 | 24.92% | 1.69 | 12.16% |
| Reduce stock projection learning rate to 0.1× after step 10 | 2.09 | 21.27% | 1.46 | 12.97% |

The reduced-rate arm selects steps 100, 13 and 52 for seeds 42, 43 and 44. The first ten validation loss values reproduce joint BCE. Test return averages −1.55 bps/week versus joint BCE, with an 8-week paired block-bootstrap 95% interval [−9.96, +4.90]. It averages −5.68 bps/week versus frozen projection, with interval [−12.72, −0.51]. This is one preset learning-rate ratio on a reused historical period.

The stock experiment fits before 2018, selects ridge penalties on 146 validation weeks and compares on 283 historical test weeks. Both pretraining objectives use 512 updates from the same seed-42 initialization. After pretraining, the temporal encoder is frozen and a fresh linear ridge predicts the weekly cross-sectional return rank. All stock arms use the same Top-40 long-only replay, entry proxy, and 10 bps turnover assumption.

| Stock representation and scoring head | Val Sharpe | Test RankIC | Test annualized net | Test Sharpe |
|---|---:|---:|---:|---:|
| Existing jointly trained baseline score | 1.09 | 0.0823 | 22.96% | 0.90 |
| Existing supervised embedding, new ridge | 0.91 | 0.0863 | 20.27% | 0.84 |
| Random frozen encoder, new ridge | 0.84 | 0.0871 | 18.24% | 0.83 |
| Masked reconstruction, then freeze | 0.79 | 0.0922 | 18.01% | 0.82 |
| Masked same-stock contrast, then freeze | 0.77 | 0.0861 | 14.12% | 0.68 |
| Raw latest row, linear ridge | 0.38 | 0.0795 | 5.42% | 0.34 |

Relative to the same random encoder and probe, reconstruction adds −0.22 bps/week in test, interval [−7.47, +7.23]. Masked contrast adds −6.78 bps/week, interval [−13.27, +0.41]. The latter's average training loss moves from 1.309 in the first 128 updates to 0.133 in the last 128. That improvement did not carry into the rank probe. The existing supervised embedding refit replaces the original score head and learning objective together. Its result does not isolate the effect of freezing from the change in rank-head supervision.

The baseline already uses an EMA same-stock future embedding in its joint InfoNCE objective. A future-latent pretraining stage, a positive-pair definition based on future co-movement, and low-rate partial unfreezing are distinct candidates for further work. Recent [equity-market JEPA work](https://proceedings.mlr.press/v306/mahns26a.html) found stronger representation of volatility and correlation structure than market direction. [FASCL](https://arxiv.org/abs/2602.10711) uses soft supervision from future cross-asset correlations, and [TimeMAE](https://ustc-time-series.github.io/TimeMAE/) reconstructs representations of masked time patches. Those tasks differ from cost-aware weekly stock ranking.

Reproduce with `python scripts/leg_projection_lr.py` and `python scripts/two_stage_stock_embedding.py`. The latter stores the encoder checkpoints, ridge coefficients and weekly outputs under `outputs/directional_2018/two_stage_embedding/`. The pretraining screen is single-seed and short-budget. The test window has already been examined in other research.
