# Multi-leg and same-week context comparisons

The [protocol](attention_context_protocol.md) fits the portfolio heads before 2018, selects steps on 77 complete-label weeks in 2018–2019, and reports a separate 2020 development check. The frozen backbone checkpoints had already used 2018–2020 for selection. The 2021–2026 results cover 283 previously examined historical weeks. Allocation weights are 70% long on positive scores and 40% long otherwise.

| Portfolio arm | Selection Sharpe | 2020 Sharpe | Historical annualized net | Historical Sharpe | Historical max drawdown |
|---|---:|---:|---:|---:|---:|
| Mean-pool BCE, matched retraining | 1.42 | 0.79 | 12.85% | 0.89 | 19.71% |
| 32-weight cross-leg interaction | 1.34 | 0.79 | 10.36% | 0.72 | 19.71% |
| BCE plus lower-quartile auxiliary | 1.49 | 0.89 | 12.31% | 0.88 | 17.21% |

The interaction starts with identical logits and receives nonzero gradients. Its 2020 decisions and net returns are identical to the matched BCE control. The lower-quartile auxiliary improves 2020 direction accuracy from 43.18% to 59.09%, while paired net return is −0.45 bps/week. Both candidates fail the predefined development gate. Historical comparison confirms neither as a robust improvement. The BCE control's historical annualized return changes from the earlier 22.28% to 12.85% when its validation selection window ends in 2019. The direction policy and candidate universe are fixed, so the checkpoint choice explains this difference.

| Stock scoring arm | Historical weekly RankIC | Historical Top-40 long annualized net | Historical Sharpe |
|---|---:|---:|---:|
| Frozen score | 0.0866 | 23.75% | 0.93 |
| Own-stock ridge residual | 0.0881 | 25.16% | 0.97 |
| Same-week sector and market residual | 0.0890 | 24.57% | 0.95 |

Versus the own-stock residual, context adds 1.56 and 0.77 bps/week in the two development slices, then −0.86 bps/week in history with 95% interval [−3.95, +1.88]. It uses same-week leave-one-out means of available frozen embeddings with retrospective industry membership. This is a score-head probe. It changes no temporal attention blocks. The stock table is a long-only replay and the portfolio table is long/short.

The prior MSE was selected by validation MSE, with no Sharpe-based checkpoint search. Its 2.10 validation Sharpe drops to 1.07 in the historical test. It beats the original BCE by 9.50 bps/week on validation and trails by 9.01 bps/week in history. Both paired 95% intervals cross zero. Its correlation with realized gap falls from 0.102 to 0.021, and the validation advantage is +13.43, +15.73 and +0.20 bps/week across 2018, 2019 and 2020. Removing the largest absolute weekly difference leaves +5.47 and −7.32 bps/week. In the historical test, static books matched to realized average exposure attribute −2.45 bps/week to lower average long weight and leave −6.56 bps/week from remaining timing and turnover differences. This decomposition uses ex post weights for attribution.

An additional diagnostic refit MSE on the same early selection window after the primary comparisons were fixed. The chosen steps are 22, 8 and 7 instead of the previous 58, 8 and 7. Sharpe is 2.55 in 2018–2019, 1.24 in 2020 and 1.05 in history. Historical net is +4.68 bps/week against the early-selected BCE, with 95% interval [−9.50, +19.66]. The historical regression correlation stays near 0.021. The different checkpoint window reverses the relative model ordering, and the results do not establish a persistent advantage for either loss.

Artifacts in `outputs/directional_2018/three_seed/followup/context_ablation/` include `portfolio_weekly.csv`, `stock_weekly.csv`, `mse_early_weekly.csv` and `report.json`. The current runner uses the [full validation protocol](attention_full_validation_protocol.md). Borrow and execution costs remain historical proxies.
