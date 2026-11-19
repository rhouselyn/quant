# 2018-cutoff direction and allocation results

Protocol: [directional_2018_prereg.md](directional_2018_prereg.md). Frozen `abl2000_base/best.pt`, seed 42, was trained through 2017 and itself selected using 2018–2020 validation. The added MLP and cross-attention heads trained on pre-2018 anchors, selected their training steps on 2018–2020 validation, and were frozen for the 2021-01-08 through 2026-10-09 test. A two-anchor-week label gap precedes each boundary. Effective allocation samples: 766 training weeks and 146 validation weeks. Model selection is based on validation BCE, not test return.

MLP validation chose step 51, BCE 0.69232. Cross-attention's single binary logit chose step 11, BCE 0.68926. Score > 0 chooses 60/40 long/short; score <= 0 chooses 40/60. On 283 test weeks, it selected 60/40 in 219 weeks and 40/60 in 64, with 50.18% accuracy against the ex-post leg-return label. The test candidate set averaged 35.11 longs out of 50.

| Test book | Annualized return | Sharpe | Max drawdown | Δ bps/week vs fixed 60/40 |
|---|---:|---:|---:|---:|
| Joint top-50 global softmax | 17.24% | 1.126 | 15.84% | -2.43 |
| Same candidates fixed 60/40 | 19.19% | 1.479 | 12.44% | 0 |
| Same candidates fixed 40/60 | 15.48% | 0.981 | 20.41% | -5.09 |
| Single-score cross-attention | 16.86% | 1.271 | 13.14% | -3.67 |
| Old-score 40/10 equal proxy, 50/50 | 9.66% | 0.802 | 20.16% | -16.03 |

The last row also changes candidate membership and exposure. The attention difference vs fixed 60/40 by anchor year is -9.36, -2.43, +0.33, -2.31, -9.23 and +3.63 bps/week in 2021–2026. Test intervals were examined in previous project research and are not fresh unseen evidence. Both baseline checkpoint and new-head selection use the same validation years. The backbone was frozen and had seen pre-2018 training labels; new heads fit on that same interval.

Before 2018 historical short eligibility is unavailable and the training proxy assumes all selected short names are eligible. Post-2018 previous-quarter public lists are used; they do not prove actual availability. Returns use 10 bps traded notional and an assumed 8% annual short fee. Eleven selected stock-week returns were missing and set to zero. No actual borrow, entry/exit fills, recalls or point-in-time stock universe are established.

The selected head states and normalization are in `outputs/directional_2018/selected_heads.pt`. Weekly anchor-level artifacts `weekly.csv`, `stocks.csv`, `attention_scores.csv` and `report.json` are in the same directory.
