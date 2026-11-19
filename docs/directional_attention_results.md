# Joint direction classifier and discrete attention results

The experiment follows [directional_attention_prereg.md](directional_attention_prereg.md). It uses the fixed 2021-vintage encoder, 2023 onward out-of-sample MLP candidate predictions, annual refitting and a two-week label gap. The 2024-01-05 to 2026-10-09 evaluation has 139 weekly observations.

The joint MLP selects the 50 highest predicted directional-win probabilities from up to 50 long and 50 eligible short candidates. Over the evaluation it selected an average of 2.42 long names, median 2. In 21 of 139 weeks it selected no long names. Mean selected confidence was 0.548 for long and 0.576 for short. Scores have not been calibrated as probabilities across the two sides.

| Book | Annualized return | Sharpe | Max drawdown | Δ bps/week vs fixed 50/50 |
|---|---:|---:|---:|---:|
| Global softmax across all selected stocks | -10.18% | -0.194 | 55.63% | -25.55 |
| Same candidates, equal within each leg, 50/50 | 6.38% | 0.498 | 23.17% | -0.26 |
| Same candidates, MLP weights, fixed 50/50 | 6.52% | 0.507 | 23.23% | 0 |
| MLP weights, fixed 60/40 | 9.25% | 0.643 | 24.84% | +5.19 |
| MLP weights, fixed 40/60 | 3.51% | 0.298 | 23.32% | -5.19 |
| MLP weights, discrete cross-attention | 5.42% | 0.414 | 22.90% | -1.69 |
| MLP weights, discrete mean-pooling | 6.58% | 0.479 | 26.33% | +0.49 |
| Old-score 40-long/10-short proxy, equal within legs | 18.93% | 1.516 | 13.42% | +20.63 |

The full 50% long allocation stays cash when no long stocks are selected. The global softmax book's average long weight is 4.71%, versus 42.45% for the nominal fixed-50/50 book. Thus the global and 50/50 comparison includes a large exposure change. The old-score book changes both membership and exposure and is a contextual baseline.

The attention allocator predicts three logits with a learned token, cross-attention and FFN. Counts of selected 60/40, 50/50 and 40/60 are 92, 12 and 35 weeks. Classification accuracy against the registered ex-post labels is 43.9%. The 50/50 class is assigned when the two extreme books' future return difference is within 10 bps, an experiment choice to give the middle class a tolerance. Those outcomes do not include action-specific turnover in their class labels; backtest results include actual simulated turnover.

The proxy charges 10 bps per traded notional and an assumed 8% annualized short holding fee. Ten selected stock-week returns are missing and set to zero. The short eligibility list is an exchange-data proxy, and execution, borrow quantities, recalls, and historical point-in-time universe are incomplete. These are historical, previously observed dates and a single encoder/seed. Results establish neither deployability nor cross-side probability calibration.

Machine-readable details include `selected_stocks.csv`, `allocation_predictions.csv`, `weekly.csv`, and `report.json` under `outputs/score_mlp_experiment/directional_discrete/`.
