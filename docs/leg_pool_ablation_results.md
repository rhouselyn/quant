# Leg pooling and return-gap ablation results

The [protocol](leg_pool_ablation_protocol.md) uses the same three frozen encoders, candidate stocks, cost assumptions and fixed positive and nonpositive long weights of 70% and 40%. All test rows cover the same 283 weeks. Training and model selection use 473 and 123 complete-label weeks. The saved mean-pool BCE control reproduces the prior 70% and 40% policy to floating-point tolerance.

| Arm | Validation Sharpe | Test annualized net | Test Sharpe | Test max drawdown | Test direction accuracy |
|---|---:|---:|---:|---:|---:|
| Mean pool, BCE | 1.85 | 22.28% | 1.54 | 12.97% | 56.88% |
| Per-stock GELU before mean | 2.08 | 16.23% | 1.09 | 14.93% | 51.09% |
| Add projected leg standard deviation | 2.05 | 21.02% | 1.41 | 12.97% | 56.52% |
| Mean pool, return-gap MSE | 2.10 | 16.44% | 1.07 | 21.14% | 53.26% |
| Mean pool, return-gap Huber | 1.72 | 18.60% | 1.25 | 19.17% | 53.99% |

The paired test mean differences versus BCE are −9.47, −1.86, −9.01, and −5.68 bps per week in table order. The respective 8-week block-bootstrap 95% intervals are [−18.56, +0.38], [−9.39, +3.86], [−21.96, +3.13], and [−16.95, +4.37] bps per week. Test direction accuracy uses the 276 weeks with complete leg labels.

On complete test weeks, the MSE and Huber forecasts have Pearson correlations of 0.021 and 0.034 with the realized return gap. The BCE control allocates to the positive-score policy in 210 of the 283 test weeks, versus 157 and 160 for MSE and Huber. The loss change affects both the weekly decisions and average exposure. These experiments apply the fixed zero boundary, and an exposure-matched comparison would be needed to decompose those effects.

The frozen Top-40 reference on the same weeks reports 21.57% annualized net, 1.59 Sharpe and 19.80% drawdown. It uses the frozen score and short buffer rules documented in [the prior follow-up](attention_followup_results.md). All returns are historical proxies. The test has been examined in earlier studies and actual borrowing and fills remain unverified.

Run `python scripts/leg_pool_ablation.py` to reproduce. The output directory is `outputs/directional_2018/three_seed/followup/leg_pool_ablation` and contains the weekly curves, per-seed checkpoints and summary JSON.
