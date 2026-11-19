# Independent MLP and attention results, 2018 cutoff

[Protocol](directional_three_seed_prereg.md). Three frozen backbones with seed 42/43/44 and selected epochs 3/2/6. The MLP heads selected steps 52/34/27 and the attention heads selected steps 7/11/16 by validation BCE. Test covers 283 anchors from 2021-01-08 through 2026-10-09. Common old-score candidate ranks use the three single-checkpoint scores standardized per week. The older multi-epoch frozen baseline uses a different candidate and execution rule.

| Model | Annualized return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| MLP seed 42 | 19.24% | 1.34 | 13.76% |
| MLP seed 43 | 11.70% | 1.04 | 10.82% |
| MLP seed 44 | 16.56% | 1.27 | 12.98% |
| MLP ensemble | 16.34% | 1.21 | 11.94% |
| Attention seed 42 | 22.49% | 0.86 | 29.08% |
| Attention seed 43 | 37.76% | 1.26 | 27.45% |
| Attention seed 44 | 23.83% | 0.90 | 33.34% |
| Attention ensemble | 34.75% | 1.19 | 30.01% |
| Same attention candidates, always long | 23.77% | 0.93 | 31.51% |
| Same attention candidates, always short | -3.21% | 0.07 | 54.13% |
| Same attention candidates, static 50/50 | 13.70% | 1.07 | 17.36% |
| Same attention candidates, static 72.4/27.6 | 19.23% | 1.24 | 16.28% |

The independent attention ensemble takes a pure long position in 205 weeks and a pure short position in 78. Relative to always long, its paired net improvement is 17.31 bps/week (Newey-West t 0.85), with year differences in bps/week of -2.3, +0.4, -23.6, +108.1, -53.7 and +94.2. Relative to the retrospective 72.4/27.6 exposure-matched control, the difference is 29.28 bps/week (t 1.82), with worse drawdown and slightly lower Sharpe. The MLP ensemble averages 31.58 long names and 63.2% long notional. Both methods have the same target gross exposure and cost rates. Short borrowing is an eligibility-list proxy and fee assumption; actual borrow, executable pure-short capacity and fills have not been verified.

Files: `outputs/directional_2018/three_seed/weekly.csv`, `scores.csv`, `mlp_stocks.csv`, `report.json`, and six selected head checkpoints. Each head was fitted on pre-2018 training data and selected on 2018–2020 validation. These backbones also use that validation interval for their existing checkpoint selection. The historical test interval has been examined in earlier work.
