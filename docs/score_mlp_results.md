# Fixed-vintage score MLP and allocation diagnostics

Registration: [score_mlp_prereg.md](score_mlp_prereg.md). The frozen encoder is seed 42, epoch 3, from `abl2000_base_r2021`. The new heads fit only after the encoder training boundary and are refit each calendar year, with a two-anchor-week maturity gap. The evaluation has 139 weekly observations, 2024-01-05 through 2026-10-09. This is a previously examined historical period, with the present-day retrospective stock universe.

## Candidate-weighting results

All numbers are changes in net bps per week against equal weights in the identical weekly candidate set. Legs are separately normalized to one unit of notional. The MLP and old-score arms mix 80% equal weight and 20% softmax. Net returns use 10 bps per unit traded notional; the short leg also pays an assumed 8% annual fee.

| Arm | Long Δ bps/week | Long NW t | Short Δ bps/week | Short NW t |
|---|---:|---:|---:|---:|
| Old score | +0.57 | 0.54 | +3.86 | 0.61 |
| MLP absolute | -1.21 | -2.83 | -1.94 | -0.22 |
| MLP within-leg relative | -0.66 | -1.68 | -0.90 | -0.14 |

The relative-label arm exceeds the absolute-label arm by 0.55 bps/week on the long leg and 1.04 bps/week on the short leg. Neither beat its own equal-weight control in this configuration. The absolute and relative long arms have average effective counts 39.73 and 39.82, respectively, versus 40 equal weight. Short effective counts are 9.47 and 9.69 versus 10.

Short results are especially sensitive to individual weeks. The relative arm differs from equal weight by -457.46 bps on 2026-04-10, when 002580.SZ's modeled simple holding-period return was +72.64% (the weekly cache shows entry open 82.24 and exit open 141.98). Dropping that week changes the relative short result from -0.90 to +2.40 bps/week, and the absolute short result from -1.94 to +2.17. These are post-hoc sensitivity checks.

## Candidate-set allocation

Equal weights within each leg. A 32-dimensional mean-pooling network and a learned-query attention network predict long-minus-short return, with a pre-registered mapping to 40%–60% long exposure. The complement is short. Both use earlier matured weeks and annual refitting.

| Arm | Net Δ bps/week vs 50/50 | NW t | Mean long allocation | Long allocation std |
|---|---:|---:|---:|---:|
| Mean pooling | -0.34 | -0.43 | 50.01% | 1.13 pp |
| Attention | +0.38 | 0.27 | 50.27% | 1.67 pp |

Attention exceeds mean pooling by 0.72 bps/week, NW t 0.69. Attention's 2024/2025/2026 relative changes against 50/50 are +1.14/-1.16/+1.50 bps/week. A static allocation retrospectively matched to attention's average exposure gives +0.09 bps/week against 50/50. These results do not establish a stable benefit from dynamic allocation.

## Interpretation and limits

Candidate selection uses past scores and the preceding quarter's exchange-eligibility proxy. The Shanghai list is a financing-activity proxy and does not establish actual borrow availability or quantity. Candidate membership never uses realized returns. Five long and two short selected stock-week returns are missing and filled as zero, identically across weighting arms. Entry availability, price limits, recall, actual borrow quotes, intraperiod actions, and the current live buffered strategy are not reproduced. Annual fee and 10 bps costs are fixed assumptions. Label construction inherits the existing weekly cache's open/close fallback. All findings are single-encoder, single-seed diagnostics with fixed hyperparameters, and historical performance is not an independent prospective validation.

Detailed per-week results and machine-readable summaries are under `outputs/score_mlp_experiment/`. Scripts are `scripts/score_mlp_experiment.py` and `scripts/score_attention_experiment.py`.
