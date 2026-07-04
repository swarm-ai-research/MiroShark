# Dealer funding-ruin under asymmetric margining — auto table (N=200)

Buffer = dealer free coin ÷ hedge notional (β·Q·F). Ruin = the MTM'd basket hedge is liquidated mid-path. β=1 is economically flat at expiry (pure funding risk).

| β | buffer | ruin freq | net P&L mean | net P&L std | net P&L p10 | ruined-only mean | survivors std |
|---|---|---|---|---|---|---|---|
| 0.8 | 0.05× | 21% | -0.55 | 5.50 | -8.00 | -0.55 | 5.10 |
| 0.8 | 0.1× | 12% | -0.50 | 5.14 | -7.57 | 1.66 | 5.07 |
| 0.8 | 0.2× | 2% | -0.57 | 5.06 | -7.89 | 4.42 | 5.01 |
| 0.8 | 0.4× | 0% | -0.57 | 5.03 | -7.89 | 0.00 | 5.03 |
| 0.8 | 0.8× | 0% | -0.57 | 5.03 | -7.89 | 0.00 | 5.03 |
| 0.8 | 1.6× | 0% | -0.57 | 5.03 | -7.89 | 0.00 | 5.03 |
| 1 | 0.05× | 38% | 0.29 | 3.24 | -2.05 | 0.76 | 0.00 |
| 1 | 0.1× | 24% | 0.07 | 2.64 | -0.23 | 0.30 | 0.00 |
| 1 | 0.2× | 8% | 0.03 | 1.12 | -0.00 | 0.33 | 0.00 |
| 1 | 0.4× | 0% | -0.00 | 0.00 | -0.00 | 0.00 | 0.00 |
| 1 | 0.8× | 0% | 0.00 | 0.00 | -0.00 | 0.00 | 0.00 |
| 1 | 1.6× | 0% | 0.00 | 0.00 | -0.00 | 0.00 | 0.00 |

_Auto-generated. Narrative interpretation lives in FINDINGS.md._