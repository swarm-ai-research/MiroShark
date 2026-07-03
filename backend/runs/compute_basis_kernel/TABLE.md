# Basis-hedge through the real order book — auto table (N=100)

Dealer residual = std of realised net P&L across seeds, from real matching + settlement. Analytic = Q·√((1−r_i)·V).

| β | r_i | corr(S,I) | settled/seed | dealer P&L μ | residual realized | residual analytic | rel err |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.04 | 0.20 | 2.0 | 0.000 | 9.714 | 9.714 | 0.0% |
| 0.45 | 0.20 | 0.45 | 2.0 | 0.000 | 8.853 | 8.853 | 0.0% |
| 0.63 | 0.40 | 0.63 | 2.0 | 0.000 | 7.699 | 7.699 | 0.0% |
| 0.8 | 0.64 | 0.80 | 2.0 | 0.000 | 5.948 | 5.948 | 0.0% |
| 0.95 | 0.90 | 0.95 | 2.0 | 0.000 | 3.096 | 3.096 | 0.0% |
| 1 | 1.00 | 1.00 | 2.0 | 0.000 | 0.000 | 0.000 | 0.0% |

_Auto-generated. Narrative interpretation lives in FINDINGS.md._