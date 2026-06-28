# H100 hedge through the order book — auto table (N=100)

Unhedged revenue std (baseline): $99.1. Ideal full hedge at fair F: $360.

| fwd depth (×Q) | coverage | hedged mean $ | p10–p90 $ | std $ | var reduction | slippage $ | slippage % |
|---|---|---|---|---|---|---|---|
| 0.25 | 25% | 354 | 268–453 | 74.4 | 43.8% | 5.4 | 1.50% |
| 0.5 | 50% | 349 | 291–415 | 49.6 | 75.0% | 10.8 | 3.00% |
| 1 | 100% | 338 | 338–338 | 0.0 | 100.0% | 21.6 | 6.00% |
| 2 | 100% | 347 | 347–347 | 0.0 | 100.0% | 13.0 | 3.60% |
| 4 | 100% | 351 | 351–351 | 0.0 | 100.0% | 8.6 | 2.40% |

_Auto-generated. Narrative interpretation lives in FINDINGS.md._