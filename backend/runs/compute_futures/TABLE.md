# H100 compute-futures hedging — auto table (N=100)

## A. Hedge sweep — net training-run cost by hedge ratio

| hedge ratio | mean $ | median $ | p10 $ | p90 $ | std $ | var reduction vs h=0 |
|---|---|---|---|---|---|---|
| 0.00 | 360 | 347 | 244 | 492 | 99.1 | 0.0% |
| 0.25 | 360 | 350 | 273 | 459 | 74.4 | 43.8% |
| 0.50 | 360 | 353 | 302 | 426 | 49.6 | 75.0% |
| 0.75 | 360 | 356 | 331 | 393 | 24.8 | 93.7% |
| 1.00 | 360 | 360 | 360 | 360 | 0.0 | 100.0% |

Full hedge (h=1.0) variance reduction vs unhedged: **100.0%**.

## B. Population robustness

- matched/seed: mean 96.6
- settled/seed: mean 81.4
- min balance over all seeds: 0.00 (>=0 required)
- max zero-sum err: 0.00e+00

## C. Naive scripted hedger

- naive hedger final-coin variance reduction: -36101.6% (expected ~0 / negative — under-hedged)

_Auto-generated. Narrative interpretation lives in FINDINGS.md._