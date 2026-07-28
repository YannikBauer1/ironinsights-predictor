# v719 benchmark — 100 random events (2026-07-28)

Production weights from `config/algorithm_state.json` (version 719).

| Metric | Value |
|--------|-------|
| Events processed | 99 |
| Divisions validated | 291 |
| Winner accuracy | **36.8%** |
| Top 3 | **58.8%** |
| Top 5 | **67.0%** |
| Top 10 | **78.1%** |
| Exact match | **11.6%** |
| MAE | **3.5 places** |
| Composite score | **32.50** |

Full per-division breakdown: `reports/benchmarks/validation_20260728_124152.json`

Run the same benchmark:

```bash
python validator/cli.py --count 100 --save
```
