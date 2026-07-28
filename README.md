# Iron Insights Predictor

IFBB Pro competition placement predictor and validation suite for [Iron Insights](https://github.com/YannikBauer1).

**Current production version:** v719 (weights in `config/algorithm_state.json`)

## Repository layout

```
ironinsights-predictor/
├── config/                 # Algorithm weights & tuning state
│   └── algorithm_state.json
├── predictor/              # Core prediction engine
│   ├── predictor.py        # Scoring algorithm + head-to-head
│   ├── predict_division.py # Division entry point (live + historical)
│   ├── data_fetcher.py     # Supabase data access
│   └── win_probability.py  # Win-chance calibration
├── validator/              # Historical backtesting
│   ├── validation.py       # Validation pipeline
│   ├── metrics.py          # Accuracy metrics
│   ├── cli.py              # Batch validation CLI
│   ├── run_logger.py       # Per-division logs
│   └── save_results.py     # JSON report export
├── batch/                  # Production batch scoring
│   ├── batch_predict_and_update.py  # Score lineups → Supabase Edge Function
│   ├── batch_run_state.py           # Incremental run window
│   └── run_batch_prediction.sh
├── tests/
├── reports/benchmarks/     # Saved benchmark runs
└── docs/
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # add Supabase credentials
```

## Validate (benchmark)

```bash
# Default: 100 random historical events, Open divisions only
python validator/cli.py --count 100 --save

# Single event
python validator/cli.py --event <event-uuid>
```

Results → `reports/validation_<timestamp>.json` and `logs/validation/`.

See [docs/BENCHMARK_100_EVENTS.md](docs/BENCHMARK_100_EVENTS.md) for the latest 100-event benchmark.

## Predict (production batch)

Scores upcoming lineups and POSTs to the `update_result_predictions` Edge Function:

```bash
./batch/run_batch_prediction.sh              # incremental since last run
python batch/batch_predict_and_update.py --all-upcoming
```

## Tests

```bash
pytest tests/ -q
```

## Algorithm

Weighted composite of:

- Recent form (~50%)
- Head-to-head (~30%)
- Consistency (~8%)
- Momentum (~2%)
- Experience (~11%)

Division-aware filtering uses Open enhancement + Open age categories only (validation and batch default).
