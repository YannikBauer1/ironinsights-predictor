# Predictor v2

Win probabilities, placings and power rankings for IFBB Pro League divisions. Findings and test
results are in REPORT.md.

## Setup

```bash
cd Predictor/v2
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Data (read only)

```bash
python3 dump.py    # SELECT-only snapshot of every pro result into data/results.csv (~2 min)
```

`dbq.sh` runs one SELECT through the linked Supabase project (`supabase db query --linked`) inside a
READ ONLY transaction, so Postgres itself refuses any write. Nothing in v2 writes to the database.

## Run

```bash
.venv/bin/python run_final.py        # train on everything, write output/*.json (~7 min); --today YYYY-MM-DD to pin the date
.venv/bin/python walk.py             # walk 2018 to now with the final config -> data/walk.pkl (~6 min)
.venv/bin/python backtest_final.py   # rolling 2024-2026 backtest, calibration tables -> data/final_test_preds.pkl
.venv/bin/python case_studies.py     # v1 vs v2 on 2026, Olympia 2026 pre-show predictions
.venv/bin/python ladder.py           # every model step on the same test divisions
.venv/bin/python -m pytest tests -q
```

`backtest_final.py` and `ladder.py` need `data/walk.pkl`; `case_studies.py` and `ladder.py` need
`data/final_test_preds.pkl`. Run them in the order above.

Outputs:

* `output/upcoming.json`: every lineup not judged yet, with win %, P(2nd), P(3rd), top 3 %, top 5 %,
  expected placing and predicted place per athlete.
* `output/power_rankings.json`: top 40 per division, with the rank three months earlier.
* `output/ranking_history.json`: top 10 per division at every year end since 2018.

## Layout

```
bbpredict/
  data.py       snapshot -> chronological contests; rating pools
  evaluate.py   leak-free walk-forward runner and metrics
  models.py     baselines: uniform, recent placings, multiplayer Elo, Weng-Lin
  plfit.py      time-weighted Plackett-Luce, MAP refit per show date (forward or reverse, log-domain likelihood)
  features.py   stage-2 features recorded during the walk (status, layoffs, switches, age groups, head to head, ...)
  stage2.py     conditional logit (Plackett-Luce) layer
  blend.py      final stage 2: logit + LightGBM ranker, split Open-age / Masters (masters get extra inputs)
  final.py      the chosen configuration: recorder() (eight rating views) and make_blend()
  pipeline.py   replay history, predict a lineup, power ranking
  montecarlo.py placing probabilities by simulation
experiments/    the exploration scripts behind REPORT.md (round 1 and round 2, see its README)
tests/          leakage, likelihood gradient and stability, whole-lineup scoring, new features, judge count, probability sums
```
