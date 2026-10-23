# Experiments

The exploration scripts behind REPORT.md. Run them from `Predictor/v2` with the package on the path:

```bash
PYTHONPATH=. .venv/bin/python experiments/run_baselines.py        # uniform, heuristic, untuned Elo, Weng-Lin
PYTHONPATH=. .venv/bin/python experiments/tune_pl.py              # coordinate search for the PL base (slow, ~1 h)
.venv/bin/python walk.py                                          # walk once with the final config (~4 min)
PYTHONPATH=. .venv/bin/python experiments/exp_multibase_eval.py   # which views and features stage 2 keeps
PYTHONPATH=. .venv/bin/python experiments/exp_lgbm3.py            # boosted ranker, iterations chosen inside 2018-2023
PYTHONPATH=. .venv/bin/python experiments/exp_validate_choices.py # split Masters layer and blend weight, on 2022-2023
PYTHONPATH=. .venv/bin/python experiments/exp_masters.py          # the same split, on 2024-2026
.venv/bin/python backtest_final.py                                # needed by exp_benter.py
PYTHONPATH=. .venv/bin/python experiments/exp_benter.py           # damped later picks (rejected)
```

The tuning log of the PL base stopped in its second round once gains fell below 0.002 in log-loss;
`bbpredict/final.py` uses the configuration reached by then, not the very last candidate tried.
