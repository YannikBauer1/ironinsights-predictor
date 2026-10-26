# Experiments

The exploration scripts behind REPORT.md. Run them from `Predictor/v2` with the package on the path.

## Round 1 (the first v2)

```bash
PYTHONPATH=. .venv/bin/python experiments/run_baselines.py        # uniform, heuristic, untuned Elo, Weng-Lin
PYTHONPATH=. .venv/bin/python experiments/tune_pl.py              # coordinate search for the PL base (slow, ~1 h)
.venv/bin/python walk.py                                          # walk once with the final config (~10 min)
PYTHONPATH=. .venv/bin/python experiments/exp_multibase_eval.py   # which views and features stage 2 keeps
PYTHONPATH=. .venv/bin/python experiments/exp_lgbm3.py            # boosted ranker, iterations chosen inside 2018-2023
PYTHONPATH=. .venv/bin/python experiments/exp_validate_choices.py # split Masters layer and blend weight, on 2022-2023
PYTHONPATH=. .venv/bin/python experiments/exp_masters.py          # the same split, on 2024-2026
.venv/bin/python backtest_final.py                                # needed by exp_benter.py
PYTHONPATH=. .venv/bin/python experiments/exp_benter.py           # damped later picks (rejected)
```

The tuning log of the PL base stopped in its second round once gains fell below 0.002 in log-loss;
`bbpredict/final.py` uses the configuration reached by then, not the very last candidate tried.

## Round 2 (Masters inputs; everything else rejected)

Choices on 2018-2023 only, then one confirmation on 2024-2026.

```bash
PYTHONPATH=. .venv/bin/python experiments/retune_pl.py            # re-tune of the main rating, whole-lineup scoring
PYTHONPATH=. .venv/bin/python experiments/per_division_grid.py    # decay and prior strength per division
.venv/bin/python walk.py                                          # candidate A = the final config
PYTHONPATH=. .venv/bin/python experiments/walk_candidate_b.py c   # candidate C: re-tuned weights only
PYTHONPATH=. .venv/bin/python experiments/walk_candidate_b.py b   # candidate B: C plus per-division settings
PYTHONPATH=. .venv/bin/python experiments/screen_round2.py data/walk.pkl logit   # each input group, logit layer
PYTHONPATH=. .venv/bin/python experiments/screen_layer.py open    # the groups in the Open-age layer (none helped)
PYTHONPATH=. .venv/bin/python experiments/screen_layer.py masters # the groups in the masters layer (all four kept)
PYTHONPATH=. .venv/bin/python experiments/round2_folds.py data/walk.pkl data/r2_tune_A.pkl 2022,2023
PYTHONPATH=. .venv/bin/python experiments/round2_folds.py data/baseline_walk.pkl data/r2_tune_base.pkl 2022,2023 baseline
PYTHONPATH=. .venv/bin/python experiments/calibration_round2.py   # size / mix calibration (rejected)
PYTHONPATH=. .venv/bin/python experiments/round2_shrink.py data/r2_tune_A.pkl    # one fixed shrink (rejected on 2024-2026)
PYTHONPATH=. .venv/bin/python experiments/before_after.py         # before / after on 2024-2026, our test and PRISM's
```

`before_after.py` compares `data/baseline_final_test_preds.pkl` (the committed round-1 model's
`backtest_final.py` output, kept aside before round 2) with the current `data/final_test_preds.pkl`.

## Round 3 (error analysis; time away for both layers, travel for masters)

Patterns searched on 2020-2023 only, fixes screened on 2021-2023, confirmed on 2024-2026. Two leaks found on the way
(finishing-order tie-breaks, unknown country) are fixed; REPORT.md describes them.

```bash
PYTHONPATH=. .venv/bin/python experiments/round2_folds.py data/walk.pkl data/r3_analysis_cand.pkl 2020,2021,2022,2023
PYTHONPATH=. .venv/bin/python experiments/error_analysis.py data/r3_analysis_cand.pkl    # expected vs actual wins by group
PYTHONPATH=. .venv/bin/python experiments/round3_eval.py open "time away"                # one input group in one layer
PYTHONPATH=. .venv/bin/python experiments/round3_eval.py masters "travel + time away"
PYTHONPATH=. .venv/bin/python experiments/round3_eval.py params logit_l2_0.3            # one stage-2 parameter setting
PYTHONPATH=. .venv/bin/python experiments/posthoc_calibration.py data/r3_analysis_cand.pkl   # rank / division corrections (rejected)
PYTHONPATH=. .venv/bin/python experiments/before_after.py data/round2_final_test_preds.pkl data/final_test_preds.pkl   # round 2 vs now
```

`round3_eval.py` lists every group and parameter setting that was tried. `round2_folds.py` now
uses the current `final.make_blend`, so its output follows whatever `final.py` holds.

## The reigning champion (final.CHAMPION_BONUS)

```bash
PYTHONPATH=. .venv/bin/python experiments/champion_between_olympias.py      # beaten between Olympias? kept the title?
PYTHONPATH=. .venv/bin/python experiments/champion_rules.py data/r3_analysis_cand.pkl data/r3_final_test_preds.pkl
PYTHONPATH=. .venv/bin/python experiments/champion_weight.py data/walk.pkl base learned base+0.25 base+0.5 base+1.0 base+0.5all
PYTHONPATH=. .venv/bin/python experiments/champion_weight.py data/walk.pkl --years 2024,2025,2026 base base+0.5
PYTHONPATH=. .venv/bin/python experiments/walk_olympia_weight.py 3 data/walk_oly3.pkl   # heavier Olympia (rejected)
PYTHONPATH=. .venv/bin/python experiments/champion_weight.py data/walk_oly3.pkl base
```

The first writes `data/champion_seasons.pkl`, which `champion_rules.py` needs. `champion_rules.py` scored a
ranking rule (champion first until beaten); it was replaced by the bonus inside the model.
`champion_weight.py` adds its own champion inputs to the snapshot, so it also runs on a walk made
before `champ_unbeaten` existed.

## PRISM's test, per division

```bash
PYTHONPATH=. .venv/bin/python experiments/prism_by_division.py     # JSON lines: v2 and PRISM's paper per division
```

The test rules are `bbpredict/prism.py`; `before_after.py` uses the same function.
