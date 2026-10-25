"""Train v2 on every result in the snapshot, then write power rankings and upcoming-show predictions.

Outputs (local only, nothing touches the database):
  output/power_rankings.json   current ranking per division (pros who competed in the last 30 months and
                               are not marked inactive), with months away and the rank three months earlier
  output/ranking_history.json  top 10 per division at every year end since 2018
  output/upcoming.json         win, P(2nd), P(3rd), top 3, top 5 and expected place for every lineup not yet judged
  data/v2_state.pkl            the trained recorder + stage 2, for reuse

Usage: .venv/bin/python run_final.py [--today YYYY-MM-DD]   (default: today; results before it are history)
"""
import argparse
import json
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np

from bbpredict.data import DIVISIONS, cached_contests
from bbpredict.evaluate import walk_forward
from bbpredict.final import make_blend, recorder
from bbpredict.pipeline import predict_contest, rank, ranking_field

OUT = Path("output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=date.today().isoformat())
    today = np.datetime64(ap.parse_args().today)
    earlier = today - np.timedelta64(92, "D")            # rank movement over about three months
    year_ends = [np.datetime64(f"{y}-12-31") for y in range(2018, int(str(today)[:4]))]
    OUT.mkdir(exist_ok=True)

    t = time.time()
    cs = cached_contests()
    inactive_file = Path("data/inactive_athletes.json")   # from `dump.py inactive`; today's flag, ranking only
    inactive = {r["person_id"] for r in json.loads(inactive_file.read_text())} if inactive_file.exists() else set()
    done = [c for c in cs if c.n_placed >= 1]
    rec = recorder()
    preds, fields, prev = [], {}, None
    for d in sorted(year_ends + [earlier]) + [today]:
        segment = [c for c in done if (prev is None or c.date >= prev) and c.date < d]
        preds += walk_forward(rec, segment, "2018-01-01", str(d))
        prev = d
        if d != today:
            for cat in DIVISIONS:
                # today's retirement flag only for recent snapshots: past year ends list who was active then
                if (f := ranking_field(rec, cs, cat, str(d), inactive=inactive if d == earlier else None)) is not None:
                    fields[(d, cat)] = f
    print("replayed", len(preds), "divisions in", round(time.time() - t), "s", flush=True)
    blend = make_blend(rec.names).fit([(rec.snap[p.contest.division_id], p.contest) for p in preds])

    rankings = {}
    for cat in DIVISIONS:
        f = ranking_field(rec, cs, cat, str(today), inactive=inactive)
        if f is None:
            continue
        now = rank(blend, f)
        before = {r["person_id"]: r["rank"] for r in rank(blend, fields[(earlier, cat)])} if (earlier, cat) in fields else {}
        for r in now:
            r["rank_3m_ago"] = before.get(r["person_id"])
        rankings[cat] = now[:40]
    (OUT / "power_rankings.json").write_text(json.dumps(dict(as_of=str(today), rankings=rankings), indent=1))
    history = {f"{d}|{cat}": rank(blend, f)[:10] for (d, cat), f in fields.items() if d != earlier}
    (OUT / "ranking_history.json").write_text(json.dumps(history, indent=1))

    upcoming = [dict(date=str(c.date), show=c.comp, division=c.cat, age=c.age, enh=c.enh, division_id=c.division_id,
                     entries=predict_contest(rec, blend, c))
                for c in cs if c.date >= today and c.n_placed == 0 and len(c.persons) >= 2]
    (OUT / "upcoming.json").write_text(json.dumps(upcoming, indent=1))
    with open("data/v2_state.pkl", "wb") as fh:
        pickle.dump((rec, blend), fh)
    print("done in", round(time.time() - t), "s")


if __name__ == "__main__":
    main()
