"""Walk 2018 to today once with the final configuration, recording leak-free stage-2 features.

Writes data/walk.pkl = (predictions, feature snapshot per division, column names). backtest_final.py,
ladder.py and the experiments read it, so whatever final.py says is exactly what gets backtested.
usage: walk.py [<start-year> <out.pkl>]     e.g. 2012 data/walk_2012.pkl, for analyses that need a longer test span
"""
import pickle
import sys
import time

from bbpredict.data import cached_contests
from bbpredict.evaluate import walk_forward
from bbpredict.final import recorder

WALK = "data/walk.pkl"


def main(start="2018", out=WALK):
    t = time.time()
    rec = recorder()
    preds = walk_forward(rec, cached_contests(), f"{start}-01-01", "2026-12-31")
    with open(out, "wb") as f:
        pickle.dump((preds, rec.snap, rec.names), f)
    print(f"walked {len(preds)} divisions from {start} in {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
