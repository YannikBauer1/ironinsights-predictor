"""Walk 2018 to today once with the final configuration, recording leak-free stage-2 features.

Writes data/walk.pkl = (predictions, feature snapshot per division, column names). backtest_final.py,
ladder.py and the experiments read it, so whatever final.py says is exactly what gets backtested.
"""
import pickle
import time

from bbpredict.data import cached_contests
from bbpredict.evaluate import walk_forward
from bbpredict.final import recorder

WALK = "data/walk.pkl"


def main():
    t = time.time()
    rec = recorder()
    preds = walk_forward(rec, cached_contests(), "2018-01-01", "2026-12-31")
    with open(WALK, "wb") as f:
        pickle.dump((preds, rec.snap, rec.names), f)
    print(f"walked {len(preds)} divisions in {time.time() - t:.0f}s -> {WALK}")


if __name__ == "__main__":
    main()
