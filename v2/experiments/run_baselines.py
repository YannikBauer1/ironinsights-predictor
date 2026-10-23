import time
from bbpredict.data import load_contests
from bbpredict.evaluate import walk_forward, summarize, fit_scale, fmt
from bbpredict.models import Uniform, RecentPlace, Elo, WengLin

cs = load_contests()
TUNE = ("2018-01-01", "2024-01-01")
for name, mk in [("uniform", Uniform), ("recent", RecentPlace), ("elo", lambda: Elo(k=32.0, field_exp=0.5, tier_k={})), ("wenglin", WengLin)]:
    t = time.time()
    preds = walk_forward(mk(), cs, *TUNE)
    s = fit_scale(preds) if name != "uniform" else 1.0
    print(f"{name:8s} scale={s:.3f} {fmt(summarize(preds, s))}  ({time.time()-t:.0f}s)")
