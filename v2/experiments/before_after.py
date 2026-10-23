"""Round 2 before/after on the held-out years 2024-2026: our main test (whole lineup, every division)
and PRISM's published protocol. Reads data/baseline_final_test_preds.pkl (before) and
data/final_test_preds.pkl (after)."""
import math
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, spearmanr

from bbpredict.data import DIVISIONS, cached_contests
from bbpredict.evaluate import calibration_table, summarize

before = pickle.load(open("data/baseline_final_test_preds.pkl", "rb"))
after = pickle.load(open("data/final_test_preds.pkl", "rb"))
assert [p.contest.division_id for p in before] == [p.contest.division_id for p in after]

SLICES = {
    "All 2024-2026": lambda c: True,
    "  2024": lambda c: c.year == 2024, "  2025": lambda c: c.year == 2025, "  2026": lambda c: c.year == 2026,
    "Open age (open circuit)": lambda c: c.is_open_age and c.enh == "Open",
    "Masters": lambda c: not c.is_open_age,
    "Natural": lambda c: c.enh == "Natural",
    "Olympia": lambda c: c.tier == "O",
    "Masters Olympia": lambda c: c.tier == "MO",
}
print(f"{'MAIN TEST':26s} {'div':>5s} | {'log-loss':>17s} | {'favourite wins':>15s} | {'top-3 overlap':>15s} | {'Spearman':>13s}")
for name, f in SLICES.items():
    b, a = summarize([p for p in before if f(p.contest)]), summarize([p for p in after if f(p.contest)])
    print(f"{name:26s} {a['divisions']:5d} | {b['ll_win']:.3f} -> {a['ll_win']:.3f} | {b['top1']:5.1%} -> {a['top1']:5.1%} | "
          f"{b['top3']:5.1%} -> {a['top3']:5.1%} | {b['spearman']:.3f} -> {a['spearman']:.3f}")
print("\nMasters by division (favourite wins, before -> after)")
for cat in DIVISIONS:
    b = [p for p in before if p.contest.cat == cat and not p.contest.is_open_age]
    a = [p for p in after if p.contest.cat == cat and not p.contest.is_open_age]
    if len(a) >= 20:
        sb, sa = summarize(b), summarize(a)
        print(f"  {cat:16s} n={sa['divisions']:4d}  {sb['top1']:5.1%} -> {sa['top1']:5.1%}   ll {sb['ll_win']:.3f} -> {sa['ll_win']:.3f}")
for label, preds in (("before", before), ("after", after)):
    tab, ece = calibration_table(preds, 1.0)
    print(f"\ncalibration {label}: ECE={ece:.4f}  " + "  ".join(f"{lo:.1f}+: {x:.3f}/{y:.3f}" for lo, _, n, x, y in tab if lo >= 0.5))

# PRISM protocol on the held-out years
DIVS = {"mensbb", "classic", "mensphysique", "bikini", "fitness", "wellness"}
MEN = {"mensbb", "classic", "mensphysique"}
START, END = np.datetime64("2023-01-01"), np.datetime64("2026-07-19")
scope = [c for c in cached_contests() if c.cat in DIVS and c.is_open_age and c.enh == "Open" and START <= c.date < END]
shows = defaultdict(list)
for c in scope:
    for p, pl in zip(c.persons, c.places):
        if not np.isnan(pl):
            shows[(c.cat, p)].append((c.date, pl))


def rated(c, p):
    past = [(d, pl) for d, pl in shows[(c.cat, p)] if d < c.date]
    return len(past) >= 3 or (c.cat in MEN and any(pl == 1 for _, pl in past))


def prism(preds):
    theta = {p.contest.division_id: p.theta for p in preds}
    ok = tot = 0
    win_show, win_rated, rho, tau, ll = [], [], [], [], []
    for c in scope:
        if c.date < np.datetime64("2024-01-01") or c.division_id not in theta:
            continue
        idx = [i for i, p in enumerate(c.persons) if not np.isnan(c.places[i]) and rated(c, p)]
        if len(idx) < 2:
            continue
        th, pl = theta[c.division_id][idx], c.places[idx]
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                if pl[x] != pl[y]:
                    w, l = (x, y) if pl[x] < pl[y] else (y, x)
                    tot += 1
                    ok += th[w] > th[l]
                    ll.append(math.log1p(math.exp(th[l] - th[w])))
        fav = int(np.argmax(th))
        win_show.append(pl[fav] == 1)
        win_rated.append(pl[fav] == pl.min())
        if len(idx) >= 3 and np.ptp(th) > 0 and np.ptp(pl) > 0:
            rho.append(spearmanr(-th, pl).statistic)
            tau.append(kendalltau(-th, pl).statistic)
    return ok / tot, np.mean(win_show), np.mean(win_rated), np.nanmean(rho), np.nanmean(tau), np.mean(ll)


pb, pa = prism(before), prism(after)
print("\nPRISM protocol, 2024 to 18 Jul 2026 (PRISM paper: pairs 77%, winner 51%, Spearman 0.57, Kendall 0.48, log loss 0.47)")
for name, i, f in [("pairs ordered correctly", 0, "{:.1%}"), ("favourite wins (show)", 1, "{:.1%}"), ("favourite wins (rated)", 2, "{:.1%}"),
                   ("Spearman", 3, "{:.3f}"), ("Kendall", 4, "{:.3f}"), ("pairwise log loss", 5, "{:.3f}")]:
    print(f"  {name:26s} {f.format(pb[i])} -> {f.format(pa[i])}")
