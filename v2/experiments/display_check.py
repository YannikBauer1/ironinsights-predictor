"""Display check for the app's Odds tab and Power Ranking (hub#334): what would fans have seen at every show
of 2024 to 2026? Stage 2 is refitted every January on everything before it (backtest_final.rolling_predictions),
and the Power Ranking is rebuilt the day before every show date from the recorder's state at that moment,
with win % over the top 50 (the app's cut-off). No inactive flag: today's flag says nothing about the past.
Favourites and the odds' top 3 break ties by athlete id (evaluate.lineup_rank), never by list position.

  compute   replay the recorder from 2018, keep 2024 to 2026, write data/display_check.pkl (about 11 minutes)
  report    print the answers to the five data questions of hub#334

usage: display_check.py compute | report
"""
import pickle
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

from bbpredict.data import cached_contests
from bbpredict.evaluate import lineup_rank, softmax, walk_forward
from bbpredict.final import make_blend, recorder
from bbpredict.montecarlo import place_probabilities
from bbpredict.pipeline import rank, ranking_field

warnings.filterwarnings("ignore")
OUT = "data/display_check.pkl"
YEARS = (2024, 2025, 2026)
TOP = 50                       # ranking places that get a "#N" chip, and the field the ranking's win % covers
UNDER_1 = 0.005                # the app shows "<1%" below half a percent
RECENT_DAYS = 730              # strict dark horse: the win over a top-3 athlete is from the last two years
DARK_HORSE_TOP3 = 0.15         # strict dark horse: the model gives 15% or more for the top 3


def shown(p):
    """The app's display: whole percent, "<1%" below half a percent."""
    return "<1%" if p < UNDER_1 else f"{round(p * 100)}%"


def open_circuit(c):
    """An Open-age class of the open circuit: the only classes with a Power Ranking."""
    return c.is_open_age and c.enh == "Open"


def compute():
    t = time.time()
    with open("data/walk.pkl", "rb") as fh:
        preds, snap, names = pickle.load(fh)
    blends = {}
    for year in YEARS:
        lo = np.datetime64(f"{year}-01-01")
        blends[year] = make_blend(names).fit([(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo])
        print(year, "stage 2 fitted", round(time.time() - t), "s", flush=True)

    shows = []
    for p in preds:
        c = p.contest
        if c.year in YEARS:
            theta = blends[c.year].utility(snap[c.division_id], c)
            shows.append(dict(contest=c, theta=theta, order=lineup_rank(theta, c.persons), win=softmax(theta),
                              top3=place_probabilities(theta, sims=4000)[:, :3].sum(1)))
    print(len(shows), "show classes predicted", round(time.time() - t), "s", flush=True)

    # the ranking the day before each show date, for divisions with an Open-age open-circuit class that day
    cs = cached_contests()
    needed = defaultdict(set)
    for s in shows:
        if open_circuit(s["contest"]):
            needed[s["contest"].date].add(s["contest"].cat)
    rec = recorder()
    start = np.datetime64("2024-01-01")
    walk_forward(rec, [c for c in cs if c.date < start], "2018-01-01", "2024-01-01")
    rankings = {}
    rest = [c for c in cs if c.date >= start]
    for d in sorted({c.date for c in rest}):
        for cat in sorted(needed.get(d, ())):
            f = ranking_field(rec, cs, cat, str(d))
            if f is not None:
                rankings[(d, cat)] = {r["person_id"]: r for r in rank(blends[int(str(d)[:4])], f, field=TOP)}
        walk_forward(rec, [c for c in rest if c.date == d], "2024-01-01", "2027-01-01")
    print(len(rankings), "rankings built", round(time.time() - t), "s", flush=True)
    # the replay must see exactly what data/walk.pkl saw, or the rankings and the odds disagree
    ids = [s["contest"].division_id for s in shows]
    same = sum(np.allclose(rec.snap[i], snap[i], equal_nan=True) for i in ids)
    print(f"features identical to data/walk.pkl for {same} of {len(ids)} show classes", flush=True)
    if same != len(ids):
        sys.exit("replay drifted from data/walk.pkl; rerun walk.py first")

    with open(OUT, "wb") as fh:
        pickle.dump(dict(shows=shows, rankings=rankings), fh)
    print("wrote", OUT, round(time.time() - t), "s")


# ---------------------------------------------------------------- report

def label(c):
    cls = "Open" if c.is_open_age else c.age
    return f"{c.date} {c.comp} {c.cat} {cls}{' Natural' if c.enh == 'Natural' else ''}"


def skill(rows):
    """1 minus the winner log-loss over the uniform guess's."""
    return 1 - np.mean([r["ll"] for r in rows]) / np.mean([r["llu"] for r in rows])


def report():
    with open(OUT, "rb") as fh:
        d = pickle.load(fh)
    shows, rankings = d["shows"], d["rankings"]
    cs = cached_contests()

    # earlier pro results and who beat whom, from every contest in the snapshot (dates sorted)
    first_placed = {}
    beat = defaultdict(dict)                        # cat -> (a, b) -> dates a placed ahead of b
    for c in cs:
        placed = [(p, pl) for p, pl in zip(c.persons, c.places) if not np.isnan(pl)]
        for p, _ in placed:
            first_placed.setdefault(p, c.date)
        for a, pa in placed:
            for b, pb in placed:
                if pa < pb:
                    beat[c.cat].setdefault((a, b), []).append(c.date)

    open_shows = [s for s in shows if open_circuit(s["contest"])]
    chip_rows(open_shows, rankings)
    long_tail(shows)
    little_history(shows, first_placed)
    for strict in (False, True):
        dark_horse(shows, beat, strict)
    lineup_strength(open_shows, rankings)


def chip_rows(open_shows, rankings):
    print("\n1. Ranking chip against the show's win % (Open-age open-circuit classes, 2024 to 2026)")
    bands = [(1, 10), (11, 30), (31, 50)]
    tally = {b: Counter() for b in bands}
    by_cat, examples = defaultdict(Counter), []
    for s in open_shows:
        c = s["contest"]
        r = rankings.get((c.date, c.cat), {})
        for i, p in enumerate(c.persons):
            if p not in r or r[p]["rank"] > TOP:
                continue
            k, low = r[p]["rank"], s["win"][i] < UNDER_1
            b = next(b for b in bands if b[0] <= k <= b[1])
            tally[b]["n"] += 1
            tally[b]["lt1"] += low
            if k >= 31:
                by_cat[c.cat]["n"] += 1
                by_cat[c.cat]["lt1"] += low
                if low:
                    examples.append((k, s["win"][i], c.names[i], label(c), len(c.persons)))
    for b in bands:
        n = tally[b]["n"]
        print(f"  #{b[0]:2d} to #{b[1]:2d}: {n:5d} chips at shows, '<1%' on {tally[b]['lt1']:4d} ({tally[b]['lt1'] / max(n, 1):.0%})")
    print("  #31 to #50 by division:")
    for cat, t in sorted(by_cat.items(), key=lambda x: -x[1]["lt1"] / x[1]["n"]):
        print(f"    {cat:16s} {t['n']:4d} chips, '<1%' on {t['lt1']:4d} ({t['lt1'] / t['n']:.0%})")
    print("  the best-ranked '#31 to #50 next to <1%' cases:")
    for k, w, nm, lab, n in sorted(examples)[:10]:
        print(f"    #{k} {nm}: {shown(w)} at {lab} ({n} in the lineup)")


def long_tail(shows):
    print("\n2. Classes where most of the lineup shows '<1%'")
    for name, keep in [("all classes", lambda c: True), ("Open-age open circuit", open_circuit),
                       ("Masters", lambda c: not c.is_open_age), ("Natural", lambda c: c.enh == "Natural")]:
        sub = [s for s in shows if keep(s["contest"])]
        low = [s["win"] < UNDER_1 for s in sub]
        rows = sum(len(x) for x in low)
        print(f"  {name:22s} {len(sub):5d} classes, rows '<1%': {sum(x.sum() for x in low) / rows:.0%}, classes with more than "
              f"half '<1%': {sum(x.mean() > 0.5 for x in low)} ({np.mean([x.mean() > 0.5 for x in low]):.0%})")
    big = [s for s in shows if len(s["contest"].persons) >= 20]
    low = [s["win"] < UNDER_1 for s in big]
    tail = [s["win"][x].sum() for s, x in zip(big, low)]
    print(f"  lineups of 20 or more: {len(big)}, more than half '<1%': {sum(x.mean() > 0.5 for x in low)}; "
          f"rows '<1%' per class median {np.median([x.sum() for x in low]):.0f}; together median {np.median(tail):.1%}, "
          f"at most {max(tail):.1%}")
    open_low = np.concatenate([s["top3"][s["win"] < UNDER_1] for s in shows if open_circuit(s["contest"])])
    print(f"  Open-age rows '<1%': their top-3 chance median {np.median(open_low):.1%}")


def little_history(shows, first_placed):
    print("\n3. Accuracy by share of the lineup with an earlier pro result (classes with a single winner)")
    rows = []
    for s in shows:
        c = s["contest"]
        w = c.winner
        if w is None:
            continue
        fav = int(np.flatnonzero(s["order"] == 1)[0])
        rows.append(dict(c=c, hist=np.mean([first_placed.get(p, np.datetime64("2100-01-01")) < c.date for p in c.persons]),
                         ll=-np.log(max(s["win"][w], 1e-12)), llu=np.log(len(c.persons)),
                         pfav=float(s["win"][fav]), favwon=float(fav == w)))
    cut = [0, 0.25, 0.5, 0.75, 0.9, 1.0001]

    def table(sub, title):
        print(f"  {title}")
        print("    with history     classes  log-loss  skill  favourite said  favourite won")
        for lo, hi in zip(cut[:-1], cut[1:]):
            g = [r for r in sub if lo <= r["hist"] < hi]
            if g:
                print(f"    {lo:4.0%} to {min(hi, 1):4.0%}   {len(g):7d}  {np.mean([r['ll'] for r in g]):8.3f}  {skill(g):5.2f}"
                      f"  {np.mean([r['pfav'] for r in g]):14.0%}  {np.mean([r['favwon'] for r in g]):13.0%}")

    table(rows, "every class")
    table([r for r in rows if open_circuit(r["c"])], "Open-age open circuit")
    table([r for r in rows if not r["c"].is_open_age], "Masters")
    table([r for r in rows if r["c"].enh == "Natural"], "Natural")
    wc = [r for r in rows if r["c"].cat == "wheelchair"]
    print(f"  Wheelchair: {len(wc)} classes, skill {skill(wc):.2f}, favourite said {np.mean([r['pfav'] for r in wc]):.0%}, "
          f"won {np.mean([r['favwon'] for r in wc]):.0%}, history share median {np.median([r['hist'] for r in wc]):.0%}")
    below = [r for r in rows if r["hist"] < 0.75]
    print(f"  classes under 75% with history: {len(below)} of {len(rows)} ({len(below) / len(rows):.0%})")


def dark_horse(shows, beat, strict):
    """Outside the odds' top 3 and has beaten one of the top 3 in the same division before (any class).
    strict: that win is from the last two years and the model gives 15% or more for the top 3."""
    rule = "beat a top-3 athlete in the last 2 years and has 15%+ for the top 3" if strict else "has beaten a top-3 athlete before"
    print(f"\n4. Dark horse: outside the odds' top 3, {rule} (classes with 4+ placed)")
    groups, per_class, examples, classes = defaultdict(list), Counter(), [], 0
    for s in shows:
        c = s["contest"]
        if c.n_placed < 4:
            continue
        classes += 1
        order = np.argsort(s["order"])
        top = order[:3]
        lo = c.date - np.timedelta64(RECENT_DAYS, "D") if strict else np.datetime64("1900-01-01")
        n_dh = 0
        for i in order[3:]:
            beaten = [j for j in top if any(lo <= day < c.date for day in beat[c.cat].get((c.persons[i], c.persons[j]), ()))]
            likely = s["top3"][i] >= DARK_HORSE_TOP3
            if strict and not likely:
                beaten = []
            finished = (not np.isnan(c.places[i])) and c.places[i] <= 3
            row = dict(p3=float(s["top3"][i]), top3=float(finished), win=float(s["win"][i]), won=float(c.places[i] == 1))
            if beaten:
                n_dh += 1
                groups["dark horses"].append(row)
                examples.append((c, i, [c.names[j] for j in beaten], finished, s["win"][i], s["top3"][i]))
            elif strict and likely:
                groups["same 15%+, no recent win over the top 3"].append(row)
            else:
                groups["everyone else outside the top 3"].append(row)
        per_class[min(n_dh, 4)] += 1
    for name, g in groups.items():
        print(f"  {name:40s} n={len(g):5d}  top-3 said {np.mean([r['p3'] for r in g]):.1%}  finished top 3 "
              f"{np.mean([r['top3'] for r in g]):.1%}  won said {np.mean([r['win'] for r in g]):.1%}  won {np.mean([r['won'] for r in g]):.1%}")
    print(f"  {classes} classes; by number of dark horses (4 = four or more):", dict(sorted(per_class.items())))
    big = [e for e in examples if open_circuit(e[0]) and e[0].tier in ("O", "2")]
    shown_examples = big if strict else big[::max(1, len(big) // 30)]
    print(f"  Olympia and tier-2 Open-age examples ({len(big)}{'' if strict else ', every ' + str(max(1, len(big) // 30)) + 'th'}):")
    for c, i, beaten, fin, w, p3 in shown_examples:
        result = "top 3" if fin else ("place " + str(int(c.places[i])) if not np.isnan(c.places[i]) else "no place")
        print(f"    {label(c)}: {c.names[i]} (win {shown(w)}, top 3 {shown(p3)}, beat {', '.join(beaten)}) -> {result}")


def lineup_strength(open_shows, rankings):
    print("\n5. Lineup strength = sum of the Power Ranking win % (top 50) standing in the lineup, Open-age open circuit")
    strength = []
    for s in open_shows:
        c = s["contest"]
        r = rankings.get((c.date, c.cat))
        if r:
            strength.append((sum(r[p]["olympia_win"] for p in c.persons if p in r), c))
    vals = np.array([v for v, _ in strength])
    print(f"  {len(vals)} classes; percentiles 10/25/50/75/90: " + " / ".join(f"{np.percentile(vals, q):.0%}" for q in (10, 25, 50, 75, 90)))
    q90, q65, q25 = np.percentile(vals, 90), np.percentile(vals, 65), np.percentile(vals, 25)
    band = lambda v: "Stacked" if v >= q90 else "Strong" if v >= q65 else "Normal" if v >= q25 else "Open"
    by_tier = defaultdict(Counter)
    for v, c in strength:
        by_tier[{"O": "Olympia", "2": "tier 2"}.get(c.tier, "other")][band(v)] += 1
    for tier, cnt in by_tier.items():
        print(f"  {tier:8s} " + ", ".join(f"{k} {cnt[k]}" for k in ("Stacked", "Strong", "Normal", "Open")))
    stacked = sorted(((v, c) for v, c in strength if v >= q90 and c.tier != "O"), key=lambda x: -x[0])
    print(f"  Stacked outside the Olympia ({len(stacked)}), by division: {dict(Counter(c.cat for _, c in stacked).most_common())}")
    for v, c in stacked[:15]:
        print(f"    {v:.0%}  {label(c)}")
    zero = sum(v < 0.01 for v in vals)
    print(f"  classes where the lineup holds less than 1% of the ranking's win chance: {zero} ({zero / len(vals):.0%})")


if __name__ == "__main__":
    {"compute": compute, "report": report}[sys.argv[1]]()
