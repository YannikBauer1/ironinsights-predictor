"""Do later picks need a damped strength (Benter)? Calibrate P(top 3) on 2024-25, check on 2026."""
import pickle
import numpy as np

out = pickle.load(open("data/final_test_preds.pkl", "rb"))

def sample_places(u, alpha, sims, rng):
    n = len(u)
    taken = np.zeros((sims, n), bool)
    place = np.zeros((sims, n), int)
    for k in range(min(n, 3)):
        a = 1.0 if k == 0 else alpha
        perf = a * u[None, :] + rng.gumbel(size=(sims, n))
        perf[taken] = -np.inf
        pick = perf.argmax(1)
        taken[np.arange(sims), pick] = True
        place[np.arange(sims), pick] = k + 1
    return place

def top3_ll(preds, alpha):
    rng = np.random.default_rng(1)
    ll, n = 0.0, 0
    for p in preds:
        c = p.contest; m = c.placed_mask
        if m.sum() < 4: continue
        pl = sample_places(p.theta[m], alpha, 3000, rng)
        q = np.clip(((pl >= 1) & (pl <= 3)).mean(0), 1e-4, 1 - 1e-4)
        y = c.places[m] <= 3
        ll += -(y * np.log(q) + (~y) * np.log(1 - q)).sum(); n += m.sum()
    return ll / n

fit = [p for p in out if p.contest.year < 2026]
chk = [p for p in out if p.contest.year == 2026]
for a in [1.0, 0.9, 0.8, 0.7]:
    print(f"alpha={a}: fit {top3_ll(fit, a):.4f}  check 2026 {top3_ll(chk, a):.4f}")
