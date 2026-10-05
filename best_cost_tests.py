#!/usr/bin/env python3
"""Paired tests for Section VII (best achievable cost per tracker and cost).

  python3 best_cost_tests.py > best_cost_report.txt

Unit of analysis: the array (seed s is the same array for every tracker). Costs compared: accuracy,
mean interference, outage at Gamma=0.03 (representative threshold) and selection gap, at M = 16, 64 and
p = 2e-4, 8e-4: 3 tracker pairs x 4 costs x 2 M x 2 p = 48 comparisons.

Two tuning protocols:
  in-sample   each tracker's step minimises the seed-mean cost over all arrays (operating frontier);
  LOSO        leave-one-array-out: the step for array s is chosen on the other arrays, then evaluated
              on s (out-of-sample).
Tests: paired t-test on per-array differences (Holm across the 48), with a Shapiro-Wilk check of
the paired differences; Wilcoxon signed-rank as a robustness check (its exact p cannot fall below
0.002 with 10 arrays). Ratios of means carry 95% array-bootstrap intervals.
"""
import numpy as np
from scipy import stats
import null_analyze as N

COSTS = [("sin_mean", "accuracy"), ("I_mean", "interference"), ("out_0.03", "outage G=0.03"), ("gap", "selection gap")]
PAIRS = [("ewcov", "oja"), ("past", "oja"), ("ewcov", "past")]
rows = N.load("results", 0, 2e-3)
rows = [r for r in rows if r.get("array", "gauss") == "gauss" and r.get("snr", 8.0) == 8.0]
rng = np.random.default_rng(0)


def per_array(R, key, loso):
    steps = sorted({r["step"] for r in R})
    tab = {s: {r["seed"]: r[key] for r in R if r["step"] == s} for s in steps}
    seeds = sorted(set.intersection(*[set(v) for v in tab.values()]))
    if not loso:
        best = min(steps, key=lambda s: np.mean([tab[s][a] for a in seeds]))
        return {a: tab[best][a] for a in seeds}
    out = {}
    for a in seeds:
        others = [b for b in seeds if b != a]
        best = min(steps, key=lambda s: np.mean([tab[s][b] for b in others]))
        out[a] = tab[best][a]
    return out


for proto in ("in-sample", "LOSO"):
    tests = []
    for M in (16, 64):
        for p in (2e-4, 8e-4):
            for key, name in COSTS:
                vals = {}
                for tr in ("oja", "past", "ewcov"):
                    R = [r for r in rows if r["method"] == tr and r.get("M", 16) == M and r["p"] == p and key in r]
                    if R:
                        vals[tr] = per_array(R, key, proto == "LOSO")
                for a_, b_ in PAIRS:
                    if a_ not in vals or b_ not in vals:
                        continue
                    sd = sorted(set(vals[a_]) & set(vals[b_]))
                    x = np.array([vals[a_][s] for s in sd]); y = np.array([vals[b_][s] for s in sd])
                    if len(sd) < 5 or y.mean() == 0:
                        continue
                    idx = rng.integers(0, len(sd), (2000, len(sd)))
                    lo, hi = np.percentile(x[idx].mean(1) / np.maximum(y[idx].mean(1), 1e-12), [2.5, 97.5])
                    d = x - y
                    pt = stats.ttest_rel(x, y).pvalue if np.any(d != 0) else 1.0
                    pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                    psw = stats.shapiro(d).pvalue if np.ptp(d) > 0 else 1.0
                    tests.append((f"M={M} p={p:g} {name}: {a_}/{b_}", x.mean() / y.mean(), lo, hi, pt, pw, psw, len(sd)))
    ps = np.array([t[4] for t in tests]); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, ps[i] * (len(ps) - rank)); adj[i] = min(run, 1.0)
    print(f"\n=== {proto} tuning: {len(tests)} comparisons (ratio of means [95% array bootstrap], paired t p, Holm, Wilcoxon p, Shapiro p of differences)")
    for (lab, r, lo, hi, pt, pw, psw, n), q in zip(tests, adj):
        print(f"  {lab:42s} {r:5.2f} [{lo:.2f}, {hi:.2f}]  t {pt:.2g}  Holm {q:.2g}{' *' if q < 0.05 else '  '}  W {pw:.2g}  SW {psw:.2g}  (n={n})")
    sm = [(t, q) for t, q in zip(tests, adj) if "selection" not in t[0]]
    se = [(t, q) for t, q in zip(tests, adj) if "selection" in t[0]]
    print(f"  summary: smooth/outage {sum(q < 0.05 for _, q in sm)}/{len(sm)} significant after Holm; "
          f"selection {sum(q < 0.05 for _, q in se)}/{len(se)}; Shapiro p<0.05 in {sum(t[6] < 0.05 for t in tests)}/{len(tests)} comparisons")
