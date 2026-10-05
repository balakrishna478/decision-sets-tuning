#!/usr/bin/env python3
"""Two robustness checks requested in review (reads results/ like null_analyze.py).

  python3 robustness.py > robustness_report.txt

(1) Exponent sensitivity to how eta* is located: raw grid minimum, 3-point parabola in log-step
    (the paper's method) and a 5-point quadratic in log-step.
(2) Change-rate misspecification: a user who believes the rate is p_hat = p * 2^d tunes at
    eta*(p_hat) but faces the true rate p. We report the cost ratio J(eta*(p_hat); p) / J(eta*(p); p),
    evaluating the seed-mean cost curve at the true rate by linear interpolation in log-step.
"""
import numpy as np
import null_analyze as N

COSTS = [("sin_mean", "accuracy"), ("I_mean", "interference"), ("out_0.01", "outage .01"),
         ("out_0.03", "outage .03"), ("out_0.1", "outage .1"), ("gap", "argmin")]
rows = N.load("results", 0, 2e-3)
rows = [r for r in rows if r.get("array", "gauss") == "gauss" and r.get("snr", 8.0) == 8.0 and r.get("M", 16) == 16]


def curves(R, key):
    rates = sorted({r["p"] for r in R}); out = {}
    for p in rates:
        steps = np.array(sorted({r["step"] for r in R if r["p"] == p and key in r}))
        if len(steps) < 3:
            continue
        mean = np.array([np.mean([r[key] for r in R if r["p"] == p and r["step"] == s]) for s in steps])
        out[p] = (steps, mean)
    return out


def locate(steps, y, method):
    i = int(np.argmin(y))
    if method == "grid":
        return steps[i], (i == 0 or i == len(y) - 1)
    w = 1 if method == "parabola3" else 2
    if i - w < 0 or i + w >= len(y):
        return steps[i], True
    x = np.log(steps[i - w:i + w + 1]); c = np.polyfit(x, y[i - w:i + w + 1], 2)
    if c[0] <= 0:
        return steps[i], False
    return float(np.exp(np.clip(-c[1] / (2 * c[0]), x[0], x[-1]))), False


print("(1) Exponent a by optimum-location method (interior optima only)")
print(f"  {'tracker':8s} {'cost':13s} {'grid min':>9s} {'parabola-3':>11s} {'quadratic-5':>12s}")
for tr in ("oja", "past", "ewcov"):
    R = [r for r in rows if r["method"] == tr]
    for key, name in COSTS:
        C = curves(R, key); res = []
        for meth in ("grid", "parabola3", "quad5"):
            pts = [(p, *locate(*C[p], meth)) for p in sorted(C)]
            pts = [(p, e) for p, e, edge in pts if not edge]
            res.append(np.polyfit(np.log([q[0] for q in pts]), np.log([q[1] for q in pts]), 1)[0] if len(pts) >= 3 else np.nan)
        print(f"  {tr:8s} {name:13s} {res[0]:9.3f} {res[1]:11.3f} {res[2]:12.3f}   max diff {np.nanmax(res) - np.nanmin(res):.3f}")

print("\n(2) Cost ratio when tuned for p_hat = p * 2^d (Oja, M=16; mean over true rates where both exist)")
print(f"  {'cost':13s} " + " ".join(f"{'p_hat/p=' + str(2.0 ** d):>14s}" for d in (-2, -1, 1, 2)))
R = [r for r in rows if r["method"] == "oja"]
for key, name in COSTS:
    C = curves(R, key); rates = sorted(C)
    opt = {p: locate(*C[p], "parabola3")[0] for p in rates}
    cells = []
    for d in (-2, -1, 1, 2):
        ratios = []
        for p in rates:
            ph = p * 2.0 ** d
            match = [q for q in rates if abs(np.log(q / ph)) < 1e-6]
            if not match:
                continue
            steps, y = C[p]
            f = lambda e: np.interp(np.log(e), np.log(steps), y)
            if not (steps[0] <= opt[match[0]] <= steps[-1]) or f(opt[p]) <= 0:
                continue
            ratios.append(f(opt[match[0]]) / f(opt[p]))
        cells.append(f"{np.mean(ratios):.3f} (n={len(ratios)})" if ratios else "--")
    print(f"  {name:13s} " + " ".join(f"{c:>14s}" for c in cells))
