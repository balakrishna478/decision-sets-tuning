#!/usr/bin/env python3
"""
Tuning-law analysis for N1 (Oja, M=16), N2 (EW covariance and PAST, M=16) and N3 (all three, M=32, 64).

  python3 null_analyze.py                     # everything under results/
  python3 null_analyze.py --max-p 2e-3        # drop p=3.2e-3 (jumps overlap recoveries)
  python3 null_analyze.py --group oja:16      # one tracker/array only

The step axis is eta for Oja and the forgetting rate mu for the others (EW covariance weight
beta = mu; PAST forgetting factor beta = 1 - mu).  For each cost the optimal step at each jump rate
comes from a parabola in log(step) through the grid minimum and its neighbours; seeds are
bootstrapped jointly across rates (seed s is the same array at every rate).

Two candidate laws are fitted for every cost:
  power law       step* ~ p^a                        (smooth costs: tracking 2/3, interference 1/2)
  log law         1/step* = c0 - c1 * log p          (threshold costs: Gaussian-tail stationary term)
For the outage costs the log-law slope times Gamma should be roughly constant if the tail model holds.
"""
import argparse, glob, json, os, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
METRICS = [("sin_mean", "tracking error", 2 / 3), ("I_mean", "mean interference", 1 / 2),
           ("out_0.01", "outage, Gamma=0.01", None), ("out_0.03", "outage, Gamma=0.03", None),
           ("out_0.1", "outage, Gamma=0.1", None), ("gap", "argmin collisions", None)]
B = 1000


def load(resdir, pmin, pmax):
    rows, seen = [], set()
    files = glob.glob(os.path.join(resdir, "*.jsonl")) + glob.glob(os.path.join(resdir, "*", "results.jsonl"))
    for fn in files:
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("exp") not in ("N1", "N2", "N3", "N4", "N5", "N6") or "error" in r or r["id"] in seen:
                continue
            if not (pmin <= r["p"] <= pmax):
                continue
            seen.add(r["id"])
            m = r["method"]
            r["step"] = r["eta"] if m == "oja" else (round(1 - r["beta"], 6) if m == "past" else r["beta"])
            r["group"] = f"{m}:{r.get('M', 16)}" + ("" if r.get("array", "gauss") == "gauss" else ":ula") \
                + ("" if r.get("snr", 8.0) == 8.0 else f":snr{r['snr']:g}") \
                + ("" if r.get("k", 2) == 2 else f":k{r['k']}")
            rows.append(r)
    return rows


def step_star(steps, vals):
    """(step*, excluded) using only the steps that have data at this rate."""
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    steps, vals = steps[ok], vals[ok]
    if len(vals) == 0 or np.max(vals) <= 0:
        return np.nan, True
    i = int(np.argmin(vals))
    ties = np.flatnonzero(vals <= vals[i] * (1 + 1e-9))
    if len(ties) > 1:
        return float(np.exp(np.mean(np.log(steps[ties])))), True
    if i == 0 or i == len(vals) - 1:
        return float(steps[i]), True
    x = np.log(steps[i - 1:i + 2]); y = vals[i - 1:i + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(steps[i]), False
    return float(np.exp(np.clip(-b / (2 * a), x[0], x[2]))), False


def r2(x, y):
    c = np.polyfit(x, y, 1)
    res = y - np.polyval(c, x)
    return (1 - res.var() / y.var() if y.var() > 0 else np.nan), c


def analyse(rows, name, rng):
    rates = sorted({r["p"] for r in rows})
    steps = np.array(sorted({r["step"] for r in rows}))
    seeds = sorted({r["seed"] for r in rows})
    print("\n" + "=" * 86)
    print(f"{name}: {len(rows)} results, {len(rates)} rates, {len(steps)} steps, {len(seeds)} seeds")
    print("=" * 86)
    out = {}
    for m, label, pred in METRICS:
        cube = {}
        for p in rates:
            A = np.full((len(seeds), len(steps)), np.nan)
            for r in rows:
                if r["p"] == p and m in r:
                    A[seeds.index(r["seed"]), int(np.searchsorted(steps, r["step"]))] = r[m]
            cube[p] = A
        est, excl = [], []
        boots = np.full((B, len(rates)), np.nan)
        for j, p in enumerate(rates):
            e, ex = step_star(steps, np.nanmean(cube[p], axis=0))
            est.append(e); excl.append(ex)
            for b in range(B):
                idx = rng.integers(0, len(seeds), len(seeds))
                boots[b, j] = step_star(steps, np.nanmean(cube[p][idx], axis=0))[0]
        est = np.array(est)
        line = "  ".join(f"{p:g}:{e:.4g}{'*' if x else ''}" for p, e, x in zip(rates, est, excl))
        print(f"\n  {label}\n    step* by rate  {line}" + ("   (* edge/flat: excluded)" if any(excl) else ""))
        ok = [j for j in range(len(rates)) if not excl[j] and np.isfinite(est[j])]
        res = dict(est=dict(zip(rates, est)), a=np.nan, lo=np.nan, hi=np.nan, r2p=np.nan, r2l=np.nan, c1=np.nan)
        if len(ok) >= 3:
            lp = np.log(np.array(rates)[ok])
            r2p, cp = r2(lp, np.log(est[ok]))
            r2l, cl = r2(lp, 1 / est[ok])
            ab = [np.polyfit(lp, np.log(boots[b, ok]), 1)[0] for b in range(B)
                  if np.all(np.isfinite(boots[b, ok]))]
            lo, hi = np.percentile(ab, [2.5, 97.5]) if ab else (np.nan, np.nan)
            res.update(a=cp[0], lo=lo, hi=hi, r2p=r2p, r2l=r2l, c1=-cl[0])
            tag = ""
            if pred is not None:
                tag = ("  consistent with prediction" if lo <= pred <= hi
                       else f"  differs from {pred:.2f} by {cp[0] - pred:+.3f}")
            print(f"    power law: a = {cp[0]:.3f} [{lo:.3f}, {hi:.3f}]  R2 {r2p:.4f}{tag}")
            print(f"    log law:   1/step* slope {-cl[0]:.4g} per unit log(1/p)   R2 {r2l:.4f}"
                  + ("   <- fits better" if r2l > r2p else ""))
        else:
            print("    too few interior optima to fit")
        out[m] = res
    c = [(float(g), out[f"out_{g}"]["c1"]) for g in ("0.01", "0.03", "0.1") if np.isfinite(out[f"out_{g}"]["c1"])]
    if len(c) >= 2:
        print("\n  log-law slope x Gamma (roughly constant if the Gaussian-tail model holds): "
              + ", ".join(f"Gamma={g:g}: {g * s:.4f}" for g, s in c))
    us = [r["update_us"] for r in rows if "update_us" in r]
    if us:
        print(f"\n  update cost: median {np.median(us):.1f} us per sample (tracker update only)")
    out["_update_us"] = float(np.median(us)) if us else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--max-p", type=float, default=1.0)
    ap.add_argument("--group", default="", help="e.g. oja:16,ewcov:64")
    a = ap.parse_args()
    rows = load(a.results, a.min_p, a.max_p)
    if not rows:
        raise SystemExit("no N1-N5 results found under " + a.results)
    groups = sorted({r["group"] for r in rows}, key=lambda g: (int(g.split(":")[1]), len(g), g))
    if a.group:
        groups = [g for g in groups if g in a.group.split(",")]
    rng = np.random.default_rng(0)
    allres = {g: analyse([r for r in rows if r["group"] == g], g, rng) for g in groups}

    print("\n" + "=" * 86)
    print("SUMMARY: power-law exponent a (step* ~ p^a) [95% interval], and the better-fitting law")
    print("=" * 86)
    print(f"  {'cost':20s}" + "".join(f"{g:>24s}" for g in groups))
    for m, label, _ in METRICS:
        cells = []
        for g in groups:
            r = allres[g][m]
            if np.isfinite(r["a"]):
                law = "log" if r["r2l"] > r["r2p"] else "pow"
                cells.append(f"{r['a']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] {law}")
            else:
                cells.append("--")
        print(f"  {label:20s}" + "".join(f"{c:>24s}" for c in cells))
    print(f"  {'update cost (us)':20s}" + "".join(f"{allres[g]['_update_us']:>24.1f}" for g in groups))
    print("\n  step*(cost) / step*(tracking error), slowest -> fastest rate:")
    for g in groups:
        tr = allres[g]["sin_mean"]["est"]
        rs = sorted(tr)
        parts = []
        for m, label, _ in METRICS[1:]:
            e = allres[g][m]["est"]
            if all(np.isfinite(e.get(p, np.nan)) and np.isfinite(tr[p]) and tr[p] > 0 for p in (rs[0], rs[-1])):
                parts.append(f"{label}: {e[rs[0]] / tr[rs[0]]:.1f} -> {e[rs[-1]] / tr[rs[-1]]:.1f}")
        print(f"   {g:9s} " + "; ".join(parts))


if __name__ == "__main__":
    main()
