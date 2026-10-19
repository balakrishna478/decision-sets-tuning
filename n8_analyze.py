#!/usr/bin/env python3
"""N8 online tuning: cost of the law-based step relative to the best constant step.

For every setting (fixed rate, or switching rate) and target cost:
  ratio = J(rule) / J(best constant step on the same arrays)
where J is the seed-mean cost, the best constant step is located on the seed-mean grid curve with
the same rule as null_analyze.py, and costs between grid steps are interpolated in log-log.
95% intervals resample arrays (seeds) with replacement, B = 1000 (paired: the rule and the grid
use the same arrays). For switching rates it also reports the per-regime oracle: the best constant
step chosen separately for each regime in hindsight.

  python3 n8_analyze.py results/            # reads results/N8_online_*/results.jsonl
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from null_analyze import step_star  # noqa: E402

COSTS = [("sin_mean", "accuracy"), ("I_mean", "mean interference"), ("out_0.01", "outage 0.01"),
         ("out_0.03", "outage 0.03"), ("out_0.1", "outage 0.1"), ("gap", "argmin selection")]
MODES = ("known", "genie", "blind")
B = 1000
rng = np.random.default_rng(0)


def load(src):
    R, seen = [], set()
    for fn in glob.glob(os.path.join(src, "*", "results.jsonl")) + glob.glob(os.path.join(src, "*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            if r.get("exp") == "N8" and "error" not in r and r["id"] not in seen:
                seen.add(r["id"]); R.append(r)
    return R


def interp(steps, v, s):
    return float(np.exp(np.interp(np.log(s), np.log(steps), np.log(v))))


def best(steps, v):
    """Best constant-step cost: the minimum of a parabola in (log step, log cost) through the grid
    minimum and its neighbours. Interpolating between grid points would overstate the minimum of a
    convex curve and flatter the rule, so the parabola's own minimum is used (never above the grid
    minimum); at an edge or a flat point, the grid minimum."""
    e, edge = step_star(steps, v)
    ok = np.isfinite(v) & (v > 0)
    st, vv = steps[ok], v[ok]
    i = int(np.argmin(vv))
    if edge or i == 0 or i == len(vv) - 1:
        return float(vv[i]), e, edge
    x, y = np.log(st[i - 1:i + 2]), np.log(vv[i - 1:i + 2])
    a2, b1, c0 = np.polyfit(x, y, 2)
    if a2 <= 0:
        return float(vv[i]), e, edge
    xm = np.clip(-b1 / (2 * a2), x[0], x[2])
    return float(min(np.exp(np.polyval([a2, b1, c0], xm)), vv[i])), e, edge


def cube(rows, steps, seeds, f):
    A = np.full((len(seeds), len(steps)), np.nan)
    for r in rows:
        A[seeds.index(r["seed"]), int(np.searchsorted(steps, r["eta"]))] = f(r)
    return A


def main():
    R = load(sys.argv[1] if len(sys.argv) > 1 else "results")
    key = lambda r: (r["p"], r.get("p_hi"))
    for st in sorted({key(r) for r in R}, key=lambda k: (k[1] is not None, k[0])):
        rows = [r for r in R if key(r) == st]
        grid = [r for r in rows if not r.get("rate")]
        rules = [r for r in rows if r.get("rate")]
        steps = np.array(sorted({r["eta"] for r in grid})); seeds = sorted({r["seed"] for r in grid})
        sw = st[1] is not None
        print("\n" + "=" * 92)
        print(f"{'switching ' + format(st[0], 'g') + ' <-> ' + format(st[1], 'g') if sw else 'fixed p = ' + format(st[0], 'g')}:"
              f" {len(grid)} grid runs, {len(rules)} rule runs, {len(seeds)} arrays")
        print("=" * 92)
        for c, lab in COSTS:
            G = cube(grid, steps, seeds, lambda r: r[c])
            Jb, eb, edge = best(steps, np.nanmean(G, 0))
            line = f"  {lab:18s} best constant {eb:.3g}{' (edge)' if edge else ''}"
            if sw:   # per-regime oracle (equal dwell, weight by slots)
                G0 = cube(grid, steps, seeds, lambda r: r["reg0"][c]); G1 = cube(grid, steps, seeds, lambda r: r["reg1"][c])
                n0 = np.mean([r["reg0"]["n"] for r in grid]); n1 = np.mean([r["reg1"]["n"] for r in grid])
                Jo = (n0 * best(steps, np.nanmean(G0, 0))[0] + n1 * best(steps, np.nanmean(G1, 0))[0]) / (n0 + n1)
                line += f" | regime oracle / best constant {Jo / Jb:.3f}"
            print(line)
            for md in MODES:
                rr = {r["seed"]: r for r in rules if r["target"] == c and r["rate"] == md}
                if not rr:
                    continue
                ss = [s for s in seeds if s in rr]
                rv = np.array([rr[s][c] for s in ss]); Gs = G[[seeds.index(s) for s in ss]]
                ratio = rv.mean() / best(steps, np.nanmean(Gs, 0))[0]
                bs = []
                for _ in range(B):
                    i = rng.integers(0, len(ss), len(ss))
                    bs.append(rv[i].mean() / best(steps, np.nanmean(Gs[i], 0))[0])
                lo, hi = np.percentile(bs, [2.5, 97.5])
                eg = np.exp(np.mean([np.log(rr[s]["eta_geo"]) for s in ss]))
                pb = np.exp(np.mean([rr[s]["phat_logbias"] for s in ss]))
                extra = ""
                if sw:
                    J0 = np.mean([rr[s]["reg0"][c] for s in ss]); J1 = np.mean([rr[s]["reg1"][c] for s in ss])
                    B0 = best(steps, np.nanmean(cube(grid, steps, seeds, lambda r: r["reg0"][c])[[seeds.index(s) for s in ss]], 0))[0]
                    B1 = best(steps, np.nanmean(cube(grid, steps, seeds, lambda r: r["reg1"][c])[[seeds.index(s) for s in ss]], 0))[0]
                    extra = f"  per regime vs its own best: lo {J0 / B0:.3f}, hi {J1 / B1:.3f}"
                print(f"      {md:5s}: rule / best constant {ratio:.3f} [{lo:.3f}, {hi:.3f}]   mean step {eg:.3g}"
                      f"   p_hat/p x{pb:.2f}{extra}")
        bl = [r for r in rules if r["rate"] == "blind"]
        if bl:
            print(f"  blind detector: alarms / true changes in the scored window "
                  f"{sum(r['alarms_eval'] for r in bl) / max(sum(r['n_jumps_eval'] for r in bl), 1):.2f}")


if __name__ == "__main__":
    main()
