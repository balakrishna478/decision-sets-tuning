#!/usr/bin/env python3
"""
Analysis of the revision runs.

  python3 rev_analyze.py                       # reads results/ (lab.sh run folders or worker files)
  python3 rev_analyze.py --exp E2,E3           # subset

All comparisons are paired by seed.  Each reports mean difference, wins, a paired
t-test and a Wilcoxon signed-rank test (gaps are zero-inflated at low jump rates, so
the rank test is the one to quote there).  Holm correction is applied across every
comparison printed in one invocation, as in the paper.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

try:
    from scipy import stats
except Exception:                                  # analysis needs scipy
    raise SystemExit("rev_analyze.py needs scipy (pip install scipy)")

TESTS = []                                          # (label, raw p) for Holm
RESDIR = "results"


# ------------------------------------------------------------------ helpers
def load(resdir="results"):
    rows, seen = [], set()
    files = sorted(glob.glob(os.path.join(resdir, "*.jsonl")))             # standalone workers
    files += sorted(glob.glob(os.path.join(resdir, "*", "results.jsonl")))  # lab.sh run folders
    for fn in files:
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r:
                print("ERROR in", r["id"], r["error"]); continue
            if r["id"] in seen:
                continue
            seen.add(r["id"]); rows.append(r)
    return rows


def ci(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return float(a.mean()) if len(a) else float("nan"), float("nan")
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def fmt(a, d=4):
    m, h = ci(a)
    return f"{m:.{d}f} ± {h:.{d}f}"


def by_seed(rows, metric):
    return {r["seed"]: r[metric] for r in rows if metric in r}


def paired(label, A, B, metric, record=True, higher_better=True):
    """A, B: lists of rows.  Returns printed line."""
    a, b = by_seed(A, metric), by_seed(B, metric)
    s = sorted(set(a) & set(b))
    if len(s) < 3:
        return f"{label}: too few paired seeds ({len(s)})"
    d = np.array([a[i] - b[i] for i in s])
    wins = int((d > 0).sum() if higher_better else (d < 0).sum())
    pt = float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else 1.0
    pw = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    if record:
        TESTS.append((label, pw))
    return (f"{label}: diff {d.mean():+.5f}, first better on {wins}/{len(s)}, "
            f"t p={pt:.2g}, Wilcoxon p={pw:.2g}")


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, ps[i] * (len(ps) - rank)); adj[i] = min(run, 1.0)
    return adj


def sel(rows, **kw):
    out = []
    for r in rows:
        if all(r.get(k, DEFAULTS.get(k)) == v for k, v in kw.items()):
            out.append(r)
    return out


DEFAULTS = dict(kind="swap", array="gauss", snr=8.0, M=16, sched="const", driver="sample",
                floor=0.0, delay=0, fa=0.0, rankmode=None)
RATES = [5e-5, 2e-4, 8e-4]


def hdr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ------------------------------------------------------------------ E3: step sweep
def e3(rows):
    hdr("E3  constant-step sweep: tracking optimum vs reward optimum")
    opt = {}
    for p in RATES:
        R = sel(rows, exp="E3", p=p)
        etas = sorted({r["eta"] for r in R})
        if not etas:
            continue
        print(f"\np = {p:g}   (seeds per cell: {len(sel(R, eta=etas[0]))})")
        print(f"{'eta':>7} {'mean sinTheta':>18} {'gap to oracle':>20} {'frac>0.95':>10} {'frac>gamma':>10}")
        sm, gm = {}, {}
        for e in etas:
            C = sel(R, eta=e)
            sm[e] = np.mean([r["sin_mean"] for r in C]); gm[e] = np.mean([r["gap"] for r in C])
            print(f"{e:7.3f} {fmt([r['sin_mean'] for r in C]):>18} {fmt([r['gap'] for r in C], 5):>20}"
                  f" {np.mean([r['frac_above_095'] for r in C]):10.4f}"
                  f" {np.mean([r['frac_above_gamma'] for r in C]):10.4f}")
        et, er = min(sm, key=sm.get), min(gm, key=gm.get)
        opt[p] = (et, er)
        print(f"tracking optimum eta={et}, reward optimum eta={er}  (ratio {er / et:g})"
              + ("   ** reward optimum at edge of grid **" if er == etas[-1] else ""))
        i, j = etas.index(et), etas.index(er)
        for nb in [etas[i - 1] if i > 0 else None, etas[i + 1] if i + 1 < len(etas) else None]:
            if nb is not None:
                print("  " + paired(f"sin: eta={et} vs {nb} (p={p:g})", sel(R, eta=et), sel(R, eta=nb),
                                    "sin_mean", higher_better=False))
        for nb in [etas[j - 1] if j > 0 else None, etas[j + 1] if j + 1 < len(etas) else None]:
            if nb is not None:
                print("  " + paired(f"gap: eta={er} vs {nb} (p={p:g})", sel(R, eta=er), sel(R, eta=nb),
                                    "gap", higher_better=False))
        print("  " + paired(f"gap: reward-opt {er} vs tracking-opt {et} (p={p:g})",
                            sel(R, eta=er), sel(R, eta=et), "gap", higher_better=False))
    return opt


# ------------------------------------------------------------------ E1: baselines
def e1(rows, opt):
    hdr("E1  baselines vs best constant-step Oja (same seeds, same environment)")
    for p in RATES:
        if p not in opt:
            continue
        best_oja = sel(rows, exp="E3", p=p, eta=opt[p][1])
        print(f"\np = {p:g}.  Oja eta={opt[p][1]}: gap {fmt([r['gap'] for r in best_oja], 5)}")
        for m in ["bf", "past", "ewcov"]:
            R = sel(rows, exp="E1", p=p, method=m)
            betas = sorted({r["beta"] for r in R})
            if not betas:
                continue
            g = {b: np.mean([r["gap"] for r in sel(R, beta=b)]) for b in betas}
            for b in betas:
                C = sel(R, beta=b)
                extra = f"  sin {np.mean([r['sin_mean'] for r in C]):.3f}" if "sin_mean" in C[0] else ""
                print(f"  {m:6s} beta={b:<6g} gap {fmt([r['gap'] for r in C], 5)}{extra}")
            bb = min(g, key=g.get)
            print("    " + paired(f"{m}(beta={bb}) vs Oja(eta={opt[p][1]}), p={p:g}",
                                  sel(R, beta=bb), best_oja, "gap", higher_better=False))


# ------------------------------------------------------------------ E2: resets
def e2(rows, opt):
    hdr("E2  reset-on-detection schedules, judged on reward AND tracking error")
    for p in RATES:
        R = sel(rows, exp="E2", p=p, sched="reset")
        if not R or p not in opt:
            continue
        ref_r = sel(rows, exp="E3", p=p, eta=opt[p][1])
        ref_t = sel(rows, exp="E3", p=p, eta=opt[p][0])
        print(f"\np = {p:g}.  references: const reward-opt eta={opt[p][1]}, tracking-opt eta={opt[p][0]}")
        keys = sorted({(r["eta0"], r.get("floor", 0.0), r.get("delay", 0), r.get("fa", 0.0)) for r in R})
        for (e0, fl, d, fa) in keys:
            C = sel(R, eta0=e0, floor=fl, delay=d, fa=fa)
            print(f"  eta0={e0:<4g} floor={fl:<6g} delay={d:<4d} fa={fa:<6g} "
                  f"sin {fmt([r['sin_mean'] for r in C], 3)}  gap {fmt([r['gap'] for r in C], 5)}"
                  f"  worst-seed gap {max(r['gap'] for r in C):.4f}")
        for (e0, fl, d, fa) in keys:
            C = sel(R, eta0=e0, floor=fl, delay=d, fa=fa)
            tag = f"reset(eta0={e0},floor={fl},delay={d},fa={fa})"
            if (fl == 0.0 and e0 == 2.0) or (fl > 0 and d == 0 and fa == 0) or (d == 100 and fa == 1e-4):
                print("    " + paired(f"gap {tag} vs const tracking-opt, p={p:g}", C, ref_t, "gap",
                                      higher_better=False))
                print("    " + paired(f"sin {tag} vs const tracking-opt, p={p:g}", C, ref_t, "sin_mean",
                                      higher_better=False))
                print("    " + paired(f"gap {tag} vs const reward-opt, p={p:g}", C, ref_r, "gap",
                                      higher_better=False))


# ------------------------------------------------------------------ E6: decaying step
def e6(rows):
    hdr("E6  decaying step eta0/t (Table I horizon) — theory: recovery needs t/t0 ~ R^(1/(eta0*Delta))")
    R = sel(rows, exp="E6", sched="decay")
    c = sel(rows, exp="E6", sched="const")
    if c:
        print(f"  const eta=0.02           sin {fmt([r['sin_mean'] for r in c], 3)}  gap {fmt([r['gap'] for r in c], 4)}")
    for e0 in sorted({r["eta0"] for r in R}):
        C = sel(R, eta0=e0)
        print(f"  decay eta0={e0:<5g}         sin {fmt([r['sin_mean'] for r in C], 3)}  "
              f"gap {fmt([r['gap'] for r in C], 4)}")


# ------------------------------------------------------------------ geometry + composition
def pcoll_curve(tri, grid):
    """swap-geometry collision probability vs sinTheta, averaged over admissible triples"""
    num, den = np.zeros(len(grid)), np.zeros(len(grid))
    for x in tri:
        th0 = x["theta0"]; ind = np.array(x["ind"]); n = len(ind)
        ok = np.arcsin(np.clip(grid, 0, 1)) <= th0
        idx = np.clip(np.round(np.arcsin(np.clip(grid, 0, 1)) / th0 * (n - 1)).astype(int), 0, n - 1)
        num += np.where(ok, ind[idx], 0); den += ok
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def closed_form_gap(tri, eta, p, delta_key="delta_k", delta_override=None):
    """gap = 3 p E[#collision slots per jump], theta(t) from the hitting-time ODE starting at
    the geometric theta0 -- no measured trajectory, nothing fitted to reward."""
    ncol = []
    for x in tri:
        th0 = x["theta0"]; ind = np.array(x["ind"]); n = len(ind)
        rate = eta * (delta_override if delta_override is not None else x[delta_key])
        tmax = int(np.log(np.tan(min(th0, 1.5707)) / np.tan(0.2)) / rate) + 2
        t = np.arange(tmax)
        th = np.arctan(np.tan(min(th0, 1.5707)) * np.exp(-rate * t))
        idx = np.clip(np.round(th / th0 * (n - 1)).astype(int), 0, n - 1)
        ncol.append(ind[idx].sum())
    return 3.0 * p * float(np.mean(ncol))


def fit_slopes(sin, jumps, lo=0.3, hi=0.9):
    out, peaks, t95 = [], [], []
    bounds = list(jumps) + [len(sin)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = sin[a:b].astype(float)
        if len(seg) < 30:
            continue
        pk = int(np.argmax(seg[:min(200, len(seg))]))
        peaks.append(seg[pk]); t95.append(int((seg > 0.95).sum()))
        # fit only the recovery leg: from the peak to the first drop below the floor, where the
        # floor sits above this run's stationary error (otherwise large-eta runs fit noise)
        floor = max(lo, float(np.median(seg[len(seg) // 2:])) + 0.15) if len(seg) > 200 else lo
        below = np.flatnonzero(seg[pk:] < floor)
        end = pk + (below[0] if len(below) else len(seg) - pk)
        m = np.zeros(len(seg), bool); m[pk:end] = True
        m &= (seg > floor) & (seg < hi)
        idx = np.flatnonzero(m)
        if len(idx) > 12:
            out.append(np.polyfit(idx, np.log(np.tan(np.arcsin(seg[idx]))), 1)[0])
    return np.array(out), np.array(peaks), np.array(t95)


def e5(rows, geo):
    hdr("E5  excursions: hitting-time rate, peak angle, time above 0.95, composed model")
    for p in [2e-4, 8e-4]:
        for drv in ["exact", "sample"]:
            R = sel(rows, exp="E5", p=p, driver=drv)
            for e in sorted({r["eta"] for r in R}):
                C = sel(R, eta=e)
                sl, pk, t95 = [], [], []
                for r in C:
                    fn = os.path.join(RESDIR, r.get("run_dir", ""),
                                      r.get("trace_file", f"traces/{r['id']}.npz"))
                    if not os.path.exists(fn):
                        continue
                    z = np.load(fn)
                    a, b, c = fit_slopes(z["sin"], z["jumps"])
                    sl += list(-a / e); pk += list(b); t95 += list(c)
                if not sl:
                    continue
                print(f"  p={p:g} {drv:6s} eta={e:<6g} Delta_meas {np.mean(sl):.4f} ± "
                      f"{np.std(sl) / np.sqrt(len(sl)):.4f} (n={len(sl)})  peak sin mean {np.mean(pk):.3f}"
                      f"  slots>0.95 per jump {np.mean(t95):.1f}  gap {np.mean([r['gap'] for r in C]):.5f}")
    if not geo:
        print("\n  (geometry.json missing: run `python3 revsim.py geometry` for the composed model)")
        return
    th0 = np.concatenate([[x["theta0"] for x in g["triples"]] for g in geo.values()])
    print(f"\n  geometric post-swap sin(theta0): mean {np.sin(th0).mean():.3f}, "
          f"range {np.sin(th0).min():.3f}-{np.sin(th0).max():.3f}  (compare 'peak sin' above)")


def composed(rows, geo):
    hdr("Composed model per seed: closed-form geometric prediction vs measured gap (E3 runs)")
    if not geo:
        print("  geometry.json missing"); return
    grid = np.linspace(0.30, 1.0, 141)
    for p in RATES:
        for e in [0.024, 0.048, 0.096, 0.192, 0.384]:
            C = sel(rows, exp="E3", p=p, eta=e)
            if len(C) < 3:
                continue
            meas, pk, pd1 = [], [], []
            for r in C:
                g = geo.get(str(r["seed"]))
                if g is None:
                    continue
                meas.append(r["gap"])
                pk.append(closed_form_gap(g["triples"], e, p, "delta_k"))
                pd1.append(closed_form_gap(g["triples"], e, p, "delta_1"))
            meas, pk, pd1 = map(np.array, (meas, pk, pd1))
            rho = stats.spearmanr(pk, meas).statistic if len(meas) > 3 else float("nan")
            print(f"  p={p:g} eta={e:<6g} measured {meas.mean():.5f}   predicted(Delta_k) {pk.mean():.5f}"
                  f"   predicted(Delta_1) {pd1.mean():.5f}   per-seed Spearman {rho:+.2f}")
    print("\n  Per-seed geometry vs measured gap (E3, reward-optimal eta per rate):")
    for p in RATES:
        C = sel(rows, exp="E3", p=p)
        if not C:
            continue
        er = min({r["eta"] for r in C}, key=lambda e: np.mean([r["gap"] for r in sel(C, eta=e)]))
        C = sel(C, eta=er)
        gm = np.array([r["gamma_min"] for r in C]); gp = np.array([r["gap"] for r in C])
        dk = np.array([geo[str(r["seed"])]["delta_k_mean"] for r in C if str(r["seed"]) in geo])
        print(f"   p={p:g} eta={er}: Spearman(gap, gamma_min) {stats.spearmanr(gp, gm).statistic:+.2f}"
              + (f", Spearman(gap, Delta) {stats.spearmanr(gp[:len(dk)], dk).statistic:+.2f}" if len(dk) > 3 else ""))
    s7 = geo.get("0")
    if s7:
        pc = pcoll_curve(s7["triples"], grid)
        print("\n  swap-geometry P_coll (seed offset 0 = paper's array):")
        for lo, hi in [(0.90, 0.93), (0.93, 0.95), (0.95, 0.965), (0.965, 0.98), (0.98, 0.99),
                       (0.99, 0.995), (0.995, 1.0)]:
            m = (grid >= lo) & (grid < hi)
            print(f"    [{lo:.3f},{hi:.3f})  {np.nanmean(pc[m]) if m.any() else float('nan'):.4f}")


# ------------------------------------------------------------------ E7: variable rank
def e7(rows):
    hdr("E7  variable occupancy: fixed width vs rank-adaptive vs beamformer")
    R = sel(rows, exp="E7", kind="var")
    if not R:
        return
    groups = {}
    for r in R:
        key = ("bf", r["beta"]) if r["method"] == "bf" else (r.get("rankmode") or "fixed", r.get("kw"))
        groups.setdefault(key, []).append(r)
    for key, C in sorted(groups.items(), key=lambda x: str(x[0])):
        slots, cols = defaultdict(float), defaultdict(float)
        for r in C:
            for rk, n in r.get("slots_by_rank", {}).items():
                slots[rk] += n; cols[rk] += n * r["coll_by_rank"][rk]
        cr = " ".join(f"r{rk}:{cols[rk] / slots[rk]:.3f}" for rk in sorted(slots) if int(rk) >= 3)
        print(f"  {str(key):22s} gap {fmt([r['gap'] for r in C], 4)}   collisions {cr}")
    base = groups.get(("fixed", 3))
    if base:
        for key, C in groups.items():
            if key != ("fixed", 3):
                print("    " + paired(f"var-rank {key} vs fixed k=3", C, base, "gap", higher_better=False))


# ------------------------------------------------------------------ E8: realism grid
def e8(rows):
    hdr("E8  array x SNR x M: best Oja vs best beamformer gap (p=2e-4)")
    R = sel(rows, exp="E8")
    for arr in ["gauss", "ula"]:
        for snr in [2.0, 4.0, 8.0, 16.0]:
            for M in [8, 16, 32]:
                C = [r for r in R if r.get("array", "gauss") == arr and r.get("snr", 8.0) == snr
                     and r.get("M", 16) == M]
                if not C:
                    continue
                oj = {e: [r for r in C if r["method"] == "oja" and r["eta"] == e] for e in
                      sorted({r["eta"] for r in C if r["method"] == "oja"})}
                bf = {b: [r for r in C if r["method"] == "bf" and r["beta"] == b] for b in
                      sorted({r["beta"] for r in C if r["method"] == "bf"})}
                eo = min(oj, key=lambda e: np.mean([r["gap"] for r in oj[e]]))
                bo = min(bf, key=lambda b: np.mean([r["gap"] for r in bf[b]]))
                gmin = np.mean([r["gamma_min"] for r in oj[eo]])
                print(f"  {arr:5s} SNR={snr:<4g} M={M:<3d} gamma_min {gmin:.3f}  Oja(eta={eo}) "
                      f"{fmt([r['gap'] for r in oj[eo]], 5)}   BF(beta={bo}) {fmt([r['gap'] for r in bf[bo]], 5)}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--geometry", default="geometry.json")
    ap.add_argument("--exp", default="E1,E2,E3,E5,E6,E7,E8,COMP")
    a = ap.parse_args()
    RESDIR = a.results
    rows = load(a.results)
    geo = json.load(open(a.geometry)) if os.path.exists(a.geometry) else None
    want = set(a.exp.split(","))
    print(f"{len(rows)} results loaded")
    opt = e3(rows) if want & {"E1", "E2", "E3", "COMP"} else {}
    if "E1" in want: e1(rows, opt)
    if "E2" in want: e2(rows, opt)
    if "E6" in want: e6(rows)
    if "E5" in want: e5(rows, geo)
    if "COMP" in want: composed(rows, geo)
    if "E7" in want: e7(rows)
    if "E8" in want: e8(rows)
    if TESTS:
        hdr(f"Holm correction across all {len(TESTS)} Wilcoxon tests printed above")
        adj = holm(np.array([p for _, p in TESTS]))
        for (lab, p), q in sorted(zip(TESTS, adj), key=lambda x: x[0][1]):
            print(f"  {p:9.2g} -> {q:7.2g}  {'*' if q < 0.05 else ' '} {lab}")
