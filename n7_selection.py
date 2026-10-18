#!/usr/bin/env python3
"""N7 (variable rank): rescore channel selection on the slots a width-kw tracker can represent.

In about 1.6% of slots (up to 8% for some arrays) more users are active than the tracked width
kw = 4; there collisions are partly unavoidable whatever the step, and a few such episodes per
seed make the selection optimum unstable. On slots with at most kw active users the oracle never
collides, so the gap there is 3 x the tracker's collision rate. This writes a copy of the N7
results with `gap` replaced by that quantity; null_analyze.py then fits it as usual.

  python3 n7_selection.py results/ n7_rep/         # reads results/N7_varrank_*/results.jsonl
  python3 null_analyze.py n7_rep --max-p 2e-3      # 'argmin collisions' = representable slots

N7b results also carry I_mean_rep and out_*_rep (null-steering costs on the same slots); for those,
the copy uses them in place of I_mean and out_* as well:
  python3 n7_selection.py results/ n7b_rep/        # picks up N7b_varrank_rep_*/results.jsonl too
"""
import glob, json, os, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(dst, "N7"), exist_ok=True)
n = 0
with open(os.path.join(dst, "N7", "results.jsonl"), "w") as out:
    for fn in glob.glob(os.path.join(src, "*", "results.jsonl")) + glob.glob(os.path.join(src, "*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            if r.get("exp") not in ("N7", "N7b") or "error" in r:
                continue
            kw, sb, cb = r.get("kw", 4), r["slots_by_rank"], r["coll_by_rank"]
            rep = [k for k in sb if int(k) <= kw]
            r["gap"] = 3 * sum(sb[k] * cb[k] for k in rep) / sum(sb[k] for k in rep)
            if "I_mean_rep" in r:                            # N7b: null-steering on the same slots
                r["I_mean"] = r["I_mean_rep"]
                for g in ("0.01", "0.03", "0.1"):
                    r[f"out_{g}"] = r[f"out_{g}_rep"]
            out.write(json.dumps(r) + "\n"); n += 1
print(f"{n} N7 runs rescored -> {dst}/N7/results.jsonl")
