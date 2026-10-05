import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Collision slots per change against step size: the argmin optimum is where this bottoms out.
import numpy as np, revsim as R
for eta in [0.384, 0.543, 0.768, 1.086, 1.536, 2.172]:
    cs = []
    for seed in range(3):
        r = R.run(dict(id="x", method="oja", eta=eta, p=1e-3, T=50000, seed=seed, sin_every=0))
        cs.append(r["coll_rate"] / 1e-3)          # collision slots per change
    print(f"eta {eta:.3f}: collision slots per change {np.mean(cs):.2f}  (x eta = {np.mean(cs)*eta:.2f})", flush=True)
