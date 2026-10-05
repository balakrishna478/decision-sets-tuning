# Stationary (no-change) Oja fluctuations vs the linearised AR(1) prediction.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, revsim as R
M, K, k, snr = int(sys.argv[1]) if len(sys.argv) > 1 else 16, 6, 2, 8.0
kappa = snr * k + M
def run(eta, seed, T=40000, burn=8000):
    env = R.Env("swap", M, K, k, 0.0, snr, seed + 7)
    U = env.subspace(); C = env.cbar(); lam = np.sort(np.linalg.eigvalsh(C))[::-1][:k]
    rng = np.random.default_rng(seed); trk = R.Oja(M, k, kappa, rng, False, eta=eta)
    trk.W = R.qf(U + 0.01 * rng.standard_normal((M, k)))
    h = np.random.default_rng(seed + 999).standard_normal(M); h /= np.linalg.norm(h)
    SA = env.S[:, env.act]; F = S2 = I = 0.0; I_all = []
    for t in range(T):
        y, _ = env.step(); trk.update(y, False, env)
        if t < burn: continue
        W = trk.W; sv = np.linalg.svd(W.T @ U, compute_uv=False); s2 = 1 - sv ** 2
        F += s2.sum(); S2 += np.sqrt(s2.max())
        w = h - W @ (W.T @ h); w /= np.linalg.norm(w); It = float(np.sum((SA.T @ w) ** 2)); I_all.append(It)
    n = T - burn; I_all = np.array(I_all)
    g = eta / kappa
    # exact AR(1) variance per (noise i, signal j) entry, noise eigenvalue 1
    var = np.array([g ** 2 * l / (1 - (1 - g * (l - 1)) ** 2) for l in lam])
    pred_F = (M - k) * var.sum()
    wts = var * (lam - 1) / snr            # I = sum_j (lam_j-1)/snr * z_j^2
    pred_I = wts.sum()
    # weighted chi-square(1) tail by simulation of the predicted law
    zz = np.random.default_rng(1).standard_normal((400000, k)) ** 2 @ wts
    out = {G: (np.mean(I_all > G), np.mean(zz > G)) for G in (0.001, 0.003, 0.01)}
    return dict(eta=eta, seed=seed, F=F / n, predF=pred_F, I=I_all.mean(), predI=pred_I,
                closedI=eta * k * (1 + snr) / (2 * kappa * snr), sin2=S2 / n, out=out)
for eta in [0.048, 0.192, 0.768]:
    for s in [0, 1]:
        r = run(eta, s)
        print(f"M={M} eta={eta} seed={s}: |sinT|F^2 {r['F']:.5f} pred {r['predF']:.5f} | E[I] {r['I']:.6f} pred {r['predI']:.6f} closed {r['closedI']:.6f} | mean|sinT|2 {r['sin2']:.4f} |",
              " ".join(f"P(I>{G}) {a:.4f}/{b:.4f}" for G, (a, b) in r['out'].items()), flush=True)
