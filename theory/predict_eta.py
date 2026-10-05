#!/usr/bin/env python3
"""Parameter-free prediction of the optimal Oja step eta*(p) for each decision cost (Section IV).

  python3 theory/predict_eta.py                     # M=16, SNR=8 (the paper's setting)
  python3 theory/predict_eta.py --snr 4             # compare with the N5 runs
  python3 theory/predict_eta.py --compare           # print measured N1 values beside the predictions

Ingredients, all computed from the array geometry and the data model, nothing fitted:
  excursion cost per change   A_f / (eta * Delta_k), integrated along the geodesic from the departing
                              to the arriving user (theta_0 from the array), using the actual null-steering
                              channel h of each seed;
  stationary cost             linearised Oja fluctuations: each tangent entry is AR(1) with variance
                              g^2 lam / (1 - (1 - g (lam - 1))^2), g = eta / kappa.
Argmin channel selection is not covered: its stationary collision probability is essentially zero, and
its optimum is set by where noisy recovery stops speeding up (theory/argmin_drag.py).
"""
import argparse, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import revsim as R

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=16); ap.add_argument("--snr", type=float, default=8.0)
ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--compare", action="store_true")
ap.add_argument("--k", type=int, default=2)
a = ap.parse_args()
M, K, k, snr = a.M, 6, a.k, a.snr
kappa = snr * k + M
GAMS = (0.01, 0.03, 0.1)
RATES = [2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
ETAS = np.exp(np.linspace(np.log(0.003), np.log(3.0), 90))
mc = np.random.default_rng(0)


def ar_var(g, lam):
    return g ** 2 * lam / (1 - (1 - g * (lam - 1)) ** 2)


def seed_parts(seed):
    env = R.Env("swap", M, K, k, 2e-4, snr, seed + 7); S = env.S
    h = np.random.default_rng(seed + 999).standard_normal(M); h /= np.linalg.norm(h)
    sets = list(itertools.combinations(range(K), k))
    lams = {s: np.sort(np.linalg.eigvalsh(snr * S[:, list(s)] @ S[:, list(s)].T + np.eye(M)))[::-1] for s in sets}
    exc_I, exc_out, paths = [], [], []
    for act in sets:
        for out in act:
            keep = [c for c in act if c != out]
            for inn in [c for c in range(K) if c not in act]:
                new = sorted(keep + [inn])
                Qk = np.linalg.qr(S[:, keep])[0]; Pc = np.eye(M) - Qk @ Qk.T
                u0 = Pc @ S[:, out]; u0 /= np.linalg.norm(u0)
                b = Pc @ S[:, inn]; b /= np.linalg.norm(b)
                c0 = float(u0 @ b); b = b * np.sign(c0); c0 = abs(c0)
                th0 = np.arccos(min(c0, 1.0)); v = b - c0 * u0; v /= np.linalg.norm(v)
                th = np.linspace(1e-4, th0, 400); phi = th0 - th
                Up = np.cos(phi)[:, None] * u0 + np.sin(phi)[:, None] * v
                hq = h - Qk @ (Qk.T @ h)
                w = hq[None] - Up * (Up @ hq)[:, None]; w /= np.linalg.norm(w, axis=1, keepdims=True)
                I = ((w @ S[:, new]) ** 2).sum(1)
                lam = lams[tuple(new)]; delta = (lam[k - 1] - lam[k]) / kappa
                kern = 1 / (np.sin(th) * np.cos(th)) / delta
                exc_I.append(np.trapezoid(I * kern, th))
                exc_out.append([np.trapezoid((I > G) * kern, th) for G in GAMS])
                paths.append((th, kern))
    return sets, lams, np.mean(exc_I), np.mean(exc_out, 0), paths


def stationary(eta, sets, lams):
    g = eta / kappa; acc = I = 0.0; outs = np.zeros(len(GAMS))
    for s in sets:
        lam = lams[s][:k]; var = ar_var(g, lam); wts = var * (lam - 1) / snr
        I += wts.sum()
        z = mc.standard_normal((20000, k)) ** 2 @ wts
        outs += np.array([np.mean(z > G) for G in GAMS])
        E = mc.standard_normal((1500, M - k, k)) * np.sqrt(var)
        sv = np.linalg.svd(E, compute_uv=False)[:, 0]
        acc += np.mean(sv / np.sqrt(1 + sv ** 2))
    n = len(sets)
    return acc / n, I / n, outs / n


J = {m: np.zeros((len(RATES), len(ETAS))) for m in ("acc", "int", "out0.01", "out0.03", "out0.1")}
for seed in range(a.seeds):
    sets, lams, AI, AO, paths = seed_parts(seed)
    st = [stationary(e, sets, lams) for e in ETAS]
    acc_s = np.array([x[0] for x in st]); I_s = np.array([x[1] for x in st]); O_s = np.array([x[2] for x in st])
    A_acc = np.array([np.mean([np.trapezoid(np.clip(np.sin(th) - s0, 0, None) * kern, th) for th, kern in paths])
                      for s0 in acc_s])
    for i, p in enumerate(RATES):
        J["acc"][i] += p * A_acc / ETAS + acc_s
        J["int"][i] += p * AI / ETAS + I_s
        for j, G in enumerate(GAMS):
            J[f"out{G}"][i] += p * AO[j] / ETAS + O_s[:, j]
    print(f"  seed {seed} done", file=sys.stderr, flush=True)


def opt(y):
    i = int(np.argmin(y)); i = min(max(i, 1), len(y) - 2)
    x = np.log(ETAS[i - 1:i + 2]); c2, c1, _ = np.polyfit(x, y[i - 1:i + 2], 2)
    return float(np.exp(-c1 / (2 * c2))) if c2 > 0 else float(ETAS[i])


MEAS = {"acc": [.01016, .015, .024, .036, .058, .085, .126], "int": [.020, .028, .038, .054, .078, .107, .148],
        "out0.01": [.036, .041, .045, .052, .062, .072, .084], "out0.03": [.083, .089, .098, .115, .126, .144, .166],
        "out0.1": [.193, .226, .234, .244, .265, .308, .336]}
print(f"Predicted eta* (Oja, M={M}, k={k}, SNR={snr:g}, {a.seeds} arrays)")
print("cost      " + " ".join(f"{p:>8g}" for p in RATES) + "   exponent a")
for m in J:
    pred = np.array([opt(J[m][i]) for i in range(len(RATES))])
    ap_ = np.polyfit(np.log(RATES), np.log(pred), 1)[0]
    print(f"{m:9s} " + " ".join(f"{x:8.4f}" for x in pred) + f"   {ap_:.3f}")
    if a.compare and M == 16 and snr == 8.0 and k == 2:
        me = np.array(MEAS[m]); am = np.polyfit(np.log(RATES), np.log(me), 1)[0]
        print(f"{'  meas':9s} " + " ".join(f"{x:8.4f}" for x in me) + f"   {am:.3f}   (pred/meas {np.min(pred/me):.2f}-{np.max(pred/me):.2f})")
