"""Fig. 3 (parity: predicted vs measured eta*) and Fig. 4 (mechanism: excursion + stationary = total).
Inputs: pred_eta.npy, pred_parts.npy from theory/predict_eta.py internals (20 arrays, M=16, SNR=8)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
P = [2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
MEAS = {"acc": [.01016, .015, .024, .036, .058, .085, .126], "int": [.020, .028, .038, .054, .078, .107, .148],
        "out0.01": [.036, .041, .045, .052, .062, .072, .084], "out0.03": [.083, .089, .098, .115, .126, .144, .166],
        "out0.1": [.193, .226, .234, .244, .265, .308, .336]}
LAB = {"acc": "Accuracy", "int": "Mean interference", "out0.01": r"Outage, $\Gamma=0.01$",
       "out0.03": r"Outage, $\Gamma=0.03$", "out0.1": r"Outage, $\Gamma=0.1$"}
MK = {"acc": "o", "int": "s", "out0.01": "^", "out0.03": "v", "out0.1": "D"}
pred = np.load("pred_eta.npy", allow_pickle=True).item()

# ---------------- Fig. 3 parity
fig, ax = plt.subplots(figsize=(3.45, 2.9))
allp, allm = [], []
for k in MEAS:
    m, q = np.array(MEAS[k]), np.array(pred[k])
    ax.loglog(m, q, MK[k], ms=3.6, mfc="none", mew=0.9, label=LAB[k])
    allp += list(q); allm += list(m)
bias = np.exp(np.mean(np.log(np.array(allp) / np.array(allm))))
lims = [0.008, 0.5]
ax.plot(lims, lims, "-", color="0.3", lw=0.8, label="$y=x$")
ax.plot(lims, [l * bias for l in lims], "--", color="0.55", lw=0.8, label=f"$y={bias:.2f}x$ (mean bias)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r"measured $\eta^\star$"); ax.set_ylabel(r"predicted $\eta^\star$")
ax.grid(True, which="major", lw=0.3, color="0.85")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=6.2, handletextpad=0.3, labelspacing=0.25)
fig.tight_layout(pad=0.3); fig.savefig("fig3_parity.pdf"); fig.savefig("fig3_parity.png", dpi=200)
print("mean multiplicative bias", round(bias, 3))

# ---------------- Fig. 4 mechanism at p = 2e-4
parts = np.load("pred_parts.npy", allow_pickle=True).item()
S = list(parts.values())
ETAS = np.exp(np.linspace(np.log(0.004), np.log(3.0), 80))       # grid used when the parts were computed
p = 2e-4; ip = P.index(p)
panels = [("acc", "Accuracy", lambda s: p * s["A_acc"] / ETAS, lambda s: s["acc_s"]),
          ("int", "Mean interference", lambda s: p * s["A_I"] / ETAS, lambda s: s["I_s"]),
          ("out0.03", r"Outage, $\Gamma=0.03$", lambda s: p * s["A_out"][1] / ETAS, lambda s: s["out_s"][:, 1])]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15))
for ax, (k, title, exc, sta) in zip(axs, panels):
    E = np.mean([exc(s) for s in S], 0); St = np.mean([sta(s) for s in S], 0); J = E + St
    m = (ETAS > 0.003) & (ETAS < 1.2)
    ax.loglog(ETAS[m], E[m], "--", color="#2b6cb0", lw=1.1, label="excursion $pA_f/(\\eta\\Delta)$")
    ax.loglog(ETAS[m], np.maximum(St[m], 1e-7), ":", color="#dd6b20", lw=1.3, label="stationary $S_f(\\eta)$")
    ax.loglog(ETAS[m], J[m], "-", color="0.15", lw=1.3, label="total $J(\\eta)$")
    i = int(np.argmin(J)); ax.plot(ETAS[i], J[i], "o", color="0.15", ms=4)
    ax.axvline(MEAS[k][ip], color="0.5", lw=0.7, ls=(0, (1, 2)))
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"step $\eta$")
    lo = max(J.min() / 30, 1e-6); ax.set_ylim(lo, J[m].max() * 1.5)
    ax.grid(True, which="major", lw=0.3, color="0.88")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("cost per slot (model)")
axs[0].legend(loc="lower left", frameon=False, fontsize=6, handlelength=1.8)
fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig("fig4_mechanism.pdf"); fig.savefig("fig4_mechanism.png", dpi=200)
print("ok")
