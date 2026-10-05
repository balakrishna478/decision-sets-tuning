"""Figures for the manuscript. Numbers are the bootstrap/optimum estimates from
null_analyze.py (N1, N1b, N2; M = 16; p = 3.2e-3 excluded)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

# ---------------------------------------------------------------- Fig. 1
P = np.array([2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3])
S = {"Accuracy": ([.01016, .015, .024, .036, .058, .085, .126], 0.61),
     "Mean interference": ([.020, .028, .038, .054, .078, .107, .148], 0.48),
     r"Outage, $\Gamma=0.01$": ([.036, .041, .045, .052, .062, .072, .084], 0.21),
     r"Outage, $\Gamma=0.03$": ([.083, .089, .098, .115, .126, .144, .166], 0.17),
     r"Outage, $\Gamma=0.1$": ([.193, .226, .234, .244, .265, .308, .336], 0.12),
     "Argmin selection": ([.919, .934, .970, 1.021, 1.094, 1.295, 1.302], 0.09)}
# p = 3.2e-3 (excluded from fits: excursions overlap), shown as open markers
EXCL = [0.180, 0.181, 0.107, 0.185, 0.367, 1.367]
fig, ax = plt.subplots(figsize=(3.45, 2.75))
blues = plt.cm.Blues(np.linspace(0.45, 0.95, 5))
handles = []
for j, (name, (v, a)) in enumerate(S.items()):
    if j == 0:
        h, = ax.plot(P, v, "o--", color="0.45", ms=2.8, lw=1.1, label=f"{name} ($a={a:.2f}$)")
    else:
        h, = ax.plot(P, v, "o-", color=blues[j - 1], ms=2.8, lw=1.1, label=f"{name} ($a={a:.2f}$)")
    handles.append(h)
    ax.plot([3.2e-3], [EXCL[j]], "o", mfc="none", mec=(h.get_color()), ms=3.4, mew=0.8)
ax.legend(handles[::-1], [h.get_label() for h in handles[::-1]], loc="lower right", frameon=False,
          fontsize=6.3, handlelength=1.8, labelspacing=0.25, borderaxespad=0.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1.8e-5, 4.4e-3); ax.set_ylim(4e-3, 2.2)
ax.set_xticks([2.5e-5, 1e-4, 4e-4, 1.6e-3]); ax.set_xticklabels([r"$2.5{\times}10^{-5}$", r"$10^{-4}$", r"$4{\times}10^{-4}$", r"$1.6{\times}10^{-3}$"])
ax.minorticks_off()
ax.set_xlabel(r"change probability per slot, $p$")
ax.set_ylabel(r"optimal step $\eta^\star$")
ax.grid(True, which="major", lw=0.3, color="0.85")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig("fig1_eta_star.pdf"); fig.savefig("fig1_eta_star.png", dpi=200)

# ---------------------------------------------------------------- Fig. 2
costs = ["Accuracy", "Mean interference", r"Outage, $\Gamma=0.01$", r"Outage, $\Gamma=0.03$",
         r"Outage, $\Gamma=0.1$", "Argmin selection"]
E = {"Oja": [(.61, .60, .62), (.48, .45, .51), (.21, .20, .21), (.17, .16, .17), (.12, .11, .14), (.09, .08, .10)],
     "PAST": [(.62, .61, .64), (.49, .44, .55), (.21, .20, .22), (.16, .15, .17), (.13, .12, .15), (.09, .09, .10)],
     "EW covariance": [(.63, .62, .64), (.49, .43, .54), (.18, .17, .19), (.14, .13, .15), (.12, .11, .13), (.11, .10, .12)]}
cols = {"Oja": "#2b6cb0", "PAST": "#dd6b20", "EW covariance": "#2f855a"}
off = {"Oja": 0.0, "PAST": 0.22, "EW covariance": -0.22}
fig, ax = plt.subplots(figsize=(3.45, 2.6))
for tk, vals in E.items():
    for i, (a, lo, hi) in enumerate(vals):
        y = i + off[tk]
        ax.plot([lo, hi], [y, y], color=cols[tk], lw=1.2)
        ax.plot(a, y, "o", color=cols[tk], ms=3.6, label=tk if i == 0 else None)
for i, pred, lab in [(0, 2 / 3, "leading order 2/3"), (1, 0.5, "leading order 1/2")]:
    ax.plot([pred, pred], [i - 0.42, i + 0.42], ls=(0, (2, 2)), color="0.4", lw=0.8)
    ax.text(pred - 0.006, i - 0.47, lab, fontsize=5.8, color="0.35", va="bottom", ha="right")
ax.set_yticks(range(len(costs))); ax.set_yticklabels(costs)
ax.invert_yaxis()
ax.set_xlim(0, 0.72)
ax.set_xlabel(r"exponent $a$ in $\eta^\star \propto p^{a}$")
ax.grid(True, axis="x", lw=0.3, color="0.85")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig("fig2_exponents.pdf"); fig.savefig("fig2_exponents.png", dpi=200)
print("ok")
