# The Decision Sets the Tuning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23160427.svg)](https://doi.org/10.5281/zenodo.23160427)

Code, experiment definitions and per-run results for

> Bala Krishna Reddy D and Ajin George Joseph, **"The Decision Sets the Tuning: Step-Size Laws for
> Subspace Trackers Under Abrupt Spectrum Changes,"** manuscript, 2026.

A streaming subspace tracker's step size trades recovery speed after a change against stationary
noise. This repository shows, by simulation and a linearised theory, that the optimal step depends on
the decision the estimate feeds: subspace accuracy, argmin channel selection, null-steering
interference, or interference outage. Each decision gives a different law for how the optimal step
scales with the change rate.

## Quick start

```bash
pip install -r requirements.txt
python3 revsim.py selftest      # must print 0.16155 and 0.99927
```

Pure numpy for the simulations; scipy for the statistics; matplotlib for the figures.

## Layout

| Path | Contents |
|---|---|
| `revsim.py` | Environments (Gaussian and complex ULA arrays, swap and variable occupancy), trackers (Oja, PAST, EW covariance, beamformer), costs, and the `cell` entry point every sweep line calls |
| `sweeps/` | One shell command per line; each line is one configuration run over five seeds |
| `results/<sweep>_<NNN>/results.jsonl` | Per-run results, one JSON line per seed, as used in the paper |
| `null_analyze.py` | Optimal step per cost and rate, exponent fits with array-level bootstrap (Figs. 3–4, Tables III–V) |
| `best_cost_tests.py` | Tracker comparison: effect sizes, paired tests, leave-one-array-out tuning (Section VII) |
| `robustness.py` | Optimum-location robustness and change-rate misspecification |
| `rev_analyze.py` | Channel-selection, reset, decay and beamformer experiments (Sections VIII–IX) |
| `theory/` | Linearised theory: `predict_eta.py` (predicted optima), `stat_check.py` (stationary formulas), `argmin_drag.py` (channel-selection optimum) |
| `paper/` | Scripts that draw the paper's figures |

## Reproducing the paper

| Paper item | Command |
|---|---|
| Figs. 3 and 4, Tables III–V, exponents in the text | `python3 null_analyze.py --max-p 2e-3` |
| Table I, Fig. 2 (predicted against measured) | `python3 theory/predict_eta.py --seeds 20 --compare` |
| Stationary-noise checks (Section IV-B) | `python3 theory/stat_check.py` |
| Channel-selection optimum (Section IV-E) | `python3 theory/argmin_drag.py` |
| Predictions for SNR 4 and 16, and for k = 3 | `python3 theory/predict_eta.py --snr 4` (or `--snr 16`, `--k 3`) |
| Tracker comparison (Section VII) | `python3 best_cost_tests.py` |
| Robustness checks (Sections V and VI) | `python3 robustness.py` |
| Sections VIII–IX | `python3 rev_analyze.py` |
| Figures | `cd paper && python3 make_figs.py && python3 make_figs2.py` |

The analysis scripts read `results/` as shipped, so every table can be checked without rerunning
the simulations.

## Rerunning the simulations

Every sweep file can be regenerated with `python3 revsim.py sweeps --set {revision,null,n2,n45,n6,n7,n7b,n8}`.
Run each line as a separate job with `RUN_DIR` set to an empty folder; the line writes
`results.jsonl` there. Any job scheduler works. The full set is about 300 core-hours.

| Sweeps | What | Lines |
|---|---|---|
| `E1`–`E8b` | Channel selection with known steering vectors: baselines, resets, step sweep, excursions, decay, variable occupancy, arrays and SNR | 321 |
| `N1_tuning_law`, `N1b_edges` | Oja, M = 16, eight change rates, all costs | 528 |
| `N2_trackers`, `N3_scale` | PAST and EW covariance at M = 16; all trackers at M = 32, 64 | 984 |
| `N4_ula`, `N5_snr` | Complex uniform-linear arrays; SNR 4 and 16 | 790 |
| `N6_k3` | Three active users | 150 |

Not included: the per-slot error traces of `E5_excursions` (about 300 MB), which give the
exact-covariance decay rate and post-change peak angle in Section IV-A. Rerunning that sweep
regenerates them.

## Licence and citation

Code is released under the MIT licence (`LICENSE`); see `CITATION.cff` for how to cite.
The simulation and analysis code was developed with the assistance of Claude (Anthropic); the authors
checked every result reported in the paper.

Archived version 1.0.0: https://doi.org/10.5281/zenodo.23160427  
Repository: https://github.com/balakrishna478/decision-sets-tuning

## N7: variable-rank occupancy

`sweeps/N7_varrank.txt` (`python3 revsim.py sweeps --set n7`) tests the tuning laws when users
arrive and depart independently instead of swapping. Each of the K = 6 users switches on and off as
a two-state Markov chain with mean occupancy 2 and total change probability p per slot
(p01 = p/8, p10 = p/4). Trackers keep a fixed width kw = 4 and null-steer against all four
directions; rank exceeds 4 in about 2% of slots. Because occupancy and tracked width differ,
`sin_mean` here is the coverage error: the largest principal-angle sine from the occupied subspace
into the tracked one. E7 (no `--p`) keeps its original fixed rates and reproduces exactly.

Analyse with `python3 null_analyze.py --max-p 2e-3`; N7 results appear as groups ending in `:var4`.

Under variable rank every decision cost is scored on slots with at most four active users, where
the tracker can null all active users; in the others collisions and interference are partly
unavoidable whatever the step. Accuracy and selection come from N7, interference and outage from
N7b (`sweeps/N7b_varrank_rep.txt`, which replays N7 exactly and records those costs per slot type):

    python3 n7_selection.py results/ n7_rep/ N7   && python3 null_analyze.py n7_rep  --max-p 2e-3
    python3 n7_selection.py results/ n7b_rep/ N7b && python3 null_analyze.py n7b_rep --max-p 2e-3

## N8: online tuning

`sweeps/N8_online.txt` (`python3 revsim.py sweeps --set n8`) asks whether a radio that does not know
the change rate can use the fitted laws as a tuning rule. Oja's step follows the N1 law of the
target cost, eta = law_eta (p_hat / 2e-4)^law_a, with p_hat updated every slot from the last ten
detected changes. Three rate estimators: `known` (the true rate), `genie` (true change instants)
and `blind` (no steering vectors, occupancy or rate: a fast auxiliary Oja tracker with step 0.1
and a one-sided CUSUM on its standardized residual energy, threshold 12, hold-off 150 slots). The
arrays (seeds 20-29) were not used to fit the laws. Rules are compared with the constant-step grid
on the same arrays at fixed rates 1e-4, 4e-4 and 1.6e-3, and with a rate that switches between
1e-4 and 1.6e-3 every 300,000 slots (the best constant step in hindsight, and the best constant
step for each regime separately).

    python3 n8_analyze.py results/
