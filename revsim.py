#!/usr/bin/env python3
"""
Revision experiments for the Objective-3 DSA paper.  numpy only.

  python3 revsim.py sweeps                              # write sweeps/E*.txt for lab.sh
  python3 revsim.py sweeps --set null                   # new work: sweeps/N1_tuning_law.txt
  python3 revsim.py sweeps --set n2                     # N2 (other trackers), N3 (M = 32, 64)
  python3 revsim.py sweeps --set n45                    # N4 (complex ULA), N5 (SNR 4 and 16)
  python revsim.py cell --exp E3 ... --seeds 0-19 --total-timesteps N   (one sweep line)
  python3 revsim.py geometry --seeds 20 --out geometry.json   # offline, ~15 s
  python3 revsim.py selftest                            # reproduces two paper numbers

Standalone mode (no lab.sh): `jobs` + `worker --shard i --nshards n`.

Lab mode: each sweep line writes $RUN_DIR/results.jsonl (one line per seed, appended as each
seed finishes, so a partial run is still usable) and, for E5, $RUN_DIR/traces/<id>.npz.
Standalone mode writes results/<host>_shard<k>.jsonl and results/traces/ instead.

Environments reproduce dsa2.py / dsa3.py exactly in the 'gauss' (legacy) mode:
same RNG stream, same seed offsets (+7 constant rank, +3 variable rank).  Note that
in legacy mode the steering vectors are drawn from the seed, so every seed is a
different array.  That is kept deliberately and reported per seed (geometry.json).
"""
import argparse, itertools, json, os, socket, sys, time
from multiprocessing import Pool
import numpy as np


# ------------------------------------------------------------------ linear algebra
def qf(A):
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    ph = d / np.where(np.abs(d) > 0, np.abs(d), 1.0)
    return Q * ph            # makes diag(R) real positive (sign for real input)


def sin_theta(A, B):
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    sv = np.linalg.svd(A.conj().T @ B, compute_uv=False)
    return float(np.sqrt(np.clip(1 - sv.min() ** 2, 0, None))) if A.shape[1] == B.shape[1] \
        else float("nan")


def sin_cover(U, B):
    """Largest principal-angle sine from the occupied subspace U (r columns) into span(B)
    (kw >= r columns): 0 when B contains U. Used as the accuracy cost under variable rank."""
    r = U.shape[1]
    if r == 0 or r > B.shape[1]:
        return float("nan")
    sv = np.linalg.svd(U.conj().T @ B, compute_uv=False)
    return float(np.sqrt(np.clip(1 - sv.min() ** 2, 0, None)))


def orth(S):
    if S.shape[1] == 0:
        return np.zeros((S.shape[0], 0), dtype=S.dtype)
    Q, _ = np.linalg.qr(S)
    return Q


# ------------------------------------------------------------------ environments
def make_steering(r, M, K, array):
    if array == "gauss":
        S = r.standard_normal((M, K))
        return S / np.linalg.norm(S, axis=0)
    if array == "ula":                       # half-wavelength ULA, DOAs in +-60 deg
        doa = r.uniform(-np.pi / 3, np.pi / 3, K)
        m = np.arange(M)[:, None]
        return np.exp(1j * np.pi * m * np.sin(doa)[None, :]) / np.sqrt(M)
    raise ValueError(array)


class Env:
    """Constant rank (swap) or variable rank (independent on/off) occupancy."""

    def __init__(self, kind, M, K, k, p, snr, seed, array="gauss", p01=1.2e-4, p10=1.6e-4,
                 p_hi=None, dwell=None, burn=0):
        self.kind, self.M, self.K, self.k, self.p, self.snr = kind, M, K, k, p, snr
        # N8 rate switching (swap occupancy only): rate p during burn-in, then regimes of `dwell`
        # slots alternating p, p_hi, p, p_hi, ...  Without p_hi the environment is unchanged.
        self.p_lo, self.p_hi, self.dwell, self.burn0, self.t = p, p_hi, dwell, burn, 0
        self.regime = 0
        self.array, self.p01, self.p10 = array, p01, p10
        self.r = np.random.default_rng(seed)
        self.S = make_steering(self.r, M, K, array)
        self.cplx = np.iscomplexobj(self.S)
        if kind == "swap":
            self.act = list(range(k))
        else:
            self.o = np.array(([1, 1] + [0] * K)[:K])
            self.act = list(np.flatnonzero(self.o))
        self._U = {}

    def _gauss(self, n):
        if self.cplx:
            return (self.r.standard_normal(n) + 1j * self.r.standard_normal(n)) / np.sqrt(2)
        return self.r.standard_normal(n)

    def subspace(self):
        key = tuple(self.act)
        if key not in self._U:
            self._U[key] = orth(self.S[:, list(key)])
        return self._U[key]

    def step(self):
        r, changed = self.r, False
        if self.kind == "swap":
            if self.p_hi is not None:
                self.t += 1
                self.regime = 0 if self.t <= self.burn0 else ((self.t - self.burn0 - 1) // self.dwell) % 2
                self.p = self.p_hi if self.regime else self.p_lo
            if r.random() < self.p:
                idle = [c for c in range(self.K) if c not in self.act]
                out = r.integers(len(self.act)); inn = idle[r.integers(len(idle))]
                self.act = sorted(self.act[:out] + self.act[out + 1:] + [inn]); changed = True
            y = self.S[:, self.act] @ (np.sqrt(self.snr) * self._gauss(self.k)) + self._gauss(self.M)
        else:
            f01 = (r.random(self.K) < self.p01) & (self.o == 0)
            f10 = (r.random(self.K) < self.p10) & (self.o == 1)
            changed = bool(f01.any() or f10.any())
            self.o = self.o ^ f01 ^ f10
            self.act = list(np.flatnonzero(self.o))
            a = self.act
            y = self.S[:, a] @ (np.sqrt(self.snr) * self._gauss(len(a))) if a else \
                np.zeros(self.M, dtype=self.S.dtype)
            y = y + self._gauss(self.M)
        return y, changed

    def cbar(self):
        Sa = self.S[:, self.act]
        return self.snr * Sa @ Sa.conj().T + np.eye(self.M)


# ------------------------------------------------------------------ change-rate estimation (N8)
class RateEstimator:
    """Online estimate of the change probability per slot.
    mode 'known' : the true current rate (a genie that knows p).
    mode 'genie' : true change instants, N-interval window (estimation noise only).
    mode 'blind' : no knowledge of steering vectors, occupancy or p: a fast auxiliary Oja tracker
                   (step eta_d) and a one-sided CUSUM on its standardized residual energy
                   ||y||^2 - ||W^H y||^2; one alarm per detection, then a hold-off.
    With fewer than N events the estimate shrinks toward the prior p0 (pseudo-count n0)."""

    def __init__(self, mode, M, k, snr, cplx, seed, N=10, p0=2e-4, n0=2.0,
                 eta_d=0.1, h=12.0, kc=0.5, hold=150, warm=3000):
        self.mode, self.N, self.p0, self.n0 = mode, N, p0, n0
        self.times, self.t = [], 0
        if mode == "blind":
            self.aux = Oja(M, k, snr * k + M, np.random.default_rng(seed + 555), cplx, eta=eta_d)
            self.mu, self.var, self.S, self.last = float(M - k), 2.0 * (M - k), 0.0, -10 ** 9
            self.h, self.kc, self.hold, self.warm = h, kc, hold, warm

    def update(self, y, changed, env):
        self.t += 1
        if self.mode == "known":
            return env.p
        if self.mode == "genie":
            if changed:
                self.times.append(self.t)
        else:
            self.aux.update(y, changed, env)
            W = self.aux.W
            e = float(np.real(y.conj() @ y) - np.sum(np.abs(W.conj().T @ y) ** 2))
            if self.t < self.warm:
                self.mu += 0.01 * (e - self.mu); self.var += 0.01 * ((e - self.mu) ** 2 - self.var)
            elif self.t - self.last >= self.hold:
                self.S = max(0.0, self.S + (e - self.mu) / np.sqrt(self.var) - self.kc)
                if self.S > self.h:
                    self.times.append(self.t); self.S = 0.0; self.last = self.t
                else:
                    self.mu += 1e-3 * (e - self.mu); self.var += 1e-3 * ((e - self.mu) ** 2 - self.var)
        n = len(self.times)
        if n >= self.N:      # (N-1)/(time spanned by the last N events): unbiased for Poisson events
            return (self.N - 1) / max(self.t - self.times[-self.N], 1)
        return (n + self.n0) / (self.t + self.n0 / self.p0)


# ------------------------------------------------------------------ trackers
class Oja:
    """Constant / decaying / reset-on-detection step.  Rank-one form of eq. (2)."""

    def __init__(self, M, kw, kappa, rng, cplx, eta=0.05, sched="const", eta0=2.0,
                 floor=0.0, delay=0, fa=0.0, det_seed=0, driver="sample"):
        W = rng.standard_normal((M, kw))
        self.W = qf(W.astype(complex) if cplx else W)
        self.kappa, self.eta, self.sched, self.eta0 = kappa, eta, sched, eta0
        self.floor, self.delay, self.fa, self.driver = floor, delay, fa, driver
        self.det = np.random.default_rng(det_seed)
        self.age, self.t, self.pending = 1, 0, []

    def step_size(self):
        if self.sched == "const":
            return self.eta
        if self.sched == "decay":
            return self.eta0 / self.t
        return max(self.eta0 / self.age, self.floor)          # reset

    def update(self, y, changed, env):
        self.t += 1
        if self.sched == "reset":
            if changed:
                self.pending.append(self.t + self.delay)
            alarm = any(tt <= self.t for tt in self.pending)
            self.pending = [tt for tt in self.pending if tt > self.t]
            if self.fa > 0 and self.det.random() < self.fa:
                alarm = True
            if alarm:
                self.age = 1
        eta = self.step_size() / self.kappa
        W = self.W
        if self.driver == "exact":
            CW = env.cbar() @ W
            G = CW - W @ (W.conj().T @ CW)
        else:
            z = W.conj().T @ y
            G = np.outer(y - W @ z, z.conj())
        self.W = qf(W + eta * G)
        self.age += 1

    def basis(self):
        return self.W


class PAST:
    """Yang (1995) projection approximation subspace tracking, forgetting factor beta."""

    def __init__(self, M, kw, rng, cplx, beta=0.99):
        W = rng.standard_normal((M, kw))
        self.W = qf(W.astype(complex) if cplx else W)
        self.P = np.eye(kw, dtype=self.W.dtype)
        self.beta, self.n = beta, 0

    def update(self, y, changed, env):
        x = self.W.conj().T @ y
        h = self.P @ x
        g = h / (self.beta + np.real(x.conj() @ h))
        self.P = (self.P - np.outer(g, h.conj())) / self.beta
        self.n += 1
        if self.n % 100 == 0:
            self.P = 0.5 * (self.P + self.P.conj().T)
        self.W = self.W + np.outer(y - self.W @ x, g.conj())
        self.Q = qf(self.W)            # the policy needs an orthonormal basis: count it as update cost

    def basis(self):
        return self.Q


class EWCov:
    """Exponentially weighted sample covariance + top-k eigendecomposition."""

    def __init__(self, M, kw, cplx, beta=0.01):
        self.C = np.eye(M, dtype=complex if cplx else float)
        self.kw, self.beta = kw, beta

    def update(self, y, changed, env):
        self.C = (1 - self.beta) * self.C + self.beta * np.outer(y, y.conj())
        _, V = np.linalg.eigh(self.C)  # the estimate is the eigenbasis: count it as update cost
        self.V = V[:, ::-1][:, :self.kw]

    def basis(self):
        return self.V


class Beamformer:
    """Per-channel EWMA of beamformer output power |s_c^H y|^2.  No subspace."""

    def __init__(self, S, beta=0.1):
        self.SH, self.beta = S.conj().T, beta
        self.e = np.full(S.shape[1], 2.0)

    def update(self, y, changed, env):
        self.e = (1 - self.beta) * self.e + self.beta * np.abs(self.SH @ y) ** 2

    def scores(self):
        return self.e


# ------------------------------------------------------------------ geometry
def active_sets(K, k):
    return list(itertools.combinations(range(K), k))


def margin_stats(S, k):
    out = []
    K = S.shape[1]
    for a in active_sets(K, k):
        U = orth(S[:, list(a)])
        idle = [c for c in range(K) if c not in a]
        rho = min(float(np.sum(np.abs(U.conj().T @ S[:, c]) ** 2)) for c in idle)
        out.append((1 - np.sqrt(rho)) / 2)
    return float(np.min(out)), float(np.mean(out))


# ------------------------------------------------------------------ one run
def run(job, outdir=None, progress=None):
    j = dict(job)
    kind = j.get("kind", "swap")
    M, K, k, snr = j.get("M", 16), j.get("K", 6), j.get("k", 2), j.get("snr", 8.0)
    p, T, seed, array = j.get("p", 2e-4), j["T"], j["seed"], j.get("array", "gauss")
    method = j["method"]
    off = 7 if kind == "swap" else 3
    vr = {}
    if kind == "var" and "p" in j:
        # N7: independent on/off users with mean occupancy k of K and total change probability p
        # per slot. Detailed balance: p01 = c*pi1, p10 = c*pi0, with K * 2c*pi0*pi1 = p.
        # (E7 has no "p" key and keeps the original fixed rates.)
        pi1 = k / K; pi0 = 1 - pi1; c = p / (2 * K * pi0 * pi1)
        vr = dict(p01=c * pi1, p10=c * pi0)
    if j.get("p_hi") is not None:
        vr = dict(p_hi=j["p_hi"], dwell=j["dwell"], burn=j.get("burn", T // 4))
    env = Env(kind, M, K, k, p, snr, seed + off, array, **vr)
    cplx = env.cplx
    kw = j.get("kw", k)
    kappa = snr * kw + M
    rng = np.random.default_rng(seed)
    burn = j.get("burn", T // 4 if kind == "swap" else min(5000, T // 4))
    sin_every = j.get("sin_every", 1 if kind == "swap" else 0)
    gmin, gmean = margin_stats(env.S, k) if kind == "swap" else (float("nan"),) * 2

    rankmode = j.get("rankmode") if method != "bf" else None   # None | 'oracle' | 'est'
    if method == "oja":
        trk = Oja(M, kw, kappa, rng, cplx, eta=j.get("eta", 0.05), sched=j.get("sched", "const"),
                  eta0=j.get("eta0", 2.0), floor=j.get("floor", 0.0), delay=j.get("delay", 0),
                  fa=j.get("fa", 0.0), det_seed=seed + 1_000_003, driver=j.get("driver", "sample"))
    elif method == "past":
        trk = PAST(M, kw, rng, cplx, beta=j["beta"])
    elif method == "ewcov":
        trk = EWCov(M, kw, cplx, beta=j["beta"])
    elif method == "bf":
        trk = Beamformer(env.S, beta=j["beta"])
    else:
        raise ValueError(method)
    # N8: step from a tuning law, eta = law_eta * (p_hat / law_pref)^law_a, p_hat estimated online
    rate = None
    if j.get("rate"):
        rate = RateEstimator(j["rate"], M, k, snr, cplx, seed, N=j.get("rate_n", 10))
        law_eta, law_a, law_pref = j["law_eta"], j["law_a"], j.get("law_pref", 2e-4)
        log_eta_sum, log_ratio_sum, log_ratio_sq, n_alarm_eval = 0.0, 0.0, 0.0, 0
    switching = j.get("p_hi") is not None
    racc = {0: np.zeros(9), 1: np.zeros(9)}    # n, sin, I, out*3, rew, orew, coll   (per regime)

    S, SH = env.S, env.S.conj().T
    ocache = {}
    n_eval = rew = orew = coll = 0
    sins, above_g, above_c, n_jumps = [], 0, 0, 0
    byrank = {}
    colpow = np.zeros(kw) if rankmode else None
    trace = j.get("trace", False)
    null = j.get("null", False)            # blind null-steering: SU never uses steering vectors
    if null:
        hr = np.random.default_rng(seed + 999)
        h = hr.standard_normal(M) + (1j * hr.standard_normal(M) if cplx else 0)
        h = h / np.linalg.norm(h)
        GAMS = (0.01, 0.03, 0.1)   # interference after a swap is ~1/(M-k) on average, so 0.3 never trips
        nI, ngain, nout = 0.0, 0.0, np.zeros(len(GAMS))
        # variable rank: the same costs on slots with at most kw active users (all nullable)
        nI_rep, nout_rep, n_rep = 0.0, np.zeros(len(GAMS)), 0
    tr_sin, tr_col, tr_jump = [], [], []
    t0 = time.time()

    every = max(1, T // 10)
    t_upd = 0.0
    for t in range(1, T + 1):
        if progress and t % every == 0:
            progress(t)
        y, ch = env.step()
        if rate is not None:
            n_before = len(rate.times)
            p_hat = rate.update(y, ch, env)
            trk.eta = float(np.clip(law_eta * (p_hat / law_pref) ** law_a, 0.003, 4.0))
        tu = time.perf_counter()
        trk.update(y, ch, env)
        t_upd += time.perf_counter() - tu
        if t <= burn:
            continue
        n_eval += 1
        if rate is not None:
            lr = np.log(p_hat / env.p)
            log_eta_sum += np.log(trk.eta); log_ratio_sum += lr; log_ratio_sq += lr * lr
            n_alarm_eval += len(rate.times) - n_before
        n_jumps += int(ch)
        act = env.act
        if method == "bf":
            sc = trk.scores()
            B = None
        else:
            B = trk.basis()
            Bfull = B
            if rankmode:
                colpow = 0.99 * colpow + 0.01 * np.abs(B.conj().T @ y) ** 2
                order = np.argsort(colpow)[::-1]
                r_use = len(act) if rankmode == "oracle" else \
                    int(np.sum(colpow > j.get("rank_thr", 1.0 + snr / 4)))
                B = B[:, order[:r_use]]
            sc = np.sum(np.abs(B.conj().T @ S) ** 2, axis=0) if B.shape[1] else np.zeros(K)
        It = None
        if null and B is not None:
            w = h - B @ (B.conj().T @ h)
            w = w / np.linalg.norm(w)
            It = float(np.sum(np.abs(SH[act] @ w) ** 2)) if act else 0.0
            nI += It; nout += np.array([It > g for g in GAMS]); ngain += float(abs(h.conj() @ w) ** 2)
            if kind == "var" and len(act) <= kw:
                nI_rep += It; nout_rep += np.array([It > g for g in GAMS]); n_rep += 1
        c = int(np.argmin(sc))
        hit = c in act
        key = tuple(act)
        if key not in ocache:
            U = env.subspace()
            osc = np.sum(np.abs(U.conj().T @ S) ** 2, axis=0) if U.shape[1] else np.zeros(K)
            ocache[key] = int(np.argmin(osc)) in act
        ohit = ocache[key]
        rew += -2.0 if hit else 1.0
        orew += -2.0 if ohit else 1.0
        coll += int(hit)
        if kind == "var":
            d = byrank.setdefault(len(act), [0, 0]); d[0] += 1; d[1] += int(hit)
        if switching:
            a_ = racc[env.regime]; a_[0] += 1; a_[6] += -2.0 if hit else 1.0
            a_[7] += -2.0 if ohit else 1.0; a_[8] += int(hit)
            if It is not None:
                a_[2] += It; a_[3:6] += [It > g for g in GAMS]
        if kind == "var" and sin_every and B is not None and (n_eval % sin_every == 0):
            s = sin_cover(env.subspace(), Bfull)
            if np.isfinite(s):
                sins.append(s); above_c += int(s > 0.95)
        elif sin_every and B is not None and (n_eval % sin_every == 0):
            s = sin_theta(B, env.subspace())
            if switching:
                racc[env.regime][1] += s
            sins.append(s)
            above_g += int(s > gmin); above_c += int(s > 0.95)
            if trace:
                tr_sin.append(s); tr_col.append(hit)
                if ch:
                    tr_jump.append(len(tr_sin) - 1)

    res = dict(job)
    res.update(reward=rew / n_eval, oracle_reward=orew / n_eval, gap=(orew - rew) / n_eval,
               coll_rate=coll / n_eval, n_eval=n_eval, n_jumps_eval=n_jumps,
               gamma_min=gmin, gamma_mean=gmean, secs=round(time.time() - t0, 1),
               update_us=round(t_upd / T * 1e6, 2))
    if sins:
        a = np.array(sins)
        res.update(sin_mean=float(a.mean()), sin_med=float(np.median(a)),
                   sin_q90=float(np.quantile(a, 0.9)),
                   frac_above_gamma=above_g / len(a), frac_above_095=above_c / len(a))
    if null:
        res.update(I_mean=nI / n_eval, gain_mean=ngain / n_eval,
                   **{f"out_{g}": float(o / n_eval) for g, o in zip(GAMS, nout)})
        if kind == "var" and n_rep:
            res.update(I_mean_rep=nI_rep / n_rep, n_rep=n_rep,
                       **{f"out_{g}_rep": float(o / n_rep) for g, o in zip(GAMS, nout_rep)})
    if rate is not None:
        res.update(eta_geo=float(np.exp(log_eta_sum / n_eval)),
                   phat_logbias=log_ratio_sum / n_eval,
                   phat_logsd=float(np.sqrt(max(log_ratio_sq / n_eval - (log_ratio_sum / n_eval) ** 2, 0))),
                   alarms_eval=n_alarm_eval)
    if switching:
        for g, a_ in racc.items():
            nn = max(a_[0], 1)
            res[f"reg{g}"] = dict(n=int(a_[0]), sin_mean=a_[1] / nn, I_mean=a_[2] / nn,
                                  **{f"out_{x}": a_[3 + i] / nn for i, x in enumerate(GAMS)},
                                  gap=(a_[7] - a_[6]) / nn, coll_rate=a_[8] / nn)
    if byrank:
        res["coll_by_rank"] = {str(rk): v[1] / v[0] for rk, v in sorted(byrank.items())}
        res["slots_by_rank"] = {str(rk): v[0] for rk, v in sorted(byrank.items())}
    if trace:
        root = outdir or "results"
        os.makedirs(os.path.join(root, "traces"), exist_ok=True)
        res["run_dir"] = os.path.basename(os.path.normpath(outdir)) if outdir else ""
        res["trace_file"] = f"traces/{job['id']}.npz"
        np.savez_compressed(os.path.join(root, res["trace_file"]),
                            sin=np.array(tr_sin, dtype=np.float32),
                            coll=np.array(tr_col, dtype=np.int8),
                            jumps=np.array(tr_jump, dtype=np.int32))
    return res


# ------------------------------------------------------------------ job list
T_OF = {5e-5: 1_200_000, 2e-4: 480_000, 8e-4: 120_000}
RATES = [5e-5, 2e-4, 8e-4]
ETAS = [0.006, 0.012, 0.024, 0.048, 0.096, 0.192, 0.384, 0.768, 1.536]


def build_jobs(nseeds=20, scale=1.0):
    J = []
    Tof = {p: int(T * scale) for p, T in T_OF.items()}
    T120, T480 = int(120_000 * scale), int(480_000 * scale)
    add = lambda exp, **kw: J.append(dict(exp=exp, **kw))
    S = range(nseeds)
    # E1 baselines (Oja itself comes from E3)
    for p in RATES:
        for s in S:
            for b in [0.01, 0.03, 0.1, 0.3]:
                add("E1", method="bf", beta=b, p=p, T=Tof[p], seed=s, sin_every=0)
            for b in [0.9, 0.97, 0.99, 0.997]:
                add("E1", method="past", beta=b, p=p, T=Tof[p], seed=s)
            for b in [0.003, 0.01, 0.03, 0.1]:
                add("E1", method="ewcov", beta=b, p=p, T=Tof[p], seed=s)
    # E2 reset family
    for p in RATES:
        for s in S:
            for e0 in [0.5, 1.0, 2.0, 4.0]:
                add("E2", method="oja", sched="reset", eta0=e0, p=p, T=Tof[p], seed=s)
            for fl in [0.02, 0.048, 0.096, 0.192]:
                add("E2", method="oja", sched="reset", eta0=2.0, floor=fl, p=p, T=Tof[p], seed=s)
            for d in [0, 25, 100, 400]:
                for fa in [0.0, 1e-4, 1e-3]:
                    if d == 0 and fa == 0.0:
                        continue
                    add("E2", method="oja", sched="reset", eta0=2.0, floor=0.048, delay=d, fa=fa,
                        p=p, T=Tof[p], seed=s)
    # E3 extended constant-step sweep
    for p in RATES:
        for s in S:
            for e in ETAS:
                add("E3", method="oja", sched="const", eta=e, p=p, T=Tof[p], seed=s)
    # E5 excursion traces (sampled and exact covariance)
    for p in [2e-4, 8e-4]:
        for s in range(min(10, nseeds)):
            for e in [0.024, 0.048, 0.096, 0.192, 0.384]:
                for drv in ["sample", "exact"]:
                    add("E5", method="oja", sched="const", eta=e, driver=drv, p=p, T=Tof[p],
                        seed=s, trace=True)
    # E6 decaying step, Table I horizon
    for s in range(min(14, nseeds)):
        for e0 in [1.0, 2.0, 5.0, 10.0, 20.0]:
            add("E6", method="oja", sched="decay", eta0=e0, p=2e-4, T=T120, seed=s, sin_every=10)
        add("E6", method="oja", sched="const", eta=0.02, p=2e-4, T=T120, seed=s, sin_every=10)
    # E7 variable occupancy
    for s in S:
        for kw in [2, 3, 4, 5]:
            add("E7", kind="var", method="oja", kw=kw, eta=0.096, T=T480, seed=s)
        for rm in ["oracle", "est"]:
            add("E7", kind="var", method="oja", kw=5, eta=0.096, rankmode=rm, T=T480, seed=s)
        add("E7", kind="var", method="bf", beta=0.1, T=T480, seed=s)
    # E8 realism: array x SNR x M
    for arr in ["gauss", "ula"]:
        for snr in [2.0, 4.0, 8.0, 16.0]:
            for M in [8, 16, 32]:
                for s in range(min(10, nseeds)):
                    for e in [0.048, 0.192, 0.384]:
                        add("E8", method="oja", eta=e, array=arr, snr=snr, M=M, p=2e-4,
                            T=T480, seed=s)
                    for b in [0.03, 0.1]:
                        add("E8", method="bf", beta=b, array=arr, snr=snr, M=M, p=2e-4,
                            T=T480, seed=s, sin_every=0)
    for i, x in enumerate(J):
        x["id"] = f"{x['exp']}_{i:05d}"
        cost = x["T"] * (0.4 if x["method"] == "bf" else 1.0) * (x.get("M", 16) / 16) ** 1.5
        x["cost"] = cost * (1.5 if x["method"] == "ewcov" else 1.0)
    return J


# ------------------------------------------------------------------ new work: tuning laws
N1_RATES = [2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3]
N1_ETAS = [round(0.012 * 2 ** (i / 2), 4) for i in range(15)]          # 0.012 .. 1.536, x sqrt2


def build_null_jobs(nseeds=20, scale=1.0):
    """N1: one Oja run records every cost at once -- mean sinTheta (tracking), argmin
    collisions (discrete decision), mean null-steering interference (smooth decision) and
    interference outage at three thresholds -- so the tuning laws are compared on identical
    trajectories.  ~80 jumps in each evaluation window."""
    J = []
    for p in N1_RATES:
        T = int(round(80 / p / 0.75 * scale))
        for e in N1_ETAS:
            for s in range(nseeds):
                J.append(dict(exp="N1", method="oja", sched="const", eta=e, p=p, T=T,
                              seed=s, null=True))
    for i, x in enumerate(J):
        x["id"] = f"N1_{i:05d}"; x["cost"] = x["T"] * 1.6
    return J


# N2: same costs, other trackers.  The forgetting rate mu plays eta's role:
#     EW covariance weight beta = mu;  PAST forgetting factor beta = 1 - mu.
N2_RATES = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
N2_MUS = [round(0.001 * 2 ** (i / 2), 6) for i in range(-2, 19)]        # 0.0005 .. 0.512, x sqrt2
# N3: large arrays, all three trackers, factor-2 grids, four rates
N3_RATES = [2e-4, 4e-4, 8e-4, 1.6e-3]
N3_ETAS = [round(0.006 * 2 ** i, 4) for i in range(10)]                 # 0.006 .. 3.072
N3_MUS = [round(0.001 * 2 ** i, 6) for i in range(10)]                  # 0.001 .. 0.512


def _mu_job(method, mu):
    return dict(method=method, beta=round(1 - mu, 6) if method == "past" else mu)


def build_n2_jobs(nseeds=10, scale=1.0):
    J = []
    T = lambda p: int(round(80 / p / 0.75 * scale))
    for p in N2_RATES:
        for m in ["ewcov", "past"]:
            for mu in N2_MUS:
                for s in range(nseeds):
                    J.append(dict(exp="N2", p=p, T=T(p), seed=s, null=True, **_mu_job(m, mu)))
    for M in [32, 64]:
        for p in N3_RATES:
            for e in N3_ETAS:
                for s in range(nseeds):
                    J.append(dict(exp="N3", method="oja", sched="const", eta=e, M=M, p=p, T=T(p),
                                  seed=s, null=True))
            for m in ["ewcov", "past"]:
                for mu in N3_MUS:
                    for s in range(nseeds):
                        J.append(dict(exp="N3", M=M, p=p, T=T(p), seed=s, null=True, **_mu_job(m, mu)))
    for i, x in enumerate(J):
        x["id"] = f"{x['exp']}_{i:05d}"; x["cost"] = x["T"]
    return J


# N4: complex ULA arrays for the null-steering costs; N5: SNR sweep.  Five rates, 10 seeds.
N45_RATES = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
N4_MUS = [round(0.001 * 2 ** (i / 2), 6) for i in range(17)]            # 0.001 .. 0.256


def build_n45_jobs(nseeds=10, scale=1.0):
    J = []
    T = lambda p: int(round(80 / p / 0.75 * scale))
    for p in N45_RATES:
        for s in range(nseeds):
            for e in N1_ETAS:
                J.append(dict(exp="N4", method="oja", sched="const", eta=e, array="ula", p=p, T=T(p),
                              seed=s, null=True))
                for snr in (4.0, 16.0):
                    J.append(dict(exp="N5", method="oja", sched="const", eta=e, snr=snr, p=p, T=T(p),
                                  seed=s, null=True))
            for m in ("ewcov", "past"):
                for mu in N4_MUS:
                    J.append(dict(exp="N4", array="ula", p=p, T=T(p), seed=s, null=True, **_mu_job(m, mu)))
    for i, x in enumerate(J):
        x["id"] = f"{x['exp']}_{i:05d}"; x["cost"] = x["T"]
    return J


# N6: three active users (k = 3), Oja, M = 16, SNR = 8: the largest modelling limitation in review.
def build_n6_jobs(nseeds=10, scale=1.0):
    J = []
    T = lambda p: int(round(80 / p / 0.75 * scale))
    for p in N45_RATES:
        for s in range(nseeds):
            for e in N1_ETAS:
                J.append(dict(exp="N6", method="oja", sched="const", eta=e, k=3, p=p, T=T(p), seed=s, null=True))
    for i, x in enumerate(J):
        x["id"] = f"N6_{i:05d}"; x["cost"] = x["T"]
    return J


# N7: variable-rank occupancy (independent on/off users, mean 2 of 6 active, total change
# probability p per slot), null-steering against a fixed width kw = 4 (rank exceeds 4 in ~2% of
# slots). Accuracy is the coverage error sin_cover. Oja, PAST and EW covariance, five rates.
N7_ETAS = [0.006, 0.0085] + N1_ETAS + [2.1722, 3.072]                    # 0.006 .. 3.072, x sqrt2
N7_MUS = [0.00025, 0.000354] + N2_MUS                                    # 0.00025 .. 0.512, x sqrt2


def build_n7_jobs(nseeds=10, scale=1.0, exp="N7"):
    """exp='N7b' reruns the same grid and seeds (identical trajectories) with accuracy off,
    to record the null-steering costs on slots with at most kw active users."""
    J = []
    T = lambda p: int(round(80 / p / 0.75 * scale))
    base = dict(exp=exp, kind="var", kw=4, null=True, sin_every=1 if exp == "N7" else 0)
    for p in N45_RATES:
        for s in range(nseeds):
            for e in N7_ETAS:
                J.append(dict(base, method="oja", sched="const", eta=e, p=p, T=T(p), burn=T(p) // 4,
                              seed=s))
            for m in ("ewcov", "past"):
                # PAST diverges for mu > 0.256 here: with kw = 4 and ~2 active users, its inverse
                # correlation matrix blows up on the noise-only columns (pilot, p = 1.6e-3).
                for mu in (N7_MUS if m == "ewcov" else [x for x in N7_MUS if x <= 0.256]):
                    J.append(dict(base, p=p, T=T(p), burn=T(p) // 4, seed=s, **_mu_job(m, mu)))
    for i, x in enumerate(J):
        x["id"] = f"{exp}_{i:05d}"; x["cost"] = x["T"] * (1.5 if x["method"] == "ewcov" else 1.0)
    return J


# N8: online tuning. The step follows the fitted N1 law of the target cost,
#     eta = law_eta * (p_hat / 2e-4)^law_a, with p_hat from RateEstimator ('known', 'genie', 'blind').
#     Fresh arrays (seeds 20-29, not used to fit the laws). Part A: fixed rates, against the
#     constant-step grid on the same arrays. Part B: the rate switches 1e-4 <-> 1.6e-3 every
#     `dwell` slots, against the best constant step in hindsight.
N8_LAWS = {"sin_mean": (0.03640, 0.6104), "I_mean": (0.05462, 0.4811), "out_0.01": (0.05367, 0.2073),
           "out_0.03": (0.11414, 0.1677), "out_0.1": (0.25398, 0.1241), "gap": (1.06633, 0.0937)}
N8_RATES = [1e-4, 4e-4, 1.6e-3]


def build_n8_jobs(nseeds=10, scale=1.0, seed0=20, dwell=300_000, rates=None, targets=None,
                  modes=("known", "genie", "blind"), etas=None):
    J = []
    seeds = range(seed0, seed0 + nseeds)
    etas = etas or N7_ETAS                              # 0.006 .. 3.072, x sqrt2
    targets = targets or list(N8_LAWS)
    D = int(dwell * scale)
    settings = [dict(p=p, T=int(round(80 / p / 0.75 * scale)))
                for p in (N8_RATES if rates is None else rates)]
    settings.append(dict(p=1e-4, p_hi=1.6e-3, dwell=D, burn=D // 2, T=D // 2 + 4 * D))
    for st in settings:
        for s in seeds:
            for e in etas:
                J.append(dict(exp="N8", method="oja", sched="const", eta=e, seed=s, null=True, **st))
            for tg in targets:
                le, la = N8_LAWS[tg]
                for md in modes:
                    J.append(dict(exp="N8", method="oja", sched="const", eta=le, rate=md, target=tg,
                                  law_eta=le, law_a=la, seed=s, null=True, **st))
    for i, x in enumerate(J):
        x["id"] = f"N8_{i:05d}"; x["cost"] = x["T"] * (1.3 if x.get("rate") == "blind" else 1.0)
    return J


# ------------------------------------------------------------------ geometry dump
def swap_geometry(S, k, snr, ngrid=400):
    """Per swap triple: initial angle theta0 and the collision indicator along the
    rotation from the retired user's direction to the arriving one (paper Sec. VII)."""
    M, K = S.shape
    kappa = snr * k + M
    triples = []
    for a in active_sets(K, k):
        for out in a:
            keep = [c for c in a if c != out]
            for inn in [c for c in range(K) if c not in a]:
                new = sorted(keep + [inn])
                Qk = orth(S[:, keep])
                Pc = np.eye(M) - Qk @ Qk.conj().T
                u0 = Pc @ S[:, out]; u0 /= np.linalg.norm(u0)
                b = Pc @ S[:, inn]; b /= np.linalg.norm(b)
                ip = u0.conj() @ b
                b = b * np.conj(ip) / max(abs(ip), 1e-15)        # phase-align
                cos0 = float(np.real(u0.conj() @ b))
                th0 = float(np.arccos(np.clip(cos0, -1, 1)))
                v = b - cos0 * u0; v /= np.linalg.norm(v)
                ths = np.linspace(0, th0, ngrid)                 # remaining angle to truth
                ind = np.zeros(ngrid, dtype=bool)
                for i, th in enumerate(ths):
                    phi = th0 - th
                    u = np.cos(phi) * u0 + np.sin(phi) * v
                    W = np.column_stack([Qk, u]) if Qk.shape[1] else u[:, None]
                    sc = np.sum(np.abs(W.conj().T @ S) ** 2, axis=0)
                    ind[i] = int(np.argmin(sc)) in new
                lam = np.sort(np.linalg.eigvalsh(snr * S[:, new] @ S[:, new].conj().T
                                                 + np.eye(M)))[::-1]
                triples.append(dict(active=list(a), out=out, inn=inn, theta0=th0,
                                    ind=ind.astype(int).tolist(),
                                    delta_k=float((lam[k - 1] - lam[k]) / kappa),
                                    delta_1=float((lam[0] - lam[k]) / kappa)))
    return triples


def geometry(nseeds, out, M=16, K=6, k=2, snr=8.0, array="gauss"):
    G = {}
    for s in range(nseeds):
        env = Env("swap", M, K, k, 2e-4, snr, s + 7, array)
        gmin, gmean = margin_stats(env.S, k)
        tri = swap_geometry(env.S, k, snr)
        G[s] = dict(gamma_min=gmin, gamma_mean=gmean,
                    delta_k_mean=float(np.mean([t["delta_k"] for t in tri])),
                    triples=tri)
        print(f"seed {s}: gamma_min {gmin:.3f}  gamma_mean {gmean:.3f}  "
              f"theta0 mean {np.degrees(np.mean([t['theta0'] for t in tri])):.1f} deg", flush=True)
    json.dump(G, open(out, "w"))



# ------------------------------------------------------------------ DSL-lab (lab.sh) mode
# One sweep line = one configuration, all its seeds run in sequence inside one $RUN_DIR.
# --total-timesteps is the summed horizon and "step N" lines report cumulative progress, which
# is exactly what lab.sh's migration logic reads to compare runs asking for the same work.
FLAGS = dict(exp=str, kind=str, method=str, sched=str, driver=str, array=str, rankmode=str,
             rate=str, target=str, law_eta=float, law_a=float, p_hi=float, dwell=int, rate_n=int,
             eta=float, eta0=float, floor=float, fa=float, beta=float, p=float, snr=float,
             rank_thr=float, delay=int, M=int, K=int, k=int, kw=int, sin_every=int, burn=int)


def parse_seeds(txt):
    out = []
    for part in txt.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def cell(a):
    base = {k: getattr(a, k) for k in FLAGS if getattr(a, k) is not None}
    seeds = parse_seeds(a.seeds)
    T = a.total_timesteps // len(seeds)
    out = os.environ.get("RUN_DIR", ".")
    runid = os.path.basename(os.path.normpath(out)) or "local"
    fn = os.path.join(out, "results.jsonl")
    have = set()
    if os.path.exists(fn):                       # rerun in the same folder: keep finished seeds
        have = {json.loads(l)["seed"] for l in open(fn) if l.strip()}
    print(f"{runid}: {base}  seeds {seeds}  T={T}", flush=True)
    for i, sd in enumerate(seeds):
        if sd in have:
            continue
        job = dict(base, T=T, seed=sd, trace=a.trace, null=a.null, id=f"{runid}_s{sd}")
        off = i * T
        res = run(job, outdir=out, progress=lambda t, off=off: print(f"step {off + t}", flush=True))
        with open(fn, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"step {off + T}  seed {sd} done: gap {res['gap']:.5f}"
              + (f"  sin {res['sin_mean']:.4f}" if "sin_mean" in res else ""), flush=True)
    print("finished", flush=True)


QUESTIONS = {
    "N8": "Can a radio that does not know the change rate set its step online from the fitted law, and how close does it get to the best constant step (fixed and switching rates)?",
    "N7b": "Under variable rank, where are the outage optima on slots the tracker can fully null (at most kw active users)?",
    "N7": "Do the tuning laws hold under variable-rank occupancy (independent on/off users, births and departures), for all three trackers?",
    "N6": "Do the tuning laws and the theory's predictions hold with three active users (k = 3)?",
    "N4": "Do the tuning laws hold for complex uniform-linear arrays, for all three trackers?",
    "N5": "Are the exponents independent of SNR (4 and 16), as the linearised theory predicts?",
    "N2": "Do the two tuning regimes (power law for smooth costs, log law for threshold costs) hold for EW covariance and PAST, not just Oja?",
    "N3": "Do the regimes, the ordering of optimal steps, and the per-update cost ranking hold at M = 32 and 64?",
    "N1": "Does the decision rule set the tuning law? Optimal step vs jump rate for tracking error, argmin collisions, mean null-steering interference and interference outage.",
    "E1": "Do simpler trackers (beamformer power EWMA, PAST, EW covariance) match or beat the best constant-step Oja on reward gap?",
    "E2": "Does reset-on-detection help or hurt on reward and tracking error once the step is tuned, floored, and the detector has delay and false alarms?",
    "E3": "Where are the tracking-optimal and reward-optimal constant steps at each jump rate, and is the reward optimum interior?",
    "E5": "How long does the tracker sit above the collision onset after a swap, and does the hitting-time rate match the spectral prediction?",
    "E6": "Does the decaying step eta0/t fail for every eta0, as the recovery-time argument predicts?",
    "E7": "Under variable occupancy, does rank-adaptive scoring remove the r=5 collisions that fixed widths leave?",
    "E8": "Across array type, SNR and M, is there any regime where Oja tracking beats the beamformer baseline?",
}
BOOL_FLAGS = ("trace", "null")
NAMES = dict(N8="N8_online", N7b="N7b_varrank_rep", N7="N7_varrank", N6="N6_k3", N4="N4_ula", N5="N5_snr", N2="N2_trackers", N3="N3_scale", N1="N1_tuning_law", E1="E1_baselines", E2="E2_resets", E3="E3_stepsweep", E5="E5_excursions",
             E6="E6_decay", E7="E7_varrank", E8="E8_realism")


def fmt_seeds(ss):
    ss = sorted(ss)
    return f"{ss[0]}-{ss[-1]}" if ss == list(range(ss[0], ss[-1] + 1)) else ",".join(map(str, ss))


def write_sweeps(J, outdir="sweeps", per_line=None):
    os.makedirs(outdir, exist_ok=True)
    cells = {}
    for x in J:
        cfg = {k: v for k, v in x.items() if k not in ("id", "cost", "seed")}
        cells.setdefault(json.dumps(cfg, sort_keys=True), []).append(x["seed"])
    by_exp = {}
    for key, seeds in cells.items():
        cfg = json.loads(key)
        T = cfg.pop("T")
        bools = "".join(f" --{b}" for b in BOOL_FLAGS if cfg.pop(b, False))
        flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in sorted(cfg.items()))
        seeds = sorted(seeds)
        n = per_line or len(seeds)
        for i in range(0, len(seeds), n):
            ch = seeds[i:i + n]
            line = (f"python revsim.py cell {flags} --seeds {fmt_seeds(ch)}{bools}"
                    f" --total-timesteps {T * len(ch)}")
            by_exp.setdefault(cfg["exp"], []).append((T * len(ch), line))
    for e, lines in sorted(by_exp.items()):
        lines.sort(key=lambda x: -x[0])                     # longest first
        path = os.path.join(outdir, NAMES[e] + ".txt")
        with open(path, "w") as f:
            f.write(f"# question: {QUESTIONS[e]}\n")
            for _, l in lines:
                f.write(l + "\n")
        print(f"{path}: {len(lines)} runs, ~{sum(t for t, _ in lines) * 75e-6 / 3600:.1f} core-hours")


# ------------------------------------------------------------------ worker (standalone mode)
def done_ids(resdir):
    ids = set()
    if os.path.isdir(resdir):
        for fn in os.listdir(resdir):
            if fn.endswith(".jsonl"):
                for line in open(os.path.join(resdir, fn)):
                    try:
                        ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return ids


def safe_run(job):
    try:
        return run(job)
    except Exception as ex:                       # never lose a shard to one bad job
        return dict(job, error=repr(ex))


def worker(a):
    jobs = [json.loads(l) for l in open(a.jobs)]
    if a.only:
        jobs = [x for x in jobs if x["exp"] in a.only.split(",")]
    jobs.sort(key=lambda x: -x["cost"])            # longest first, round-robin => balanced
    mine = [x for i, x in enumerate(jobs) if i % a.nshards == a.shard]
    os.makedirs("results", exist_ok=True)
    skip = done_ids("results") | done_ids("done")        # done/ = results pushed back by launch.sh
    mine = [x for x in mine if x["id"] not in skip]
    fn = f"results/{socket.gethostname()}_shard{a.shard}.jsonl"
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} jobs, {a.workers} workers -> {fn}", flush=True)
    with Pool(a.workers) as pool, open(fn, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(safe_run, mine, chunksize=1), 1):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 10 == 0 or i == len(mine):
                print(f"  {i}/{len(mine)}  {time.strftime('%H:%M:%S')}", flush=True)


def selftest():
    # Table I, constant step, seed 0: paper's raw value 0.16155 (every 10th slot)
    r = run(dict(id="t1", method="oja", sched="const", eta=0.02, p=2e-4, T=120_000, seed=0,
                 sin_every=10))
    print("Table I const seed0 sin_mean", round(r["sin_mean"], 5), "(paper raw 0.16155)")
    r = run(dict(id="t7", method="oja", sched="const", eta=0.384, p=2e-4, T=120_000, seed=0))
    print("Table VII eta=0.384 seed0 reward", round(r["reward"], 5), "(paper raw 0.99927)")
    print("oracle reward", r["oracle_reward"], " secs", r["secs"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("jobs"); b.add_argument("--out", default="jobs.jsonl")
    b.add_argument("--seeds", type=int, default=20)
    b.add_argument("--scale", type=float, default=1.0, help="horizon multiplier (testing only)")
    w = sub.add_parser("worker"); w.add_argument("--jobs", default="jobs.jsonl")
    w.add_argument("--shard", type=int, default=0); w.add_argument("--nshards", type=int, default=1)
    w.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    w.add_argument("--only", default="", help="comma list of experiments, e.g. E2,E3")
    c = sub.add_parser("cell")
    for k, ty in FLAGS.items():
        c.add_argument("--" + k.replace("_", "-"), dest=k, type=ty, default=None)
    c.add_argument("--seeds", required=True)
    c.add_argument("--trace", action="store_true")
    c.add_argument("--null", action="store_true")
    c.add_argument("--total-timesteps", dest="total_timesteps", type=int, required=True)
    sw = sub.add_parser("sweeps"); sw.add_argument("--seeds", type=int, default=20)
    sw.add_argument("--scale", type=float, default=1.0)
    sw.add_argument("--out", default="sweeps")
    sw.add_argument("--set", choices=["revision", "null", "n2", "n45", "n6", "n7", "n7b", "n8"], default="revision")
    sw.add_argument("--per-line", dest="per_line", type=int, default=None)
    g = sub.add_parser("geometry"); g.add_argument("--seeds", type=int, default=20)
    g.add_argument("--out", default="geometry.json")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "jobs":
        J = build_jobs(a.seeds, a.scale)
        with open(a.out, "w") as f:
            for x in J:
                f.write(json.dumps(x) + "\n")
        from collections import Counter
        print(len(J), "jobs", dict(Counter(x["exp"] for x in J)))
        print(f"~{sum(x['cost'] for x in J) * 75e-6 / 3600:.0f} core-hours at ~75 us/slot")
    elif a.cmd == "cell":
        cell(a)
    elif a.cmd == "sweeps":
        if a.set == "n8":
            write_sweeps(build_n8_jobs(a.seeds if a.seeds != 20 else 10, a.scale), a.out, a.per_line or 5)
        elif a.set == "n7b":
            write_sweeps(build_n7_jobs(a.seeds if a.seeds != 20 else 10, a.scale, exp="N7b"), a.out,
                         a.per_line or 5)
        elif a.set == "n7":
            write_sweeps(build_n7_jobs(a.seeds if a.seeds != 20 else 10, a.scale), a.out, a.per_line or 5)
        elif a.set == "n6":
            write_sweeps(build_n6_jobs(a.seeds if a.seeds != 20 else 10, a.scale), a.out, a.per_line or 5)
        elif a.set == "n45":
            write_sweeps(build_n45_jobs(a.seeds if a.seeds != 20 else 10, a.scale), a.out, a.per_line or 5)
        elif a.set == "n2":
            write_sweeps(build_n2_jobs(min(a.seeds, 10) if a.seeds != 20 else 10, a.scale), a.out,
                         a.per_line or 5)
        elif a.set == "null":
            write_sweeps(build_null_jobs(a.seeds, a.scale), a.out, a.per_line or 5)
        else:
            write_sweeps(build_jobs(a.seeds, a.scale), a.out, a.per_line)
    elif a.cmd == "worker":
        worker(a)
    elif a.cmd == "geometry":
        geometry(a.seeds, a.out)
    else:
        selftest()
