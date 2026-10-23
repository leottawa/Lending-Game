"""
Lending Game - HECredit (Lender 2) - Stage 2 (Round 2)

    python stage2_pricing.py [data_folder]              -> writes Lender2_Round2_offers.csv (id, rate)
    python stage2_pricing.py [data_folder] --validate   -> split-sample backtest on Round 1 instead

The file actually submitted for Round 2 is written by `ch_model.py <this folder> <data_folder> --final`, which uses
this module with a level-k (cognitive hierarchy) prediction of competitors instead of the uniform ADAPT_PROB scenario.

data_folder must contain PastLoans.csv, NewApplications_Lender2_Round1.csv, Round1_Diagnostic_22.csv
and NewApplications_Lender2_Round2.csv.  Requirements: pandas, numpy, scipy, scikit-learn.

What changed from Stage 1
-------------------------
Stage 1 priced against a *prior* over competitors' scorecards and pricing rules. Round 1 showed that
prior was badly off: competitors refused risky applicants far more often than assumed, and one type-1
lender quoted 91.99% to about 14% of all applicants. Stage 2 replaces the prior by the *empirical*
distribution of competitor offers observed in Round 1.

1. Default model: unchanged GAM fitted on PastLoans. Our PD p0(x, u2) integrates over the two signals
   we do not see. Out of sample on Round 1 it predicted 9.93% defaults vs 9.95% realised.
2. Competitor offers: for a new applicant i, the pair of competitor offers (c1, c3) is drawn from the
   pairs received by the K Round 1 applicants with the same employment status and the closest public
   risk score A(x). Competitors' offers depend on public information and on their own signals, which
   are independent of ours, so matching on public risk is enough.
3. Winner's curse: a competitor's offer reveals part of its private signal. On Round 1 we fit
       logit P(D=1 | p0, c1, c3) = f(logit p0) + g(s1, s3) + logit p0 * g'(s1, s3)
   where s_k is the rank of competitor k's offer among offers to applicants with similar A(x), and a
   refusal is its own category.
4. Expected profit of rate r for applicant i, averaging over neighbours j and the three borrower types:
       Pi_i(r) = 1/(3K) sum_j sum_t 1{ r < T_t(c1_j, c3_j) } * [ r - (1 + r) * P(D | p0_i, c1_j, c3_j) ]
   T_1 = min(c1 - 2%, c3), T_2 = min(c1, c3) + 2%, T_3 = min(c1, c3 - 2%)   (we win on a strict "<").
   We offer argmax_r Pi_i(r) on a grid of rates ending in 9 (0.0049, 0.0059, ..., 0.9999) when it is > 0.
5. Adaptation: competitors also saw their Round 1 results and will move into the high-rate segment.
   With probability ADAPT_PROB, a competitor that refused or quoted >= ADAPT_FROM instead quotes a rate
   drawn uniformly in [ADAPT_LO, ADAPT_HI]. This is the main strategic assumption: raise ADAPT_PROB if
   you expect more undercutting of high quotes, set it to 0 to best-respond to Round 1 as it was.
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit, logit, ndtr
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import SplineTransformer

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
HERE = Path(ARGS[0]) if ARGS else Path(__file__).resolve().parent
VALIDATE = "--validate" in sys.argv
PAST_FILE = HERE / "PastLoans.csv"
R1_FILE = HERE / "NewApplications_Lender2_Round1.csv"
DIAG_FILE = HERE / "Round1_Diagnostic_22.csv"
NEW_FILE = HERE / "NewApplications_Lender2_Round2.csv"
OUT_FILE = HERE / "Lender2_Round2_offers.csv"

EMP = ["employed", "retired", "student", "unemployed"]
SIGS = ["signal1", "signal2", "signal3"]
LOAN = 10_000
DELTA = 0.02                       # loyalty premium of each borrower type

# ---- Stage 2 parameters
K_NEIGH = 500                      # Round 1 neighbours per applicant
N_BINS = 40                        # public-risk bins used to rank competitor offers
ADAPT_PROB = 0.25                  # P(a refusing / high-quoting competitor now undercuts high quotes)
ADAPT_FROM = 0.50                  # responses that can adapt: refusal or a quote >= 50%
ADAPT_LO, ADAPT_HI = 0.50, 0.95    # adapted quotes ~ Uniform[ADAPT_LO, ADAPT_HI]
ADAPT_REPS = 8                     # Monte Carlo replicates of each neighbour's offers
PD_SHIFT = 0.0                     # extra log-odds safety margin on default probabilities
RGRID = np.round(0.0049 + 0.001 * np.arange(996), 4)   # 0.0049 ... 0.9999


# ============================================================ 1. default model (Stage 1, unchanged)
class SignalRanker:
    """Within-employment empirical CDF of the non-zero signal values (three signals pooled)."""

    def fit(self, past):
        self.ref = {}
        for e in EMP:
            g = past[past.employment == e]
            v = np.concatenate([g[s].values for s in SIGS])
            self.ref[e] = np.sort(v[v > 0])
        return self

    def rank(self, emp, sig):
        out = np.full(len(sig), np.nan)
        emp, sig = np.asarray(emp), np.asarray(sig, dtype=float)
        for e in EMP:
            m = (emp == e) & (sig > 0)
            out[m] = np.searchsorted(self.ref[e], sig[m], side="right") / len(self.ref[e])
        return out


class GAM:
    """logit P(D=1) = employment + male + married + f(income) + zero-income(student/unemployed)
                      + sum_k h(u_k)        (h shared across the signals given, h(missing) = const)"""

    def __init__(self, n_inc_knots=4, inc_tf="log", n_sig_knots=6, C=10.0):
        self.n_inc_knots, self.inc_tf, self.n_sig_knots, self.C = n_inc_knots, inc_tf, n_sig_knots, C

    def _inc(self, inc):
        return {"log": np.log, "sqrt": np.sqrt, "asinh": lambda v: np.arcsinh(v / 5000.0)}[self.inc_tf](inc)

    def _design(self, df, U, fit=False):
        emp = np.column_stack([(df["employment"].values == e) for e in EMP]).astype(float)
        inc = df["income"].values.astype(float)
        pos = inc > 0
        li = self._inc(np.where(pos, inc, 1.0))[:, None]
        if fit:
            self.ist = SplineTransformer(n_knots=self.n_inc_knots, degree=3, knots="quantile",
                                         extrapolation="linear").fit(li[pos])
            self.sst = SplineTransformer(n_knots=self.n_sig_knots, degree=3, knots="quantile",
                                         extrapolation="linear").fit(np.linspace(0.0005, 0.9995, 200)[:, None])
        cols = [emp[:, 1:], (df["sex"].values == "M")[:, None].astype(float),
                df["married"].values[:, None].astype(float),
                self.ist.transform(li) * pos[:, None],
                ((~pos) & (emp[:, 2] == 1))[:, None].astype(float),
                ((~pos) & (emp[:, 3] == 1))[:, None].astype(float)]
        S, Mi = 0.0, 0.0
        for k in range(U.shape[1]):
            u = U[:, k]
            miss = np.isnan(u)
            S = S + self.sst.transform(np.where(miss, 0.5, u)[:, None]) * (~miss)[:, None]
            Mi = Mi + miss[:, None]
        return np.hstack(cols + [S, np.asarray(Mi, dtype=float)])

    def fit(self, df, U, y):
        self.m = LogisticRegression(C=self.C, max_iter=20000).fit(self._design(df, U, fit=True), y)
        return self

    def decision(self, df, U):
        return self.m.decision_function(self._design(df, U))


class Truth:
    """Average (in logit space) of three GAM variants fitted with all three signals; it stays additive:
    logit P(D | x, u1, u2, u3) = A(x) + h(u1) + h(u2) + h(u3), with h(missing) = 0."""

    VARIANTS = [dict(n_inc_knots=4, inc_tf="log"), dict(n_inc_knots=5, inc_tf="sqrt"),
                dict(n_inc_knots=4, inc_tf="asinh")]

    def __init__(self, past, U):
        y = past["default"].values
        self.models = [GAM(**kw).fit(past, U, y) for kw in self.VARIANTS]
        self.q = {e: float(np.mean(np.isnan(U[past.employment.values == e]))) for e in EMP}
        row, nan3 = past.iloc[:1], np.full((1, 3), np.nan)
        base = np.mean([m.decision(row, nan3)[0] for m in self.models])
        self.ug = np.linspace(0.0, 1.0, 2001)
        self.hv = np.array([np.mean([m.decision(row, np.array([[u, np.nan, np.nan]]))[0]
                                     for m in self.models]) for u in self.ug]) - base

    def A(self, df):
        nan3 = np.full((len(df), 3), np.nan)
        return np.mean([m.decision(df, nan3) for m in self.models], axis=0)

    def h(self, u):
        u = np.asarray(u, dtype=float)
        return np.where(np.isnan(u), 0.0, np.interp(np.nan_to_num(u, nan=0.5), self.ug, self.hv))


M = 50                                     # cells for an unseen signal rank (+1 cell = missing)
UMID = (np.arange(M) + 0.5) / M
RHO = 0.045                                # rank correlation between lenders' signals


def signal_weights(emp, u2, q, rho):
    """P(competitor signal cell | our u2): missing w.p. q_emp, else Gaussian-copula ranks (corr rho)."""
    qv = np.array([q[e] for e in emp])
    z2 = np.where(np.isnan(u2), 0.0, norm.ppf(np.clip(np.nan_to_num(u2, nan=0.5), 1e-4, 1 - 1e-4)))
    edges = norm.ppf(np.linspace(0, 1, M + 1))
    cell = np.diff(ndtr((edges[None, :] - rho * z2[:, None]) / np.sqrt(1 - rho ** 2)), axis=1)
    return np.column_stack([qv, (1 - qv)[:, None] * cell])


class DefaultModel:
    def __init__(self, past):
        self.ranker = SignalRanker().fit(past)
        U = np.column_stack([self.ranker.rank(past.employment.values, past[s].values) for s in SIGS])
        self.truth = Truth(past, U)

    def score(self, df):
        """Public risk score A(x) and our PD p0(x, u2) = E[ sigmoid(A + h(u1) + h(u2) + h(u3)) | x, u2 ]."""
        t = self.truth
        u2 = self.ranker.rank(df.employment.values, df.signal2.values)
        A, h2 = t.A(df), t.h(u2)
        w = signal_weights(df.employment.values, u2, t.q, RHO)
        hm = np.r_[0.0, t.h(UMID)]
        H = hm[:, None] + hm[None, :]
        p0 = np.empty(len(df))
        for s in range(0, len(df), 2000):
            sl = slice(s, s + 2000)
            P = expit(A[sl, None, None] + h2[sl, None, None] + H[None])
            p0[sl] = np.einsum("ia,ib,iab->i", w[sl], w[sl], P)
        return A, p0


# ============================================================ 2. winner's curse from competitor offers
def offer_ranks(A, c, n_bins=N_BINS):
    """Rank (0-1] of each competitor offer among offers to applicants in the same public-risk bin.
    NaN = no offer."""
    bins = pd.qcut(A, n_bins, labels=False, duplicates="drop")
    return pd.Series(c).groupby(bins).rank(pct=True).values


class WinnersCurse:
    """logit P(D=1 | p0, s1, s3) = f(L) + B(s1, s3) . b0 + Lz * B(s1, s3) . b1,  L = logit p0."""

    def _comp(self, s1, s3):
        cols = []
        for s in (s1, s3):
            r = np.isnan(s)
            cols += [r[:, None].astype(float), self.spS.transform(np.where(r, 0.5, s)[:, None]) * (~r)[:, None]]
        return np.hstack(cols)

    def _design(self, L, s1, s3, fit=False):
        if fit:
            self.Lm, self.Ls = L.mean(), L.std()
            self.spL = SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="linear").fit(L[:, None])
            self.spS = SplineTransformer(n_knots=5, degree=3).fit(np.linspace(0, 1, 101)[:, None])
        Cb = self._comp(s1, s3)
        return np.hstack([self.spL.transform(L[:, None]), Cb, Cb * self.Lz(L)[:, None]])

    def Lz(self, L):
        return (L - self.Lm) / self.Ls

    def fit(self, L, s1, s3, y, C=1.0):
        self.m = LogisticRegression(C=C, max_iter=5000).fit(self._design(L, s1, s3, fit=True), y)
        coef, nb = self.m.coef_[0], self.spL.n_features_out_
        q = (len(coef) - nb) // 2
        self.bL, self.b0, self.b1 = coef[:nb], coef[nb:nb + q], coef[nb + q:]
        return self

    def predict(self, L, s1, s3):
        return self.m.predict_proba(self._design(L, s1, s3))[:, 1]

    def F(self, L):
        return self.m.intercept_[0] + self.spL.transform(L[:, None]) @ self.bL

    def G(self, s1, s3):
        Cb = self._comp(s1, s3)
        return Cb @ self.b0, Cb @ self.b1


# ============================================================ 3. empirical best response
def adapt_offers(c, prob, rng, reps):
    """reps replicates of competitor offers c (inf = no offer) under the adaptation scenario."""
    C = np.repeat(c[:, None], reps, axis=1)
    if prob > 0:
        flip = (rng.random(C.shape) < prob) & ((C >= ADAPT_FROM) | ~np.isfinite(C))
        C[flip] = rng.uniform(ADAPT_LO, ADAPT_HI, flip.sum())
    return C


def thresholds(C1, C3):
    """Rate below which we win, for borrower types 1, 2, 3 (stacked on the last axis)."""
    return np.stack([np.minimum(C1 - DELTA, C3), np.minimum(C1, C3) + DELTA, np.minimum(C1, C3 - DELTA)], axis=-1)


class Pool:
    """Round 1 applicants used as neighbours: their competitor offers and winner's-curse terms."""

    def __init__(self, emp, A, c1, c3, G0, G1, adapt_prob, seed=0):
        reps = ADAPT_REPS if adapt_prob > 0 else 1
        rng = np.random.default_rng(seed)
        c1 = np.where(np.isnan(c1), np.inf, c1)
        c3 = np.where(np.isnan(c3), np.inf, c3)
        T = thresholds(adapt_offers(c1, adapt_prob, rng, reps), adapt_offers(c3, adapt_prob, rng, reps))
        kidx = np.searchsorted(RGRID, T, side="left").reshape(len(A), -1).astype(np.int16)  # win iff grid idx < k
        code = pd.Categorical(emp, categories=EMP).codes
        order = np.lexsort((A, code))
        self.A, self.code, self.k, self.G0, self.G1 = A[order], code[order], kidx[order], G0[order], G1[order]
        self.start = np.searchsorted(self.code, np.arange(len(EMP)), side="left")
        self.end = np.searchsorted(self.code, np.arange(len(EMP)), side="right")

    def neighbours(self, emp, A, K):
        code = pd.Categorical(emp, categories=EMP).codes
        lo = np.empty(len(A), dtype=np.int64)
        for e in range(len(EMP)):
            m = code == e
            s, t = self.start[e], self.end[e]
            pos = s + np.searchsorted(self.A[s:t], A[m])
            lo[m] = np.clip(pos - K // 2, s, t - K)
        return lo[:, None] + np.arange(K)[None, :]


def best_response(pool, emp, A, L, wc, K=K_NEIGH, pd_shift=PD_SHIFT, chunk=400):
    """argmax_r Pi_i(r) for each applicant. wc=None ignores the winner's curse (P(D) = p0)."""
    n, G = len(A), len(RGRID)
    idx_all = pool.neighbours(emp, A, K)
    F = wc.F(L) if wc is not None else L
    Lz = wc.Lz(L) if wc is not None else np.zeros(n)
    out = {k: np.empty(n) for k in ["rate", "eprofit", "pwin", "pd_win"]}
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        idx = idx_all[sl]
        c = idx.shape[0]
        z = F[sl, None] + pd_shift
        if wc is not None:
            z = z + pool.G0[idx] + Lz[sl, None] * pool.G1[idx]
        P = np.broadcast_to(expit(z), idx.shape)                                   # (c, K)
        k = pool.k[idx].astype(np.int64)                                           # (c, K, reps*3)
        flat = (np.arange(c)[:, None, None] * (G + 1) + k).ravel()
        wts = np.broadcast_to(P[:, :, None], k.shape).ravel()
        hn = np.bincount(flat, minlength=c * (G + 1)).reshape(c, G + 1)
        hd = np.bincount(flat, weights=wts, minlength=c * (G + 1)).reshape(c, G + 1)
        tot = k.shape[1] * k.shape[2]
        Nw = np.cumsum(hn[:, ::-1], axis=1)[:, ::-1][:, 1:] / tot                  # P(win at RGRID[g])
        Dw = np.cumsum(hd[:, ::-1], axis=1)[:, ::-1][:, 1:] / tot                  # P(win and default)
        Pi = RGRID[None, :] * Nw - (1 + RGRID[None, :]) * Dw
        g = np.argmax(Pi, axis=1)
        rr = np.arange(c)
        out["rate"][sl], out["eprofit"][sl], out["pwin"][sl] = RGRID[g], Pi[rr, g], Nw[rr, g]
        out["pd_win"][sl] = Dw[rr, g] / np.maximum(Nw[rr, g], 1e-12)
    out["offer"] = out["eprofit"] > 0
    out["rate"] = np.where(out["offer"], out["rate"], np.nan)
    return out


def build_pool(r1, A1, L1, adapt_prob, use_wc=True, seed=0):
    s1, s3 = offer_ranks(A1, r1.competing1.values), offer_ranks(A1, r1.competing3.values)
    wc = WinnersCurse().fit(L1, s1, s3, r1["default"].values) if use_wc else None
    G0, G1 = wc.G(s1, s3) if use_wc else (np.zeros(len(A1)), np.zeros(len(A1)))
    return Pool(r1.employment.values, A1, r1.competing1.values, r1.competing3.values, G0, G1, adapt_prob, seed), wc


# ============================================================ 4. replay (validation)
def replay(rate, c1, c3, btype, D):
    """Realised profit of our offers given competitors' offers (NaN = no offer)."""
    r = np.where(np.isnan(rate), np.inf, rate)
    c1, c3 = np.where(np.isnan(c1), np.inf, c1), np.where(np.isnan(c3), np.inf, c3)
    eo = np.where(btype == 2, r - DELTA, r)
    e1 = np.where(btype == 1, c1 - DELTA, c1)
    e3 = np.where(btype == 3, c3 - DELTA, c3)
    win = (eo < e1 - 1e-9) & (eo < e3 - 1e-9) & np.isfinite(r)
    return np.where(win, np.where(D == 1, -LOAN, r * LOAN), 0.0)


def replay_adapted(rate, r1, true_prob, n_mc=20, seed=123):
    """Mean realised profit when competitors adapt with probability true_prob (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    c1 = np.where(r1.competing1.isna(), np.inf, r1.competing1.values)
    c3 = np.where(r1.competing3.isna(), np.inf, r1.competing3.values)
    tot = 0.0
    for _ in range(n_mc if true_prob > 0 else 1):
        a1 = adapt_offers(c1, true_prob, rng, 1)[:, 0]
        a3 = adapt_offers(c3, true_prob, rng, 1)[:, 0]
        tot += replay(rate, a1, a3, r1.borrowertype.values, r1["default"].values).sum()
    return tot / (n_mc if true_prob > 0 else 1)


def validate(r1, A1, L1):
    """Two-fold cross-fit on Round 1: build the pool on one half, price the other half, replay it."""
    half = (r1.id.values % 2).astype(bool)
    y = r1["default"].values
    # 1. does the winner's-curse model add information beyond our PD?
    ll = []
    for tr in (half, ~half):
        te = ~tr
        s1, s3 = offer_ranks(A1, r1.competing1.values), offer_ranks(A1, r1.competing3.values)
        wc = WinnersCurse().fit(L1[tr], s1[tr], s3[tr], y[tr])
        ll.append((log_loss(y[te], expit(L1[te])), log_loss(y[te], wc.predict(L1[te], s1[te], s3[te]))))
    ll = np.mean(ll, axis=0)
    print(f"out-of-fold log-loss: our PD alone {ll[0]:.5f} | with competitor offers {ll[1]:.5f}")

    true_probs = [0.0, 0.25, 0.5]
    policies = {"Stage 1 offers (submitted)": None, "Stage 1, 97.89% quotes -> 91.89%": "fix"}
    for p in [0.0, 0.25, 0.5]:
        policies[f"Stage 2, ADAPT_PROB={p}"] = (p, True)
    policies["Stage 2, ADAPT_PROB=0.25, no winner's curse"] = (0.25, False)
    rows = []
    for name, pol in policies.items():
        t0 = time.time()
        rate = np.full(len(r1), np.nan)
        if pol is None:
            rate = r1.own.values.copy()
        elif pol == "fix":
            rate = np.where(r1.own.values >= 0.9, 0.9189, r1.own.values)
        else:
            for tr in (half, ~half):
                te = ~tr
                pool, wc = build_pool(r1[tr], A1[tr], L1[tr], pol[0], use_wc=pol[1])
                res = best_response(pool, r1.employment.values[te], A1[te], L1[te], wc)
                rate[te] = res["rate"]
        row = {"policy": name, "offers": int(np.sum(~np.isnan(rate)))}
        for tp in true_probs:
            row[f"profit if adapt={tp} (M EUR)"] = round(replay_adapted(rate, r1, tp) / 1e6, 2)
        rows.append(row)
        print(f"  {name}: done in {time.time() - t0:.0f}s", flush=True)
    print(pd.DataFrame(rows).set_index("policy").to_string())


# ============================================================ main
def main():
    t0 = time.time()
    past = pd.read_csv(PAST_FILE)
    r1 = pd.read_csv(R1_FILE).merge(pd.read_csv(DIAG_FILE), on="id", validate="1:1")
    dm = DefaultModel(past)
    A1, p01 = dm.score(r1)
    L1 = logit(np.clip(p01, 1e-6, 1 - 1e-6))
    print(f"default model fitted, Round 1 scored ({time.time() - t0:.0f}s)")
    if VALIDATE:
        validate(r1, A1, L1)
        return

    new = pd.read_csv(NEW_FILE)
    A2, p02 = dm.score(new)
    L2 = logit(np.clip(p02, 1e-6, 1 - 1e-6))
    pool, wc = build_pool(r1, A1, L1, ADAPT_PROB)
    res = best_response(pool, new.employment.values, A2, L2, wc)
    rate = np.round(res["rate"], 4)
    assert np.all(np.isnan(rate) | ((rate > 0) & (rate <= 1.0)))
    pd.DataFrame({"id": new["id"], "rate": rate}).to_csv(OUT_FILE, index=False, float_format="%.4f")

    offer = res["offer"]
    print(f"wrote {OUT_FILE}: {offer.sum()} offers / {len(new)} applications ({time.time() - t0:.0f}s)")
    print(f"expected loans won {res['pwin'][offer].sum():,.0f}, expected profit "
          f"{res['eprofit'][offer].sum() * LOAN / 1e6:,.2f}M EUR (Round 1 competitors, ADAPT_PROB={ADAPT_PROB})")
    d = pd.DataFrame({"pd": p02, "rate": rate, "pwin": res["pwin"], "pd_win": res["pd_win"],
                      "eprofit": res["eprofit"] * LOAN})
    d["pd_band"] = pd.cut(d["pd"], [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0])
    print(d.groupby("pd_band", observed=True).agg(n=("pd", "size"), offers=("rate", "count"),
                                                  median_rate=("rate", "median"), p_win=("pwin", "mean"),
                                                  pd_if_won=("pd_win", "mean"),
                                                  exp_profit_eur=("eprofit", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
