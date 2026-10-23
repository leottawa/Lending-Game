"""
Lending Game - Lender 2 - Stage 1 (Round 1)

Produces the stage-1 submission: one interest-rate offer (or no offer) per loan application.

    python stage1_pricing.py [data_folder]   -> writes Lender2_Round1_offers.csv (columns: id, rate)

Requirements: pandas, numpy, scipy, scikit-learn.  Runtime: about 1 minute.

Method
------
1. Default model. On PastLoans we fit an additive logistic model (spline GAM) that uses all three
   lenders' signals:  logit P(D=1 | x, u1, u2, u3) = A(x) + h(u1) + h(u2) + h(u3).
   x = employment, sex, married, income (spline in log income + zero-income dummies);
   u_k = rank of signal k inside the applicant's employment group. A signal equal to 0 carries no
   information (default rate equals the group average), so it is treated as missing: h = 0.
   The three signals have identical distributions and effects, so they share one function h,
   estimated on 3x more data. This model beat gradient boosting in 5-fold cross-validation.
2. Winner's curse. Lenders 1 and 3 see their own signals, which we do not. For every rate r we
   integrate over their unknown signals (u1, u3 given our u2) and over a prior on how they score
   and price applicants (model type, mark-up rule, refusal cut-off, noise):
       Pi(r) = E[ 1{borrower picks us} * ( r*(1-D) - D ) | x, u2 ]
   with the three borrower types (2% loyalty premium). We tend to win exactly when competitors saw
   bad news, so Pi(r) uses the default rate conditional on winning, not our raw PD.
3. Decision. Offer argmax_r Pi(r) when that maximum is positive; otherwise make no offer.
   PDs get a small safety margin (+0.05 on the log-odds) against model error, and every rate is
   shaded by 0.0001 to win ties against round-number offers.

Fintech name: HECredit
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit, logit, ndtr
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
PAST_FILE = HERE / "PastLoans.csv"
NEW_FILE = HERE / "NewApplications_Lender2_Round1.csv"
OUT_FILE = HERE / "Lender2_Round1_offers.csv"

EMP = ["employed", "retired", "student", "unemployed"]
SIGS = ["signal1", "signal2", "signal3"]
LOAN = 10_000
DELTA = 0.02                      # loyalty premium of each borrower type


# ============================================================ 1. features
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

    def raw(self, e, u):
        return np.quantile(self.ref[e], u)


# ============================================================ 2. default model (GAM)
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


# ============================================================ 3. competitors
M = 50                                     # cells for a competitor's signal rank (+1 cell = missing)
UMID = (np.arange(M) + 0.5) / M
DT = 0.001                                 # rate grid step
TGRID = np.round(np.arange(-0.020, 1.020 + 1e-9, DT), 6)
SH = int(round(DELTA / DT))
RATE_IDX = np.where((TGRID > 0.0005) & (TGRID <= 1.0 + 1e-9))[0]   # rates we may offer
LGRID = np.arange(-14.0, 6.0 + 1e-9, 0.01)                         # competitor base-logit grid


def raw_frame(df, sig):
    e = df["employment"].values
    return pd.DataFrame({"male": (df["sex"].values == "M").astype(float), "married": df["married"].values.astype(float),
                         "ret": (e == "retired").astype(float), "stu": (e == "student").astype(float),
                         "une": (e == "unemployed").astype(float), "income": df["income"].values.astype(float),
                         "sig": np.asarray(sig, dtype=float)})


def fit_competitor_models(past):
    """Two typical competitor scorecards on raw features (each competitor sees one signal):
    a gradient-boosting model and a plain logistic regression."""
    st = pd.concat([raw_frame(past, past[s].values) for s in SIGS], ignore_index=True)
    ys = np.tile(past["default"].values, 3)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(st, ys)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(st, ys)
    return {"gbm": gbm, "lr": lr}


def competitor_logits(df, truth, comp_models, ranker, chunk=4000):
    """Base logit of each competitor model type at each of the M+1 competitor-signal cells."""
    n, emp, A = len(df), df["employment"].values, truth.A(df)
    hm = np.r_[0.0, truth.h(UMID)]
    out = {"ideal": np.empty((n, M + 1))}
    for e in EMP:
        # ideal one-signal model: P(D | x, u_k) = E[ sigmoid(A + h(u_k) + h(u_a) + h(u_b)) ]
        q = truth.q[e]
        wv = np.r_[q, np.full(M, (1 - q) / M)]
        W, H = np.outer(wv, wv).ravel(), (hm[:, None] + hm[None, :]).ravel()
        tab = np.array([W @ expit(L + H) for L in LGRID])
        m = emp == e
        out["ideal"][m] = logit(np.clip(np.interp(A[m][:, None] + hm[None, :], LGRID, tab), 1e-7, 1 - 1e-7))
    raws = np.zeros((n, M + 1))
    for e in EMP:
        raws[emp == e, 1:] = ranker.raw(e, UMID)[None, :]
    for name, mdl in comp_models.items():
        res = np.empty((n, M + 1))
        for s in range(0, n, chunk):
            sl = slice(s, min(n, s + chunk))
            rep = df.iloc[np.repeat(np.arange(sl.start, sl.stop), M + 1)]
            p = mdl.predict_proba(raw_frame(rep, raws[sl].ravel()))[:, 1]
            res[sl] = logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, M + 1)
        out[name] = res
    return out


def signal_weights(emp, u2, q, rho):
    """P(competitor signal cell | our u2): missing w.p. q_emp, else Gaussian-copula ranks (corr rho)."""
    qv = np.array([q[e] for e in emp])
    z2 = np.where(np.isnan(u2), 0.0, norm.ppf(np.clip(np.nan_to_num(u2, nan=0.5), 1e-4, 1 - 1e-4)))
    edges = norm.ppf(np.linspace(0, 1, M + 1))
    cell = np.diff(ndtr((edges[None, :] - rho * z2[:, None]) / np.sqrt(1 - rho ** 2)), axis=1)
    return np.column_stack([qv, (1 - qv)[:, None] * cell])


def theta(family, m, t):
    """PD threshold of a pricing rule: competitor rate > t  <=>  p_hat > theta(t)."""
    t = np.asarray(t, dtype=float)
    if family == "add":        # r = p/(1-p) + m
        x = t - m
        return np.where(x > 0, x / (1 + np.maximum(x, 0)), 0.0)
    if family == "mult":       # r = (1+m) p/(1-p)
        return np.where(t > 0, t / (1 + m + np.maximum(t, 0)), 0.0)
    raise ValueError(family)


def survival_table(pricing, cutoffs, sigma):
    """S[t, l] = P(competitor rate > t | competitor base logit l), mixing pricing rules and cut-offs.
    p_hat = sigmoid(l + sigma*eps). No offer = rate +inf; a formula above 100% is not offered.
    Cut-off c: no offer if p_hat > c.  ("max", tau, c): quote 100% if tau < p_hat <= c."""
    S = np.zeros((len(TGRID), len(LGRID)))
    below1 = TGRID < 1.0
    for (fam, m), wp in pricing.items():
        th, th1 = theta(fam, m, np.minimum(TGRID, 1.0)), float(theta(fam, m, 1.0))
        for c, wc in cutoffs.items():
            if isinstance(c, tuple):
                thr = np.where(below1, np.minimum(th, c[1]), min(th1, c[2]))
            else:
                thr = np.where(below1, np.minimum(th, c), min(th1, c))
            lt = np.where(thr <= 0, -np.inf, logit(np.clip(thr, 1e-12, 1 - 1e-12)))
            S += wp * wc * ndtr((LGRID[None, :] - lt[:, None]) / sigma)
    return S.astype(np.float32)


# ============================================================ 4. expected profit and best rate
def best_rates(A, h2, w, comp_idx, type_w, S_tab, truth, pd_shift=0.0, chunk=500):
    """For each applicant, Pi(r) on the rate grid and its argmax.
    Type 2 (prefers us):    win iff r1 > r-2% and r3 > r-2%
    Type 1 (prefers lender 1): win iff r1 > r+2% and r3 > r   (type 3 symmetric)
    Pi(r) = sum_{u1,u3} w(u1) w(u3) P(win | u1,u3) [ r - (1+r) PD(x,u1,u2,u3) ]"""
    n = len(A)
    hm = np.r_[0.0, truth.h(UMID)]
    H = hm[:, None] + hm[None, :]
    R = TGRID[RATE_IDX]
    j = RATE_IDX
    out = {k: np.empty(n) for k in ["rate", "profit", "pwin", "pd_win", "pd"]}
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        P = expit(A[sl, None, None] + h2[sl, None, None] + H[None] + pd_shift).astype(np.float32)
        Sv = sum(tw * S_tab[:, comp_idx[t][sl]] for t, tw in type_w.items())       # (T, k, M+1)
        a = np.transpose(Sv, (1, 0, 2)) * w[sl][:, None, :].astype(np.float32)      # (k, T, M+1)
        B = np.matmul(a, P)
        sa = a.sum(axis=2)
        pwin = (sa[:, j - SH] ** 2 + 2 * sa[:, j + SH] * sa[:, j]) / 3
        pdw = (np.einsum("ktm,ktm->kt", B[:, j - SH], a[:, j - SH])
               + 2 * np.einsum("ktm,ktm->kt", B[:, j + SH], a[:, j])) / 3               # E[win * D]
        prof = R[None, :] * pwin - (1 + R[None, :]) * pdw
        b = np.argmax(prof, axis=1)
        rr = np.arange(len(b))
        out["rate"][sl], out["profit"][sl], out["pwin"][sl] = R[b], prof[rr, b], pwin[rr, b]
        out["pd_win"][sl] = pdw[rr, b] / np.maximum(pwin[rr, b], 1e-12)
        out["pd"][sl] = (w[sl][:, :, None] * w[sl][:, None, :] * P).sum(axis=(1, 2))
    return out


def lgrid_index(L):
    return np.clip(np.rint((L - LGRID[0]) / 0.01).astype(np.int64), 0, len(LGRID) - 1)


# ============================================================ 5. prior on competitors' behaviour
# Mark-up rules (r = p/(1-p) + m, or r = (1+m) p/(1-p)): aggressive / moderate / conservative teams.
AGG = {("add", 0.0): 0.2, ("add", 0.005): 0.2, ("add", 0.01): 0.2, ("add", 0.015): 0.2, ("add", 0.02): 0.2}
MOD = {("add", 0.02): 0.2, ("add", 0.025): 0.2, ("add", 0.03): 0.2, ("add", 0.035): 0.2, ("add", 0.04): 0.2}
CON = {("add", 0.05): 0.25, ("add", 0.07): 0.25, ("mult", 0.5): 0.25, ("mult", 1.0): 0.25}
# Refusal behaviour: lend to all (PD < 50%), refuse above 30-40%, refuse above 15-25%,
# or quote 100% to risky applicants.
NONE = {1.0: 1.0}
LOOSE = {0.3: 0.5, 0.4: 0.5}
TIGHT = {0.15: 1 / 3, 0.2: 1 / 3, 0.25: 1 / 3}
MONO = {("max", 0.15, 0.5): 0.5, ("max", 0.25, 0.5): 0.5}


def mix(*parts):
    out = {}
    for d, wt in parts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + wt * v
    return out


PRICING = mix((AGG, 0.3), (MOD, 0.4), (CON, 0.3))
CUTOFFS = mix((NONE, 0.3), (LOOSE, 0.3), (TIGHT, 0.3), (MONO, 0.1))
MODEL_TYPES = {"ideal": 0.35, "gbm": 0.45, "lr": 0.20}   # how good their scorecards are
SIGMA = 0.25                                             # extra noise in their PD (logit scale)
RHO = 0.045                                              # rank correlation between lenders' signals
PD_SHIFT = 0.05                                          # safety margin added to the log-odds of default (~+5% PD)


# ============================================================ main
def main():
    past = pd.read_csv(PAST_FILE)
    new = pd.read_csv(NEW_FILE)
    ranker = SignalRanker().fit(past)
    U = np.column_stack([ranker.rank(past.employment.values, past[s].values) for s in SIGS])
    truth = Truth(past, U)

    u2 = ranker.rank(new.employment.values, new.signal2.values)
    A, h2 = truth.A(new), truth.h(u2)
    w = signal_weights(new.employment.values, u2, truth.q, RHO)
    comp = competitor_logits(new, truth, fit_competitor_models(past), ranker)
    comp_idx = {k: lgrid_index(v) for k, v in comp.items()}
    res = best_rates(A, h2, w, comp_idx, MODEL_TYPES, survival_table(PRICING, CUTOFFS, SIGMA), truth,
                     pd_shift=PD_SHIFT)

    offer = res["profit"] > 0
    # shade every rate by 0.01 percentage point to win ties against round-number offers
    rate = np.where(offer, np.round(res["rate"] - 0.0001, 4), np.nan)
    pd.DataFrame({"id": new["id"], "rate": rate}).to_csv(OUT_FILE, index=False, float_format="%.4f")

    print(f"wrote {OUT_FILE.name}: {offer.sum()} offers / {len(new)} applications")
    print(f"expected loans won {res['pwin'][offer].sum():,.0f}, expected profit "
          f"{res['profit'][offer].sum() * LOAN / 1e6:,.2f}M EUR (competitor prior, PDs incl. safety margin)")
    d = pd.DataFrame({"pd": res["pd"], "rate": rate, "pwin": res["pwin"], "pd_win": res["pd_win"],
                      "eprofit": res["profit"] * LOAN})
    d["pd_band"] = pd.cut(d["pd"], [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0])
    print(d.groupby("pd_band", observed=True).agg(n=("pd", "size"), offers=("rate", "count"),
                                                  median_rate=("rate", "median"), p_win=("pwin", "mean"),
                                                  pd_if_won=("pd_win", "mean"),
                                                  exp_profit_eur=("eprofit", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
