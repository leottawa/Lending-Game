"""Try model improvements with the cross-fitted Round 1 replay. usage: python -I improve.py <stage2_dir> <data_dir>"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[:1] + [sys.argv[2]]
import stage2_pricing as s2
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
D = s2.HERE
past = pd.read_csv(D / "PastLoans.csv")
r1 = pd.read_csv(D / "NewApplications_Lender2_Round1.csv").merge(pd.read_csv(D / "Round1_Diagnostic_22.csv"), on="id")
dm = s2.DefaultModel(past); A1, p01 = dm.score(r1); L1 = logit(np.clip(p01, 1e-6, 1 - 1e-6))
n = len(r1); half = (r1.id.values % 2).astype(bool); y = r1["default"].values
emp = r1.employment.values; E = np.column_stack([(emp == e) for e in s2.EMP[1:]]).astype(float)
c1, c3, bt = r1.competing1.values, r1.competing3.values, r1.borrowertype.values
BE = p01 / (1 - p01) + 0.03

class WCemp(s2.WinnersCurse):
    """adds employment effects and employment x competitor-offer interactions"""
    def fit(self, L, s1, s3, y, Eq, C=1.0):
        X0 = self._design(L, s1, s3, fit=True); Cb = self._comp(s1, s3)
        X = np.hstack([X0, Eq] + [Cb * Eq[:, [k]] for k in range(Eq.shape[1])])
        self.m = LogisticRegression(C=C, max_iter=5000).fit(X, y)
        coef, nb = self.m.coef_[0], self.spL.n_features_out_
        q = Cb.shape[1]
        self.bL, self.b0, self.b1 = coef[:nb], coef[nb:nb + q], coef[nb + q:nb + 2 * q]
        self.bE = coef[nb + 2 * q:nb + 2 * q + Eq.shape[1]]
        self.bEC = coef[nb + 2 * q + Eq.shape[1]:].reshape(Eq.shape[1], q)
        return self
    def predict(self, L, s1, s3, Eq):
        Cb = self._comp(s1, s3)
        X = np.hstack([self._design(L, s1, s3), Eq] + [Cb * Eq[:, [k]] for k in range(Eq.shape[1])])
        return self.m.predict_proba(X)[:, 1]
    def F(self, L):
        return self.m.intercept_[0] + self.spL.transform(L[:, None]) @ self.bL + self.Eq_query @ self.bE
    def Gemp(self, s1, s3, Eq):
        Cb = self._comp(s1, s3)
        return Cb @ self.b0 + np.einsum("ik,kq,iq->i", Eq, self.bEC, Cb), Cb @ self.b1

def logloss_check():
    out = []
    for tr in (half, ~half):
        te = ~tr
        s1, s3 = s2.offer_ranks(A1, c1), s2.offer_ranks(A1, c3)
        base = s2.WinnersCurse().fit(L1[tr], s1[tr], s3[tr], y[tr])
        rich = WCemp().fit(L1[tr], s1[tr], s3[tr], y[tr], E[tr])
        out.append((log_loss(y[te], p01[te]), log_loss(y[te], base.predict(L1[te], s1[te], s3[te])),
                    log_loss(y[te], rich.predict(L1[te], s1[te], s3[te], E[te]))))
    print("out-of-fold log-loss  p0: %.5f | current: %.5f | + employment: %.5f" % tuple(np.mean(out, axis=0)), flush=True)

def run(prob=0.25, K=500, pd_shift=0.0, rich=False):
    s2.ADAPT_LO, s2.ADAPT_HI = 0.50, 0.95
    rate = np.full(n, np.nan)
    for tr in (half, ~half):
        te = ~tr
        if not rich:
            pool, wc = s2.build_pool(r1[tr], A1[tr], L1[tr], prob)
        else:
            s1, s3 = s2.offer_ranks(A1[tr], c1[tr]), s2.offer_ranks(A1[tr], c3[tr])
            wc = WCemp().fit(L1[tr], s1, s3, y[tr], E[tr])
            G0, G1 = wc.Gemp(s1, s3, E[tr])
            pool = s2.Pool(emp[tr], A1[tr], c1[tr], c3[tr], G0, G1, prob)
            wc.Eq_query = E[te]
        rate[te] = s2.best_response(pool, emp[te], A1[te], L1[te], wc, K=K, pd_shift=pd_shift)["rate"]
    return rate

def formula(rate, psi, n_mc=20, seed=7):
    rng = np.random.default_rng(seed); tot = 0
    for _ in range(n_mc):
        a1 = np.where(np.isnan(c1) & (rng.random(n) < psi), BE, c1)
        a3 = np.where(np.isnan(c3) & (rng.random(n) < psi), BE, c3)
        tot += s2.replay(rate, a1, a3, bt, y).sum()
    return tot / n_mc / 1e6

def evaluate(name, rate):
    s2.ADAPT_LO, s2.ADAPT_HI = 0.50, 0.95
    row = {"variant": name, "unchanged": s2.replay_adapted(rate, r1, 0.0) / 1e6,
           "wide 25%": s2.replay_adapted(rate, r1, 0.25) / 1e6, "wide 50%": s2.replay_adapted(rate, r1, 0.5) / 1e6}
    s2.ADAPT_LO, s2.ADAPT_HI = 0.85, 0.97
    row["narrow 50%"] = s2.replay_adapted(rate, r1, 0.5) / 1e6
    s2.ADAPT_LO, s2.ADAPT_HI = 0.50, 0.95
    row["refusals end 50%"] = formula(rate, 0.5)
    row["mean"] = np.mean([v for k, v in row.items() if k != "variant"])
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row

logloss_check()
rows = []
for name, kw in [("current (K=500, 25%)", {}), ("K=1000", {"K": 1000}), ("K=2000", {"K": 2000}),
                 ("+ employment in default model", {"rich": True}), ("PD margin +0.10 log-odds", {"pd_shift": 0.10}),
                 ("assume 20%", {"prob": 0.20}), ("assume 30%", {"prob": 0.30})]:
    t0 = time.time(); rows.append(evaluate(name, run(**kw)))
print(pd.DataFrame(rows).set_index("variant").round(1).to_string())
