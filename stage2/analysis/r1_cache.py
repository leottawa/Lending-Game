"""Re-run the Stage 1 pipeline on Round 1 applicants and cache per-applicant model quantities.
usage: python -I r1_cache.py <repo_dir> <round1_data_dir> <diagnostic_csv> <out_npz>"""
import sys, time
import numpy as np, pandas as pd
repo, r1dir, diag_csv, out = sys.argv[1:5]
sys.path.insert(0, repo)
import stage1_pricing as s1
from scipy.special import expit
t0 = time.time()
past = pd.read_csv(f"{r1dir}/PastLoans.csv")
new = pd.read_csv(f"{r1dir}/NewApplications_Lender2_Round1.csv")
diag = pd.read_csv(diag_csv)
assert (new.id.values == diag.id.values).all()
ranker = s1.SignalRanker().fit(past)
U = np.column_stack([ranker.rank(past.employment.values, past[s].values) for s in s1.SIGS])
truth = s1.Truth(past, U)
u2 = ranker.rank(new.employment.values, new.signal2.values)
A, h2 = truth.A(new), truth.h(u2)
w = s1.signal_weights(new.employment.values, u2, truth.q, s1.RHO)
comp = s1.competitor_logits(new, truth, s1.fit_competitor_models(past), ranker)
print(f"model + competitor logits: {time.time()-t0:.0f}s")
comp_idx = {k: s1.lgrid_index(v) for k, v in comp.items()}
S_tab = s1.survival_table(s1.PRICING, s1.CUTOFFS, s1.SIGMA)
res = s1.best_rates(A, h2, w, comp_idx, s1.MODEL_TYPES, S_tab, truth, pd_shift=s1.PD_SHIFT)
print(f"best_rates: {time.time()-t0:.0f}s")
offer = res["profit"] > 0
rate = np.where(offer, np.round(res["rate"] - 0.0001, 4), np.nan)
own = diag.own.values
same = (np.isnan(rate) & np.isnan(own)) | (np.abs(rate - own) < 5e-5)
print(f"reproduces submitted rates: {same.mean():.5f}  (max abs diff {np.nanmax(np.abs(rate-own)):.6f})")
hm = np.r_[0.0, truth.h(s1.UMID)]
H = hm[:, None] + hm[None, :]
pd0 = np.empty(len(A))
for s in range(0, len(A), 2000):
    sl = slice(s, s + 2000)
    P = expit(A[sl, None, None] + h2[sl, None, None] + H[None])
    pd0[sl] = (w[sl][:, :, None] * w[sl][:, None, :] * P).sum(axis=(1, 2))
np.savez_compressed(out, A=A, h2=h2, u2=u2, w=w.astype(np.float32), pd0=pd0, rate=rate,
                    pwin=res["pwin"], pd_win=res["pd_win"], eprofit=res["profit"], pd_shifted=res["pd"],
                    **{f"comp_{k}": v.astype(np.float32) for k, v in comp.items()},
                    emp=new.employment.values.astype(str), hm=hm, q=np.array([truth.q[e] for e in s1.EMP]))
print(f"saved {out}: {time.time()-t0:.0f}s")
