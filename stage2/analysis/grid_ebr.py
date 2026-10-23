"""usage: python -I grid_ebr.py <stage2_dir> <data_dir>"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[:1] + [sys.argv[2]]
import stage2_pricing as s2
from scipy.special import logit
D = s2.HERE
past = pd.read_csv(D / "PastLoans.csv")
r1 = pd.read_csv(D / "NewApplications_Lender2_Round1.csv").merge(pd.read_csv(D / "Round1_Diagnostic_22.csv"), on="id")
dm = s2.DefaultModel(past); A1, p01 = dm.score(r1); L1 = logit(np.clip(p01, 1e-6, 1 - 1e-6))
half = (r1.id.values % 2).astype(bool)
trues = [0.0, 0.25, 0.5, 0.75]
rows = []
def run(prob, K=s2.K_NEIGH):
    rate = np.full(len(r1), np.nan)
    for tr in (half, ~half):
        te = ~tr
        pool, wc = s2.build_pool(r1[tr], A1[tr], L1[tr], prob)
        rate[te] = s2.best_response(pool, r1.employment.values[te], A1[te], L1[te], wc, K=K)["rate"]
    return rate
for prob in [0.0, 0.1, 0.25, 0.35, 0.5, 0.75]:
    rate = run(prob)
    row = {"assumed": prob, "median rate": np.nanmedian(rate), "share >= 90%": np.mean(rate >= 0.9), "share < 50%": np.mean(rate < 0.5)}
    for tp in trues:
        row[f"true {tp}"] = round(s2.replay_adapted(rate, r1, tp) / 1e6, 1)
    rows.append(row); print(row, flush=True)
t = pd.DataFrame(rows).set_index("assumed")
cols = [f"true {tp}" for tp in trues]
best = t[cols].max()
t["max regret"] = (best - t[cols]).max(axis=1)
t["mean over true"] = t[cols].mean(axis=1)
print(t.round(3).to_string())
for K in [250, 1000]:
    print(f"K={K}, assumed 0.25: stationary {s2.replay_adapted(run(0.25, K), r1, 0.0)/1e6:.1f}M", flush=True)
