"""usage: python -I grid2.py <stage2_dir> <data_dir>"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[:1] + [sys.argv[2]]
import stage2_pricing as s2
from scipy.special import logit
D = s2.HERE
past = pd.read_csv(D / "PastLoans.csv")
r1 = pd.read_csv(D / "NewApplications_Lender2_Round1.csv").merge(pd.read_csv(D / "Round1_Diagnostic_22.csv"), on="id")
dm = s2.DefaultModel(past); A1, p01 = dm.score(r1); L1 = logit(np.clip(p01, 1e-6, 1 - 1e-6))
half = (r1.id.values % 2).astype(bool)
pol = {"Stage 1 submitted": r1.own.values.copy()}
for prob in [0.25, 0.3, 0.35]:
    rate = np.full(len(r1), np.nan)
    for tr in (half, ~half):
        te = ~tr
        pool, wc = s2.build_pool(r1[tr], A1[tr], L1[tr], prob)
        rate[te] = s2.best_response(pool, r1.employment.values[te], A1[te], L1[te], wc)["rate"]
    pol[f"Stage 2, assumed {prob}"] = rate
c1 = r1.competing1.values; c3 = r1.competing3.values; bt = r1.borrowertype.values; Dd = r1["default"].values
def truth_formula(rate, psi, n_mc=20, seed=7):
    """refusals become offers at break-even + 3 points with probability psi"""
    rng = np.random.default_rng(seed); be = p01 / (1 - p01) + 0.03; tot = 0
    for _ in range(n_mc):
        a1 = np.where(np.isnan(c1) & (rng.random(len(c1)) < psi), be, c1)
        a3 = np.where(np.isnan(c3) & (rng.random(len(c3)) < psi), be, c3)
        tot += s2.replay(rate, a1, a3, bt, Dd).sum()
    return tot / n_mc / 1e6
rows = []
for name, rate in pol.items():
    row = {"policy": name}
    s2.ADAPT_LO, s2.ADAPT_HI = 0.50, 0.95
    for tp in [0.0, 0.25, 0.5]:
        row[f"U[50,95] {tp}"] = round(s2.replay_adapted(rate, r1, tp) / 1e6, 1)
    s2.ADAPT_LO, s2.ADAPT_HI = 0.85, 0.97
    for tp in [0.25, 0.5]:
        row[f"U[85,97] {tp}"] = round(s2.replay_adapted(rate, r1, tp) / 1e6, 1)
    for psi in [0.25, 0.5]:
        row[f"refusals->formula {psi}"] = round(truth_formula(rate, psi), 1)
    rows.append(row); print(row, flush=True)
print(pd.DataFrame(rows).set_index("policy").to_string())
