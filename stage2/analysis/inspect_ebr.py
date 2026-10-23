"""usage: python -I inspect_ebr.py <stage2_dir> <data_dir>"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[:1] + [sys.argv[2]]
import stage2_pricing as s2
from scipy.special import logit
D = s2.HERE
past = pd.read_csv(D / "PastLoans.csv")
r1 = pd.read_csv(D / "NewApplications_Lender2_Round1.csv").merge(pd.read_csv(D / "Round1_Diagnostic_22.csv"), on="id")
dm = s2.DefaultModel(past); A1, p01 = dm.score(r1); L1 = logit(np.clip(p01, 1e-6, 1 - 1e-6))
# where do competitors refuse? by id range and by our PD
r1["idq"] = pd.qcut(r1.id, 10, labels=False)
safe = p01 < 0.03
print("offer share among applicants with PD<3%, by id decile:")
print(r1[safe].groupby("idq").agg(c1=("competing1", lambda x: x.notna().mean()), c3=("competing3", lambda x: x.notna().mean())).round(3).T.to_string())
print("offer share among PD<3% by sex / married / income quartile:")
r1["incq"] = pd.qcut(r1.income.rank(method="first"), 4, labels=False)
for col in ["sex", "married", "incq"]:
    print(r1[safe].groupby(col).agg(c1=("competing1", lambda x: x.notna().mean()), c3=("competing3", lambda x: x.notna().mean())).round(3).T.to_string())
half = (r1.id.values % 2).astype(bool)
r1["pd0"] = p01
r1["band"] = pd.cut(p01, [0, .01, .02, .04, .06, .08, .10, .15, .20, .30, 1])
for prob in [0.0, 0.25]:
    rate = np.full(len(r1), np.nan); pw = np.full(len(r1), np.nan)
    for tr in (half, ~half):
        te = ~tr
        pool, wc = s2.build_pool(r1[tr], A1[tr], L1[tr], prob)
        res = s2.best_response(pool, r1.employment.values[te], A1[te], L1[te], wc)
        rate[te], pw[te] = res["rate"], res["pwin"]
    prof = s2.replay(rate, r1.competing1.values, r1.competing3.values, r1.borrowertype.values, r1["default"].values)
    r1["rate"], r1["prof"], r1["won"] = rate, prof, prof != 0
    print(f"\n===== EBR with ADAPT_PROB={prob}: realised {prof.sum()/1e6:.2f}M on {int(r1.won.sum())} loans")
    g = r1.groupby("band", observed=True)
    print(pd.DataFrame({"n": g.size(), "med_rate": g.rate.median(), "share_rate>=0.9": g.rate.apply(lambda x: (x >= 0.9).mean()),
                        "share_rate<0.3": g.rate.apply(lambda x: (x < 0.3).mean()), "won": g.won.sum(),
                        "def_won": r1[r1.won].groupby("band", observed=True)["default"].mean(),
                        "profit_M": g.prof.sum() / 1e6}).round(3).to_string())
    rb = pd.cut(r1.rate, [0, .05, .1, .2, .3, .5, .7, .9, .95, 1.0])
    g = r1.groupby(rb, observed=True)
    print(pd.DataFrame({"n": g.size(), "won": g.won.sum(), "def_won": r1[r1.won].groupby(rb[r1.won], observed=True)["default"].mean(), "profit_M": g.prof.sum() / 1e6, "pd0": g.pd0.mean()}).round(3).to_string())
    print("most common rates:", pd.Series(rate).value_counts().head(8).to_dict())
