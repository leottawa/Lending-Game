"""Round 1 diagnosis. usage: python -I r1_diagnose.py <r1_cache.npz> <diagnostic_csv>"""
import sys, numpy as np, pandas as pd
c = np.load(sys.argv[1], allow_pickle=False); d = pd.read_csv(sys.argv[2])
pd.set_option("display.width", 250)
d["pd0"], d["pwin_pred"], d["pdwin_pred"], d["eprof_pred"] = c["pd0"], c["pwin"], c["pd_win"], c["eprofit"]*1e4
d["u2"], d["emp"] = c["u2"], c["emp"]
acc = d.accepted == 1
bands = [0, .01, .02, .04, .06, .08, .10, .15, .20, .30, .50, 1]
d["band"] = pd.cut(d.pd0, bands)

print("=== 1. default model calibration on Round 1 applicants (out of sample) ===")
print(f"mean predicted PD {d.pd0.mean():.4f} vs realized default {d['default'].mean():.4f}")
d["dec"] = pd.qcut(d.pd0, 10, labels=False)
print(d.groupby("dec").agg(pred=("pd0","mean"), real=("default","mean"), n=("id","size")).round(4).T.to_string())
from sklearn.metrics import roc_auc_score
print("AUC of our PD on Round 1 defaults:", round(roc_auc_score(d["default"], d.pd0), 4))

print("\n=== 2. by our PD band: predicted vs realized ===")
g = d.groupby("band", observed=True)
t = pd.DataFrame({
    "n": g.size(), "offers": g.own.count(),
    "win_pred": g.pwin_pred.mean(), "win_real": g.accepted.mean(),
    "pd0": g.pd0.mean(),
    "pdwin_pred": d[acc].groupby("band", observed=True).pdwin_pred.mean(),
    "def_won_real": d[acc].groupby("band", observed=True)["default"].mean(),
    "def_lost_real": d[~acc].groupby("band", observed=True)["default"].mean(),
    "rate_won": d[acc].groupby("band", observed=True).own.mean(),
    "profit_pred_M": g.eprof_pred.sum()/1e6, "profit_real_M": g.profit.sum()/1e6})
print(t.round(4).to_string())
print(f"TOTAL predicted profit {d.eprof_pred.sum()/1e6:.2f}M | realized {d.profit.sum()/1e6:.2f}M | loans won pred {d.pwin_pred.sum():.0f} real {acc.sum()}")

print("\n=== 3. by our offered rate ===")
d["rband"] = pd.cut(d.own, [0, .03, .05, .08, .12, .2, .3, .5, .9, .97, 1.0])
g = d.groupby("rband", observed=True)
print(pd.DataFrame({"n": g.size(), "won": g.accepted.sum(), "win_rate": g.accepted.mean(), "pd0": g.pd0.mean(),
      "def_won": d[acc].groupby("rband", observed=True)["default"].mean(), "profit_M": g.profit.sum()/1e6,
      "profit_per_loan": d[acc].groupby("rband", observed=True).profit.mean()}).round(4).to_string())

print("\n=== 4. competitors' behaviour by our PD band ===")
g = d.groupby("band", observed=True)
cmp = pd.DataFrame({"c1_offer": g.competing1.count()/g.size(), "c3_offer": g.competing3.count()/g.size(),
    "c1_med": g.competing1.median(), "c3_med": g.competing3.median(), "own_med": g.own.median(),
    "c1_p25": g.competing1.quantile(.25), "c3_p25": g.competing3.quantile(.25),
    "def_if_c1_offers": d[d.competing1.notna()].groupby("band", observed=True)["default"].mean(),
    "def_if_c1_refuses": d[d.competing1.isna()].groupby("band", observed=True)["default"].mean(),
    "def_if_both_refuse": d[d.competing1.isna() & d.competing3.isna()].groupby("band", observed=True)["default"].mean(),
    "n_both_refuse": d[d.competing1.isna() & d.competing3.isna()].groupby("band", observed=True).size()})
print(cmp.round(4).to_string())
both_ref = d.competing1.isna() & d.competing3.isna()
print(f"both competitors refuse: {both_ref.sum()} applicants, we won {d.accepted[both_ref].sum()}, default {d['default'][both_ref & acc].mean():.4f}, profit {d.profit[both_ref].sum()/1e6:.2f}M, mean rate {d.own[both_ref & acc].mean():.3f}")
oneoff = (d.competing1.notna() ^ d.competing3.notna())
print(f"exactly one competitor offers: {oneoff.sum()}, won {d.accepted[oneoff].sum()}, profit {d.profit[oneoff].sum()/1e6:.2f}M")
bothoff = d.competing1.notna() & d.competing3.notna()
print(f"both competitors offer: {bothoff.sum()}, won {d.accepted[bothoff].sum()}, profit {d.profit[bothoff].sum()/1e6:.2f}M")

print("\n=== 5. competitor pricing: implied markup over our PD (rate - pd0/(1-pd0)), offered loans ===")
be = d.pd0/(1-d.pd0)
for col in ["competing1", "competing3"]:
    m = (d[col] - be)
    print(col, d.assign(m=m).groupby("band", observed=True).m.median().round(4).to_dict())

print("\n=== 6. counterfactuals on Round 1 (exact replay: competitors' offers and defaults fixed) ===")
inf = np.inf
c1, c3, ty, D = d.competing1.fillna(inf).values, d.competing3.fillna(inf).values, d.borrowertype.values, d["default"].values
def replay(r):
    r = np.where(np.isnan(r), inf, r)
    eo = np.where(ty == 2, r - .02, r); e1 = np.where(ty == 1, c1 - .02, c1); e3 = np.where(ty == 3, c3 - .02, c3)
    win = (eo < e1 - 1e-9) & (eo < e3 - 1e-9) & np.isfinite(r)
    return win, np.where(win, np.where(D == 1, -1e4, r * 1e4), 0.0)
w0, p0 = replay(d.own.values)
print(f"replay of submitted offers: {p0.sum()/1e6:.3f}M (reported 62.511M), won {w0.sum()}")
own = d.own.values
print("uniform shift of all our rates:")
for s in [-.02, -.01, -.005, 0, .005, .01, .02, .03]:
    print(f"   {s:+.3f}: {replay(np.clip(own + s, 0.0001, 1.0))[1].sum()/1e6:7.2f}M")
print("scale rates below 90% by factor k (keep 97.89% quotes):")
for k in [0.9, 1.0, 1.1, 1.2, 1.3]:
    r = np.where(own < .9, own*k, own)
    print(f"   k={k}: {replay(np.clip(r, 0.0001, 1.0))[1].sum()/1e6:7.2f}M")
print("drop the ~98% quotes (no offer instead):", f"{replay(np.where(own >= .9, np.nan, own))[1].sum()/1e6:.2f}M")
print("refuse applicants with our PD above a cut-off:")
for cut in [.1, .15, .2, .25, .3, .4]:
    print(f"   PD>{cut}: {replay(np.where(d.pd0.values > cut, np.nan, own))[1].sum()/1e6:7.2f}M")
print("profit by segment of the submitted offers:")
seg = np.select([own >= .9, own >= .2], ["~98% quotes", "20-90%"], "below 20%")
print(pd.DataFrame({"seg": seg, "won": w0, "profit": p0, "D": D}).groupby("seg").agg(won=("won","sum"), profit_M=("profit", lambda x: x.sum()/1e6)).to_string())
print("\nex-post best uniform shift by PD band (upper bound, in-sample):")
for b, idx in d.groupby("band", observed=True).groups.items():
    ii = np.asarray(idx)
    best = max(((s, replay(np.where(np.arange(len(d)) == -1, 0, np.clip(own + s, 1e-4, 1.0)))[1][ii].sum()) for s in np.arange(-.03, .0501, .005)), key=lambda z: z[1])
    print(f"   {str(b):14s} n={len(ii):6d} current {p0[ii].sum()/1e6:6.2f}M  best shift {best[0]:+.3f} -> {best[1]/1e6:6.2f}M")
