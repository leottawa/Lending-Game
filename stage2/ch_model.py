"""Predict competitors' Round 2 reactions with a cognitive-hierarchy (level-k) model, then best-respond.
usage: python -I ch_model.py <stage2_dir> <data_dir> [--final]
Level 0 repeats its Round 1 behaviour. Level k >= 1 best-responds to a market of levels 0..k-1 (in the CH
proportions) in the high-rate segment: a Round 1 refusal or quote >= 50% becomes its level-k quote.
Level shares: Poisson(TAU), Camerer, Ho & Chong (2004); level 3 holds '3 or more'."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); FINAL = "--final" in sys.argv; sys.argv = sys.argv[:1] + [sys.argv[2]]
import stage2_pricing as s2
from scipy.special import logit, expit
from scipy.stats import poisson

TAU, LEVELS, HIGH, REPS = 1.5, 4, 0.50, 8

def ch_weights(tau=TAU, levels=LEVELS):
    f = poisson.pmf(np.arange(levels), tau); f[-1] += 1 - f.sum(); return f

def pool_from_offers(emp, A, C1, C3, G0, G1):
    p = s2.Pool.__new__(s2.Pool)
    T = s2.thresholds(C1, C3)
    kidx = np.searchsorted(s2.RGRID, T, side="left").reshape(len(A), -1).astype(np.int16)
    code = pd.Categorical(emp, categories=s2.EMP).codes
    order = np.lexsort((A, code))
    p.A, p.code, p.k, p.G0, p.G1 = A[order], code[order], kidx[order], G0[order], G1[order]
    p.start = np.searchsorted(p.code, np.arange(len(s2.EMP)), side="left")
    p.end = np.searchsorted(p.code, np.arange(len(s2.EMP)), side="right")
    return p

def mixed_offers(c, quotes, probs, rng, reps):
    """Replicates of competitor offers: a high Round 1 response becomes the level-k quote w.p. probs[k]."""
    C = np.repeat(c[:, None], reps, axis=1)
    high = (~np.isfinite(c)) | (c >= HIGH)
    cum = np.cumsum(probs)
    lvl = np.searchsorted(cum, rng.random(C.shape) * cum[-1], side="right")
    for k in range(1, len(probs)):
        m = (lvl == k) & high[:, None]
        C[m] = np.broadcast_to(quotes[k][:, None], C.shape)[m]
    return C

def br(pool, emp, A, L, wc, min_rate=0.0, K=s2.K_NEIGH, chunk=400):
    """s2.best_response restricted to rates >= min_rate; NaN when no positive expected profit."""
    n, G = len(A), len(s2.RGRID)
    idx_all = pool.neighbours(emp, A, K)
    F, Lz = wc.F(L), wc.Lz(L)
    ok = s2.RGRID >= min_rate
    rate, eprof = np.empty(n), np.empty(n)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk)); idx = idx_all[sl]; c = idx.shape[0]
        P = expit(F[sl, None] + pool.G0[idx] + Lz[sl, None] * pool.G1[idx])
        k = pool.k[idx].astype(np.int64)
        flat = (np.arange(c)[:, None, None] * (G + 1) + k).ravel()
        hn = np.bincount(flat, minlength=c * (G + 1)).reshape(c, G + 1)
        hd = np.bincount(flat, weights=np.broadcast_to(P[:, :, None], k.shape).ravel(), minlength=c * (G + 1)).reshape(c, G + 1)
        tot = k.shape[1] * k.shape[2]
        Nw = np.cumsum(hn[:, ::-1], axis=1)[:, ::-1][:, 1:] / tot
        Dw = np.cumsum(hd[:, ::-1], axis=1)[:, ::-1][:, 1:] / tot
        Pi = np.where(ok[None, :], s2.RGRID[None, :] * Nw - (1 + s2.RGRID[None, :]) * Dw, -np.inf)
        g = np.argmax(Pi, axis=1); rr = np.arange(c)
        rate[sl], eprof[sl] = s2.RGRID[g], Pi[rr, g]
    return np.where(eprof > 0, rate, np.nan), eprof

t0 = time.time()
D = s2.HERE
past = pd.read_csv(D / "PastLoans.csv")
r1 = pd.read_csv(D / "NewApplications_Lender2_Round1.csv").merge(pd.read_csv(D / "Round1_Diagnostic_22.csv"), on="id")
dm = s2.DefaultModel(past); A1, p01 = dm.score(r1); L1 = logit(np.clip(p01, 1e-6, 1 - 1e-6))
n, y, emp = len(r1), r1["default"].values, r1.employment.values
c1 = np.where(r1.competing1.isna(), np.inf, r1.competing1.values)
c3 = np.where(r1.competing3.isna(), np.inf, r1.competing3.values)
f = ch_weights()
print("CH level shares (0..3+):", f.round(3), flush=True)

# ---- 1. predicted level-k quotes of competitors (full Round 1 as their environment)
s1r, s3r = s2.offer_ranks(A1, r1.competing1.values), s2.offer_ranks(A1, r1.competing3.values)
wc_all = s2.WinnersCurse().fit(L1, s1r, s3r, y); G0a, G1a = wc_all.G(s1r, s3r)
rng = np.random.default_rng(11)
quotes = [None]
for k in range(1, LEVELS):
    probs = f[:k] / f[:k].sum()
    if k == 1:
        P = pool_from_offers(emp, A1, c1[:, None], c3[:, None], G0a, G1a)
    else:
        P = pool_from_offers(emp, A1, mixed_offers(c1, quotes, probs, rng, REPS), mixed_offers(c3, quotes, probs, rng, REPS), G0a, G1a)
    q, _ = br(P, emp, A1, L1, wc_all, min_rate=HIGH)
    quotes.append(np.where(np.isnan(q), np.inf, q))
    fin = np.isfinite(quotes[k])
    print(f"level {k}: quotes to {fin.mean():.1%} of applicants, median {np.median(quotes[k][fin]):.4f}, "
          f"p10 {np.quantile(quotes[k][fin], .1):.4f}, p90 {np.quantile(quotes[k][fin], .9):.4f} ({time.time()-t0:.0f}s)", flush=True)

def ch_pool(sub, seed):
    """Pool for our best response: competitors drawn from the CH mixture (levels 0..3)."""
    r = np.random.default_rng(seed)
    s1, s3 = s2.offer_ranks(A1[sub], r1.competing1.values[sub]), s2.offer_ranks(A1[sub], r1.competing3.values[sub])
    wc = s2.WinnersCurse().fit(L1[sub], s1, s3, y[sub]); G0, G1 = wc.G(s1, s3)
    qs = [None] + [q[sub] for q in quotes[1:]]
    return pool_from_offers(emp[sub], A1[sub], mixed_offers(c1[sub], qs, f, r, REPS), mixed_offers(c3[sub], qs, f, r, REPS), G0, G1), wc

if FINAL:
    new = pd.read_csv(s2.NEW_FILE)
    A2, p02 = dm.score(new); L2 = logit(np.clip(p02, 1e-6, 1 - 1e-6))
    pool, wc = ch_pool(np.ones(n, bool), 5)
    rate, eprof = br(pool, new.employment.values, A2, L2, wc)
    rate = np.round(rate, 4)
    out = s2.HERE / "Lender2_Round2_offers.csv"
    pd.DataFrame({"id": new["id"], "rate": rate}).to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out}: {np.sum(~np.isnan(rate))} offers, median {np.nanmedian(rate):.4f}, expected profit "
          f"{np.nansum(np.where(np.isnan(rate), 0, eprof)) * s2.LOAN / 1e6:.2f}M under the CH prediction")
    sys.exit()

# ---- 2. cross-fitted evaluation on Round 1
half = (r1.id.values % 2).astype(bool)
def crossfit(make):
    rate = np.full(n, np.nan)
    for tr in (half, ~half):
        te = ~tr
        pool, wc = make(tr)
        rate[te] = br(pool, emp[te], A1[te], L1[te], wc)[0]
    return rate
pol = {"Stage 1 submitted": r1.own.values.copy(),
       "uniform 25% (current file)": crossfit(lambda tr: s2.build_pool(r1[tr], A1[tr], L1[tr], 0.25)),
       "CH best response": crossfit(lambda tr: ch_pool(tr, 3))}
BE = p01 / (1 - p01) + 0.03
bt = r1.borrowertype.values
def mc(rate, gen, n_mc=20, seed=99):
    r = np.random.default_rng(seed); tot = 0.0
    for _ in range(n_mc):
        a1, a3 = gen(r)
        tot += s2.replay(rate, np.where(np.isfinite(a1), a1, np.nan), np.where(np.isfinite(a3), a3, np.nan), bt, y).sum()
    return tot / n_mc / 1e6
def unif(prob, lo, hi):
    def g(r):
        out = []
        for c in (c1, c3):
            m = r.random(n) < prob
            out.append(np.where(m & ((c >= HIGH) | ~np.isfinite(c)), r.uniform(lo, hi, n), c))
        return out
    return g
def refusals_end(psi):
    return lambda r: [np.where(~np.isfinite(c) & (r.random(n) < psi), BE, c) for c in (c1, c3)]
def ch_world(r):
    return [mixed_offers(c, quotes, f, r, 1)[:, 0] for c in (c1, c3)]
scen = {"unchanged": lambda r: [c1, c3], "CH prediction": ch_world, "wide 25%": unif(.25, .5, .95),
        "wide 50%": unif(.5, .5, .95), "narrow 50%": unif(.5, .85, .97), "refusals end 50%": refusals_end(.5)}
rows = []
for name, rate in pol.items():
    row = {"policy": name, "median rate": np.nanmedian(rate)}
    for sname, gen in scen.items():
        row[sname] = mc(rate, gen, 1 if sname == "unchanged" else 20)
    rows.append(row); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
t = pd.DataFrame(rows).set_index("policy")
t["mean of scenarios"] = t[list(scen)].mean(axis=1)
print(t.round(1).to_string())
print(f"total {time.time()-t0:.0f}s")
