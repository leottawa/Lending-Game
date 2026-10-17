"""
Lending Game — Stage 1, version 5

Nouveautés v5 : (i) scénarios où les rivaux jouent EUX AUSSI l'hybride (taux élevés au lieu
de refuser) ; (ii) high testé à 30/50/75/100 % ; (iii) part de prêts effectivement remportés ;
(iv) bootstrap sur l'objectif complet (moyenne + pire cas), pas seulement sur la moyenne.

Rappel des règles : taux interdits AU-DESSUS de 100 %, donc 100 % est autorisé.

Nouveauté v4 : stratégie HYBRIDE. Au-delà de pmax, au lieu de refuser, on peut offrir
un taux élevé fixe `high` (ex. 75 %). Si les rivaux prêtent à ces profils, l'offre n'est
quasi jamais choisie (≈ refus) ; s'ils les refusent, on les capte à prix de monopole.
high = inf -> refus pur (comme en v2/v3).

Par rapport à la v2 :
  A. Grille complète : inclut s=0 et pmax=1 (la stratégie v1 « marge fixe sans refus »
     est donc en compétition directe dans l'optimisation).
  B. Sensibilité aux poids des scénarios : 4 jeux de poids (base, uniforme,
     concurrents agressifs, concurrents prudents) -> on vérifie si la stratégie
     retenue reste bonne quand les hypothèses changent (« regret »).
  C. Concurrents qui refusent eux aussi les profils risqués (pmax des rivaux).
  D. Tableau du profit par scénario pour la stratégie retenue vs la v1.
  E. Contrôle OOF vs modèle final : PD moyennes, part de refus, part d'offres.
  F. Simulation vectorisée plus rapide (le meilleur taux concurrent est précalculé).

Rappel : tous les résultats sont CONDITIONNELS aux scénarios de concurrence supposés.
"""
import glob
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ================================================================== paramètres
PAST = "PastLoans.csv"
APPS = "NewApplications_Lender2_Round1.csv"
if not glob.glob(APPS):
    APPS = glob.glob("*Applications*.csv")[0]
OUT = "submission_v5.csv"

LOAN = 10_000
PREF = 0.02
SEEDS = [0, 1]
N_BOOT = 300
W_MEAN, W_WORST = 0.75, 0.25

M_GRID = np.round(np.arange(0.0, 0.0401, 0.005), 4)
S_GRID = [0.0, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0]
PMAX_GRID = [1.0, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04]
HIGH_GRID = [np.inf, 0.30, 0.50, 0.75, 1.00]    # taux au-delà de pmax (inf = refus)

STRAT_FIXED = None        # ex. (0.02, 0.3, 0.10, 0.75) pour imposer (m, s, pmax, high)

# Scénarios : (rival A, rival B, poids de base). Rival = (modèle, marge, pmax du rival).
# Les poids sont des HYPOTHÈSES sur le comportement des autres équipes.
INF = np.inf
SCENARIOS = [
    # rival = (modèle, marge, pmax, high) ; high = inf -> le rival refuse au-delà de pmax
    (("gb", 0.00, 1.0, INF), ("gb", 0.00, 1.0, INF), 0.08),   # deux équipes naïves (break-even)
    (("lr", 0.00, 1.0, INF), ("lr", 0.00, 1.0, INF), 0.03),   # naïves + logistique
    (("gb", 0.01, 1.0, INF), ("gb", 0.01, 1.0, INF), 0.08),
    (("gb", 0.02, 1.0, INF), ("gb", 0.02, 1.0, INF), 0.10),
    (("gb", 0.03, 1.0, INF), ("gb", 0.03, 1.0, INF), 0.10),
    (("gb", 0.04, 1.0, INF), ("gb", 0.04, 1.0, INF), 0.08),
    (("gb", 0.02, 1.0, INF), ("gb", 0.04, 1.0, INF), 0.08),   # une agressive, une prudente
    (("lr", 0.02, 1.0, INF), ("gb", 0.03, 1.0, INF), 0.08),
    (("lr", 0.03, 1.0, INF), ("lr", 0.03, 1.0, INF), 0.04),
    (("gb", 0.00, 1.0, INF), ("gb", 0.04, 1.0, INF), 0.04),
    (("gb", 0.06, 1.0, INF), ("gb", 0.06, 1.0, INF), 0.03),
    # rivaux qui refusent les profils risqués
    (("gb", 0.02, 0.15, INF), ("gb", 0.02, 0.15, INF), 0.04),
    (("gb", 0.03, 0.10, INF), ("gb", 0.03, 0.10, INF), 0.04),
    (("gb", 0.02, 0.10, INF), ("gb", 0.03, 1.0, INF), 0.03),
    (("gb", 0.00, 0.20, INF), ("gb", 0.01, 1.0, INF), 0.03),
    # rivaux qui jouent eux aussi l'hybride (taux élevé au lieu du refus)
    (("gb", 0.03, 0.10, 1.00), ("gb", 0.03, 1.0, INF), 0.04),
    (("gb", 0.03, 0.10, 1.00), ("gb", 0.03, 0.10, 1.00), 0.04),
    (("gb", 0.02, 0.08, 0.75), ("gb", 0.02, 1.0, INF), 0.04),
]
W_BASE = np.array([sc[2] for sc in SCENARIOS])
assert abs(W_BASE.sum() - 1) < 1e-9, "les poids doivent sommer à 1"


def scenario_label(sc):
    f = lambda r: (f"{r[0]}{r[1]:.2f}" + ("" if r[2] >= 1 else f"/p<{r[2]:.2f}")
                   + ("" if np.isinf(r[3]) else f"->{r[3]:.0%}"))
    return f"{f(sc[0])} + {f(sc[1])}"


# jeux de poids alternatifs pour l'analyse de sensibilité
mean_m = np.array([(sc[0][1] + sc[1][1]) / 2 for sc in SCENARIOS])
refuse = np.array([any(r[2] < 1 and np.isinf(r[3]) for r in sc[:2]) for sc in SCENARIOS])
rival_hyb = np.array([any(np.isfinite(r[3]) for r in sc[:2]) for sc in SCENARIOS])
_norm = lambda w: w / w.sum()
WEIGHT_SETS = {
    "base": W_BASE,
    "uniforme": _norm(np.ones(len(SCENARIOS))),
    "rivaux agressifs": _norm(W_BASE / (0.005 + mean_m)),
    "rivaux prudents": _norm(W_BASE * (0.005 + mean_m) * np.where(refuse, 2.0, 1.0)),
}

# ================================================================== données
past = pd.read_csv(PAST)
apps = pd.read_csv(APPS)
ME = [k for k in (1, 2, 3) if f"signal{k}" in apps.columns][0]
y = past["default"].astype(int).values
n = len(y)
IDX = np.arange(n)
SCALE = 100_000 / n
EMP_CATS = sorted(past["employment"].astype(str).unique())
OTHERS = [k for k in range(3) if k != ME - 1]
print(f"Applications : {APPS} | tu es lender {ME} | {n} prêts passés, défaut {y.mean():.2%}")


def features(df, k):
    X = pd.DataFrame({
        "male": (df["sex"].astype(str).str.upper() == "M").astype(int),
        "married": df["married"].astype(int),
        "log_income": np.log1p(df["income"].clip(lower=0)),
        "topcoded": (df["income"] >= 1_000_000).astype(int),
        "signal": df[f"signal{k}"].astype(float),
    }, index=df.index)
    emp = pd.Categorical(df["employment"].astype(str), categories=EMP_CATS)
    return pd.concat([X, pd.get_dummies(emp, prefix="emp").set_index(df.index)], axis=1).astype(float)


def make_gb():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0)


def make_lr():
    return LogisticRegression(max_iter=2000)


def oof(k):
    X = features(past, k)
    p_gb, p_lr = np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p_gb[te] = make_gb().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        p_lr[te] = make_lr().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    print(f"  lender {k}: AUC GB={roc_auc_score(y, p_gb):.4f} LR={roc_auc_score(y, p_lr):.4f} | "
          f"logloss GB={log_loss(y, p_gb):.4f} LR={log_loss(y, p_lr):.4f}")
    return p_gb, p_lr


print("\n[1] Modèles de PD (out-of-fold)")
res = [oof(k) for k in (1, 2, 3)]
PD = {"gb": np.column_stack([r[0] for r in res]), "lr": np.column_stack([r[1] for r in res])}
print("  corrélation des PD GB entre lenders :\n",
      pd.DataFrame(PD["gb"], columns=["L1", "L2", "L3"]).corr().round(3).to_string())


# ================================================================== marché (vectorisé)
def price(p, m, s=0.0, pmax=1.0, high=np.inf):
    be = p / (1 - p)                                   # taux de break-even
    r = np.where(p <= pmax, be + m + s * p, high)      # au-delà de pmax : taux fixe élevé
    return np.where((r <= 1.0) & (r >= be), r, np.inf)  # jamais sous le break-even, jamais > 100 %


def make_types(seed):
    t = IDX % 3
    np.random.default_rng(seed).shuffle(t)
    return t


TYPES = [make_types(s) for s in SEEDS]

# meilleur taux effectif concurrent, pour chaque scénario et chaque tirage de types
RIVAL_BEST = []                                    # [scénario][seed] -> array (n,)
for sc in SCENARIOS:
    per_seed = []
    rival_rates = {slot: price(PD[model][:, slot], m, 0.0, pmax, high)
                   for slot, (model, m, pmax, high) in zip(OTHERS, sc[:2])}
    for t in TYPES:
        eff = [r - PREF * (t == slot) for slot, r in rival_rates.items()]
        per_seed.append(np.minimum(*eff))
    RIVAL_BEST.append(per_seed)

MY_PREF = [(t == ME - 1) * PREF for t in TYPES]


def profit_vecs(my_rates):
    """Profit par prêt (moyenne sur seeds) pour chaque scénario -> array (n_scen, n)."""
    offered = np.isfinite(my_rates)
    gain = np.where(y == 1, -LOAN, np.where(offered, my_rates, 0.0) * LOAN)
    out = np.zeros((len(SCENARIOS), n))
    for si, t in enumerate(TYPES):
        my_eff = my_rates - MY_PREF[si]
        for j in range(len(SCENARIOS)):
            win = offered & (my_eff < RIVAL_BEST[j][si])
            out[j] += np.where(win, gain, 0.0)
    return out / len(TYPES)


def win_share(my_rates):
    """Part des demandeurs que tu remportes effectivement, par scénario."""
    offered = np.isfinite(my_rates)
    out = np.zeros(len(SCENARIOS))
    for si in range(len(TYPES)):
        my_eff = my_rates - MY_PREF[si]
        for j in range(len(SCENARIOS)):
            out[j] += (offered & (my_eff < RIVAL_BEST[j][si])).mean()
    return out / len(TYPES)


def profits_by_scenario(strat, p):
    return profit_vecs(price(p, *strat)).sum(axis=1) * SCALE


def objective(prof, w):
    return W_MEAN * (prof @ w) + W_WORST * prof.min()


# ================================================================== optimisation
pd_me = PD["gb"][:, ME - 1]

if STRAT_FIXED is not None:
    BEST = tuple(STRAT_FIXED)
else:
    strats = [st for st in product(M_GRID, S_GRID, PMAX_GRID, HIGH_GRID)
              if not (st[2] == 1.0 and st[3] != np.inf)]   # high inutile sans seuil
    print(f"\n[2] Optimisation : {len(strats)} stratégies x {len(SCENARIOS)} scénarios")
    P = np.array([profits_by_scenario(st, pd_me) for st in strats])      # (n_strat, n_scen)
    offers = np.array([np.isfinite(price(pd_me, *st)).mean() for st in strats])

    R = pd.DataFrame(strats, columns=["m", "s", "pmax", "high"])
    R["offres"] = offers
    R["mean_w"] = P @ W_BASE
    R["worst"] = P.min(axis=1)
    R["objective"] = [objective(P[i], W_BASE) for i in range(len(strats))]
    order = R["objective"].sort_values(ascending=False).index

    top = R.loc[order].head(10).copy()
    for c in ["mean_w", "worst", "objective"]:
        top[c] = top[c] / 1000
    print("\nTop 10 stratégies (poids de base, k€ pour 100k demandes) :")
    print(top.round(3).to_string(index=False))

    i_best = order[0]
    BEST = tuple(R.loc[i_best, ["m", "s", "pmax", "high"]])
    v1_mask = (R.s == 0) & (R.pmax == 1.0) & np.isinf(R.high)
    i_v1 = R[v1_mask]["objective"].idxmax()
    print(f"\nRéférence v1 (marge fixe, aucun refus) : m={R.loc[i_v1, 'm']:.3f} -> "
          f"objectif {R.loc[i_v1, 'objective'] / 1000:.0f} k€, moyenne {R.loc[i_v1, 'mean_w'] / 1000:.0f} k€")

    if (BEST[0] in (M_GRID[0], M_GRID[-1]) or BEST[1] == S_GRID[-1] or BEST[2] == min(PMAX_GRID)):
        print("  /!\\ la meilleure stratégie est au bord de la grille.")
    if R.loc[i_best, "offres"] < 0.25:
        print(f"  /!\\ la stratégie retenue ne fait d'offres qu'à {R.loc[i_best, 'offres']:.0%} des demandeurs : "
              "très dépendante de l'hypothèse de concurrents 'clones'.")

    # ---------------------------------------------- B. sensibilité aux poids
    print("\n[3] Sensibilité aux poids des scénarios (k€)")
    rows = []
    for name, w in WEIGHT_SETS.items():
        objs = np.array([objective(P[i], w) for i in range(len(strats))])
        j = objs.argmax()
        rows.append({"poids": name,
                     "meilleure (m,s,pmax,high)": f"({R.m[j]:.3f}, {R.s[j]:.1f}, {R.pmax[j]:.2f}, {R.high[j]:.2f})",
                     "obj. meilleure": objs[j] / 1000,
                     "obj. retenue": objs[i_best] / 1000,
                     "regret retenue": (objs[j] - objs[i_best]) / 1000,
                     "obj. v1": objs[i_v1] / 1000})
    print(pd.DataFrame(rows).round(0).to_string(index=False))
    print("  regret = ce que tu perds en gardant la stratégie retenue si ces poids étaient les bons.")

    # ---------------------------------------------- D. profit par scénario
    print("\n[4] Profit par scénario (k€) : retenue, meilleure de chaque famille, v1")
    fam = {"refus pur": np.isinf(R.high) & (R.pmax < 1),
           "sans seuil": (R.pmax == 1.0),
           "hybride": np.isfinite(R.high)}
    best_fam = {k: R[msk]["objective"].idxmax() for k, msk in fam.items()}
    print("  Meilleure stratégie par famille (poids de base) :")
    for k, i in best_fam.items():
        print(f"    {k:<11}: m={R.m[i]:.3f} s={R.s[i]:.1f} pmax={R.pmax[i]:.2f} high={R.high[i]:.2f} "
              f"-> objectif {R.objective[i] / 1000:.0f} k€, offres {R.offres[i]:.0%}")
    tab = pd.DataFrame({"scénario": [scenario_label(sc) for sc in SCENARIOS],
                        "poids": W_BASE,
                        "retenue": P[i_best] / 1000,
                        **{k: P[i] / 1000 for k, i in best_fam.items()},
                        "v1": P[i_v1] / 1000,
                        "% gagnés (retenue)": 100 * win_share(price(pd_me, *BEST))})
    print(tab.round({c: 0 for c in tab.columns[2:]}).to_string(index=False))
    cols = [c for c in tab.columns[2:] if not c.startswith("%")]
    for lab, msk in [("HORS scénarios où les rivaux refusent", ~refuse),
                     ("UNIQUEMENT scénarios où les rivaux jouent l'hybride", rival_hyb)]:
        print(f"  Moyenne pondérée {lab} (k€) :",
              {c: round(float(tab[c][msk] @ W_BASE[msk] / W_BASE[msk].sum())) for c in cols})

    # ---------------------------------------------- bootstrap
    print("\n[5] Bootstrap sur les prêts — objectif complet (moyenne pondérée + pire scénario)")
    cands = list(order[:5]) + [i_v1]
    VS = np.stack([profit_vecs(price(pd_me, *strats[i])) * SCALE for i in cands])   # (cand, scen, n)
    rng = np.random.default_rng(42)
    boot = np.empty((N_BOOT, len(cands)))
    for b in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        prof = VS @ cnt                                                             # (cand, scen)
        boot[b] = W_MEAN * (prof @ W_BASE) + W_WORST * prof.min(axis=1)
    diff = boot - boot[:, [0]]
    for k, i in enumerate(cands):
        lo, hi = np.percentile(boot[:, k], [5, 95]) / 1000
        tag = " (v1)" if i == i_v1 else ""
        print(f"  m={R.m[i]:.3f} s={R.s[i]:.1f} pmax={R.pmax[i]:.2f} high={R.high[i]:.2f}{tag}: "
              f"objectif IC90% [{lo:.0f}, {hi:.0f}] k€"
              + ("" if k == 0 else f" | P(bat la retenue) = {(diff[:, k] > 0).mean():.0%}"))

print(f"\nStratégie retenue pour lender {ME} : m={BEST[0]:.4f}, s={BEST[1]:.2f}, pmax={BEST[2]:.2f}, high={BEST[3]:.2f}")

# ================================================================== soumission + E. contrôle OOF vs final
model = make_gb().fit(features(past, ME), y)
p_new = model.predict_proba(features(apps, ME))[:, 1]
p_past_final = model.predict_proba(features(past, ME))[:, 1]

r_new = price(p_new, *BEST)
sub = pd.DataFrame({"id": apps["id"], "rate": np.where(np.isfinite(r_new), r_new, np.nan)})
assert len(sub) == 100_000 and sub["id"].is_unique and sub["rate"].dropna().between(0, 1).all()
sub.to_csv(OUT, index=False)

print("\n[6] Contrôle OOF vs modèle final")
chk = pd.DataFrame({
    "PD moyenne": [pd_me.mean(), p_past_final.mean(), p_new.mean()],
    "part PD > pmax": [(pd_me > BEST[2]).mean(), (p_past_final > BEST[2]).mean(), (p_new > BEST[2]).mean()],
    "part d'offres": [np.isfinite(price(pd_me, *BEST)).mean(), np.isfinite(price(p_past_final, *BEST)).mean(),
                      np.isfinite(r_new).mean()],
}, index=["passé, OOF (simulation)", "passé, modèle final", "nouvelles demandes, final"])
print(chk.round(4).to_string())
print(f"  corrélation OOF vs final sur le passé : {np.corrcoef(pd_me, p_past_final)[0, 1]:.3f}")

offered = sub["rate"].notna()
print(f"\n{OUT} : {offered.mean():.1%} d'offres | taux médian {sub['rate'].median():.2%} | "
      f"min {sub['rate'].min():.2%} | max {sub['rate'].max():.2%}")
