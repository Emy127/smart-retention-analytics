import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from common import SEED, RES, load, split, features, to_codes, boot_auc_diff

df = load()
tr, va, te = split(df)
X = features(df)
y = df["churn"].values
ytr, yva, yte = y[tr], y[va], y[te]


def new_features(A):
    A = A.copy()
    A["mou_trend_3_6"] = A["avg3mou"] / A["avg6mou"]
    A["mou_trend_3_life"] = A["avg3mou"] / A["avgmou"]
    A["mou_recent_vs_6"] = A["mou_Mean"] / A["avg6mou"]
    A["rev_trend_3_6"] = A["avg3rev"] / A["avg6rev"]
    A["rev_recent_vs_life"] = A["rev_Mean"] / A["avgrev"]
    A["overage_share"] = A["ovrrev_Mean"] / A["rev_Mean"]
    A["price_per_minute"] = A["rev_Mean"] / A["mou_Mean"]
    A["charge_vs_revenue"] = A["totmrc_Mean"] / A["rev_Mean"]
    A["failed_call_rate"] = A["drop_blk_Mean"] / A["attempt_Mean"]
    A["care_per_call"] = A["custcare_Mean"] / A["attempt_Mean"]
    A["equip_vs_tenure"] = A["eqpdays"] / (A["months"] * 30)
    A["active_subs_ratio"] = A["actvsubs"] / A["uniqsubs"]
    A["handsets_per_year"] = A["phones"] / A["months"] * 12
    return A.replace([np.inf, -np.inf], np.nan)


sets = {"original": X, "with_new_features": new_features(X)}
coded = {k: tuple(to_codes(A.loc[i], A.loc[tr]) for i in (tr, va, te)) for k, A in sets.items()}


def lgb_fit(name, params):
    Ctr, Cva, Cte = coded[name]
    m = lgb.LGBMClassifier(n_estimators=6000, random_state=SEED, n_jobs=-1, verbose=-1,
                           subsample_freq=1, **params)
    m.fit(Ctr, ytr, eval_set=[(Cva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict_proba(Cva)[:, 1], m.predict_proba(Cte)[:, 1]


current = dict(learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
               colsample_bytree=0.6, reg_lambda=5.0)
trials = []
for name in sets:
    pv, pt = lgb_fit(name, current)
    trials.append((roc_auc_score(yva, pv), name, "current", pv, pt))
    print(name, "parametros actuales, val AUC", round(trials[-1][0], 5))

# Busqueda aleatoria de parametros (solo se mira validacion)
rng = np.random.default_rng(SEED)
for t in range(25):
    params = dict(learning_rate=float(rng.choice([0.01, 0.02])), num_leaves=int(rng.choice([15, 31, 63, 127])),
                  min_child_samples=int(rng.choice([50, 100, 200, 400])), subsample=float(rng.choice([0.7, 0.85, 1.0])),
                  colsample_bytree=float(rng.choice([0.3, 0.5, 0.7])), reg_lambda=float(rng.choice([1, 5, 20])),
                  reg_alpha=float(rng.choice([0, 1, 5])))
    pv, pt = lgb_fit("with_new_features", params)
    trials.append((roc_auc_score(yva, pv), "with_new_features", str(params), pv, pt))
print("mejor val AUC tras la busqueda:", round(max(t[0] for t in trials), 5))

best = max(trials, key=lambda t: t[0])
Ctr, Cva, Cte = coded["with_new_features"]
xg = XGBClassifier(n_estimators=6000, learning_rate=0.02, max_depth=6, min_child_weight=20, subsample=0.85,
                   colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist", eval_metric="auc",
                   early_stopping_rounds=200, random_state=SEED, n_jobs=-1)
xg.fit(Ctr, ytr, eval_set=[(Cva, yva)], verbose=False)
xv, xt = xg.predict_proba(Cva)[:, 1], xg.predict_proba(Cte)[:, 1]
blend_v, blend_t = (best[3] + xv) / 2, (best[4] + xt) / 2

cur = trials[0]
cands = {"current model": (cur[3], cur[4]), "new features, same params": (trials[1][3], trials[1][4]),
         "tuned LightGBM": (best[3], best[4]), "tuned LightGBM + XGBoost average": (blend_v, blend_t)}
rows = []
for k, (pv, pt) in cands.items():
    lo, hi = boot_auc_diff(yte, pt, cur[4], n=1000)
    rows.append({"model": k, "val_auc": roc_auc_score(yva, pv), "test_auc": roc_auc_score(yte, pt),
                 "test_accuracy": accuracy_score(yte, pt > 0.5), "test_precision": precision_score(yte, pt > 0.5),
                 "test_recall": recall_score(yte, pt > 0.5),
                 "gain_vs_current_ci95": "" if k == "current model" else f"[{lo:+.4f}, {hi:+.4f}]"})
res = pd.DataFrame(rows)
res.round(4).to_csv(RES / "improvement_trials.csv", index=False)
print("mejores parametros:", best[2])
print(res.round(4).to_string(index=False))
print("exactitud de adivinar siempre la clase mayoritaria:", round(max(yte.mean(), 1 - yte.mean()), 4))
