import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import SEED, RES, load, split, features, to_codes, boot_auc_diff

df = load()
tr, va, te = split(df)
X = features(df)
y = df["churn"].values
Xtr, Xva, Xte = X.loc[tr], X.loc[va], X.loc[te]
ytr, yva, yte = y[tr], y[va], y[te]
print("tamanos:", len(tr), len(va), len(te), "| churn:", y.mean().round(3))

# ---- Revision de fuga: AUC de cada variable por separado (en entrenamiento) ----
Xc = to_codes(Xtr, Xtr)
single = {}
for c in Xc.columns:
    v = Xc[c].fillna(Xc[c].median())
    a = roc_auc_score(ytr, v)
    single[c] = max(a, 1 - a)
leak = pd.Series(single).sort_values(ascending=False)
leak.head(15).round(4).to_csv(RES / "single_feature_auc_top15.csv", header=["auc"])
print("Top AUC por variable sola:\n", leak.head(8).round(3).to_string())

# ---- Modelos ----
num = Xtr.select_dtypes(exclude="object").columns.tolist()
cat = Xtr.select_dtypes(include="object").columns.tolist()


def lr_pipe(cols_num, cols_cat, C=1.0):
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cols_num),
        ("c", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("o", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200))]), cols_cat),
    ])
    return Pipeline([("pre", pre), ("m", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])


preds = {}
# Base simple: regresion logistica con 5 variables de negocio
base_cols = ["eqpdays", "months", "mou_Mean", "rev_Mean", "change_mou"]
base = lr_pipe(base_cols, []).fit(Xtr[base_cols], ytr)
preds["Baseline: logistic regression (5 features)"] = (base.predict_proba(Xva[base_cols])[:, 1],
                                                       base.predict_proba(Xte[base_cols])[:, 1])

# Regresion logistica completa: C elegido en validacion
best = None
for C in [0.01, 0.1, 1.0]:
    m = lr_pipe(num, cat, C).fit(Xtr, ytr)
    a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    print("LR C=", C, "val AUC", round(a, 4))
    if best is None or a > best[0]:
        best = (a, m, C)
m = best[1]
preds["Logistic regression (all features)"] = (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])

# LightGBM y XGBoost: arboles y parada temprana con validacion
Ctr, Cva, Cte = to_codes(Xtr, Xtr), to_codes(Xva, Xtr), to_codes(Xte, Xtr)
lg = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                        random_state=SEED, n_jobs=-1, verbose=-1)
lg.fit(Ctr, ytr, eval_set=[(Cva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
preds["LightGBM"] = (lg.predict_proba(Cva)[:, 1], lg.predict_proba(Cte)[:, 1])
print("LightGBM arboles:", lg.best_iteration_)

xg = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=5, min_child_weight=10, subsample=0.8,
                   colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", eval_metric="auc",
                   early_stopping_rounds=100, random_state=SEED, n_jobs=-1)
xg.fit(Ctr, ytr, eval_set=[(Cva, yva)], verbose=False)
preds["XGBoost"] = (xg.predict_proba(Cva)[:, 1], xg.predict_proba(Cte)[:, 1])
print("XGBoost arboles:", xg.best_iteration + 1)


def lift_at(y_true, p, frac):
    k = int(len(p) * frac)
    top = np.argsort(-p)[:k]
    return y_true[top].mean() / y_true.mean()


rows = []
for name, (pv, pt) in preds.items():
    rows.append({"model": name, "val_auc": roc_auc_score(yva, pv), "test_auc": roc_auc_score(yte, pt),
                 "test_logloss": log_loss(yte, pt), "test_brier": brier_score_loss(yte, pt),
                 "test_lift_top10": lift_at(yte, pt, 0.10), "test_lift_top20": lift_at(yte, pt, 0.20)})
res = pd.DataFrame(rows)

# Mejora frente al baseline con intervalo de confianza al 95% (bootstrap pareado en prueba)
b = preds["Baseline: logistic regression (5 features)"][1]
lo_hi = {k: boot_auc_diff(yte, v[1], b) for k, v in preds.items() if not k.startswith("Baseline")}
res["auc_gain_vs_baseline_ci95"] = res["model"].map(
    lambda k: "" if k.startswith("Baseline") else f"[{lo_hi[k][0]:+.4f}, {lo_hi[k][1]:+.4f}]")
res.round(4).to_csv(RES / "model_comparison.csv", index=False)
print(res.round(4).to_string())

# Guardar predicciones para los pasos de calibracion y valor esperado
out_va = pd.DataFrame({"Customer_ID": df.loc[va, "Customer_ID"].values, "churn": yva})
out_te = pd.DataFrame({"Customer_ID": df.loc[te, "Customer_ID"].values, "churn": yte})
for name, (pv, pt) in preds.items():
    out_va[name] = pv
    out_te[name] = pt
out_va.to_csv(RES / "preds_val.csv", index=False)
out_te.to_csv(RES / "preds_test.csv", index=False)
print("listo")
