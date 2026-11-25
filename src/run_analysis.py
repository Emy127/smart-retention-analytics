import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from common import SEED, RES, FIGS, load, split, features, to_codes

df = load()
tr, va, te = split(df)
pv = pd.read_csv(RES / "preds_val.csv")
pt = pd.read_csv(RES / "preds_test.csv")
LG, BASE = "LightGBM", "Baseline: logistic regression (5 features)"
yva, yte = pv["churn"].values, pt["churn"].values
rev_te = pt[["Customer_ID"]].merge(df[["Customer_ID", "rev_Mean"]], on="Customer_ID")["rev_Mean"].fillna(df.loc[tr, "rev_Mean"].median()).values

# ---------- 1. EDA: tasa de churn por decil de las variables con mas senal ----------
d = df.loc[tr]
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
for a, (col, lab) in zip(ax.ravel(), [("eqpdays", "Equipment age (days)"),
                                      ("hnd_price", "Handset price"),
                                      ("totmrc_Mean", "Monthly recurring charge"),
                                      ("mou_Mean", "Monthly minutes of use")]):
    q = pd.qcut(d[col].rank(method="first"), 10, labels=False)
    rate = d.groupby(q)["churn"].mean() * 100
    a.plot(range(1, 11), rate.values, marker="o")
    a.axhline(d["churn"].mean() * 100, color="gray", linestyle="--", label="Average")
    a.set_title(f"Churn rate by {lab} decile")
    a.set_xlabel("Decile (1 = lowest)")
    a.set_ylabel("Churn rate (%)")
ax[0, 0].legend()
plt.tight_layout()
plt.savefig(FIGS / "eda_drivers.png", dpi=150)
plt.close()

# ---------- 2. Calibracion (isotonica ajustada en validacion, evaluada en prueba) ----------
iso = IsotonicRegression(out_of_bounds="clip").fit(pv[LG], yva)
cal_te = iso.predict(pt[LG])
cal_rows = [{"version": "LightGBM raw", "brier": brier_score_loss(yte, pt[LG]), "logloss": log_loss(yte, pt[LG])},
            {"version": "LightGBM + isotonic calibration",
             "brier": brier_score_loss(yte, cal_te), "logloss": log_loss(yte, np.clip(cal_te, 1e-6, 1 - 1e-6))}]
pd.DataFrame(cal_rows).round(4).to_csv(RES / "calibration.csv", index=False)
print(pd.DataFrame(cal_rows).round(4))
fig, ax = plt.subplots(figsize=(5.5, 5))
for lab, p in [("Raw", pt[LG].values), ("Calibrated", cal_te)]:
    fx, fy = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    ax.plot(fx, fy, marker="o", label=lab)
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("Predicted churn probability")
ax.set_ylabel("Observed churn rate")
ax.set_title("Calibration (test set)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "calibration.png", dpi=150)
plt.close()

# ---------- 3. Curva de ganancia ----------
fig, ax = plt.subplots(figsize=(6, 5))
n = len(yte)
for lab, p in [("LightGBM", pt[LG].values), ("Baseline (5 features)", pt[BASE].values)]:
    order = np.argsort(-p)
    gain = np.cumsum(yte[order]) / yte.sum() * 100
    ax.plot(np.arange(1, n + 1) / n * 100, gain, label=lab)
ax.plot([0, 100], [0, 100], "--", color="gray", label="Random")
ax.set_xlabel("% of customers contacted (highest risk first)")
ax.set_ylabel("% of churners reached")
ax.set_title("Cumulative gain (test set)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "gain_curve.png", dpi=150)
plt.close()

# ---------- 4. SHAP del LightGBM (mismo ajuste que run_models.py) ----------
X = features(df)
Ctr, Cva, Cte = to_codes(X.loc[tr], X.loc[tr]), to_codes(X.loc[va], X.loc[tr]), to_codes(X.loc[te], X.loc[tr])
lg = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                        random_state=SEED, n_jobs=-1, verbose=-1)
lg.fit(Ctr, df.loc[tr, "churn"], eval_set=[(Cva, df.loc[va, "churn"])], eval_metric="auc",
       callbacks=[lgb.early_stopping(100, verbose=False)])
samp = Cte.sample(8000, random_state=SEED)
sv = shap.TreeExplainer(lg).shap_values(samp)
if isinstance(sv, list):
    sv = sv[1]
if sv.ndim == 3:
    sv = sv[:, :, 1]
imp = pd.Series(np.abs(sv).mean(axis=0), index=samp.columns).sort_values(ascending=False)
imp.head(15).round(4).to_csv(RES / "shap_top15.csv", header=["mean_abs_shap"])
print(imp.head(10).round(4))
plt.figure(figsize=(7, 5))
imp.head(12)[::-1].plot.barh(color="#4c78a8")
plt.xlabel("Mean |SHAP value| (impact on churn score)")
plt.title("What drives the churn score")
plt.tight_layout()
plt.savefig(FIGS / "shap_bar.png", dpi=150)
plt.close()

# ---------- 5. Valor esperado y punto de corte ----------
# SUPUESTOS (no medidos en los datos): margen, horizonte, costo de oferta y exito de retencion.
MARGIN, HORIZON = 0.30, 12


def policy_benefit(y, p_rank, value, target, s, c):
    return float(np.sum(y[target] * s * value[target]) - c * target.sum())


value = rev_te * MARGIN * HORIZON


def evaluate(s, c):
    thr_rule = (pt[LG].values * s * value - c) > 0
    k20 = int(0.2 * n)
    top20_lg = np.zeros(n, bool); top20_lg[np.argsort(-pt[LG].values)[:k20]] = True
    top20_bs = np.zeros(n, bool); top20_bs[np.argsort(-pt[BASE].values)[:k20]] = True
    everyone = np.ones(n, bool)
    pol = {"Contact nobody": np.zeros(n, bool), "Contact everyone": everyone,
           "Top 20% by baseline model": top20_bs, "Top 20% by LightGBM": top20_lg,
           "Expected-value rule (LightGBM)": thr_rule}
    return {k: policy_benefit(yte, None, value, m, s, c) / n * 1000 for k, m in pol.items()}, pol


S0, C0 = 0.20, 15.0
base_res, pols = evaluate(S0, C0)
pol_df = pd.DataFrame({"policy": list(base_res), "net_benefit_per_1000_customers": list(base_res.values()),
                       "customers_contacted_%": [pols[k].mean() * 100 for k in base_res]})
pol_df.round(1).to_csv(RES / "policy_comparison_base_case.csv", index=False)
print("Caso base: s=%.2f, costo=%.0f, margen=%.2f, horizonte=%d meses" % (S0, C0, MARGIN, HORIZON))
print(pol_df.round(1).to_string(index=False))

# Intervalo de confianza (bootstrap) de la ganancia de la regla de valor esperado vs contactar a todos
rng = np.random.default_rng(SEED)
rule = pols["Expected-value rule (LightGBM)"]
bs20 = pols["Top 20% by baseline model"]
diff_all, diff_base = [], []
for _ in range(1000):
    i = rng.integers(0, n, n)
    f = lambda m: (np.sum(yte[i] * S0 * value[i] * m[i]) - C0 * m[i].sum()) / n * 1000
    diff_all.append(f(rule) - f(pols["Contact everyone"]))
    diff_base.append(f(rule) - f(bs20))
ci = {"vs_contact_everyone": np.percentile(diff_all, [2.5, 97.5]),
      "vs_baseline_top20": np.percentile(diff_base, [2.5, 97.5])}
pd.DataFrame({k: [f"[{v[0]:.0f}, {v[1]:.0f}]"] for k, v in ci.items()}).to_csv(RES / "ev_rule_gain_ci95.csv", index=False)
print("IC95 ganancia por 1000 clientes de la regla:", {k: np.round(v, 0) for k, v in ci.items()})

# Sensibilidad: ganancia de la regla frente a "contactar a todos" segun exito y costo
S_grid, C_grid = [0.05, 0.10, 0.20, 0.30], [5, 10, 15, 25, 40]
sens = pd.DataFrame(index=C_grid, columns=S_grid, dtype=float)
for c in C_grid:
    for s in S_grid:
        r, _ = evaluate(s, c)
        sens.loc[c, s] = r["Expected-value rule (LightGBM)"]
sens.round(0).to_csv(RES / "sensitivity_net_benefit_per_1000.csv")
print(sens.round(0))
fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(sens.values.astype(float), cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(S_grid)))
ax.set_xticklabels([f"{int(s * 100)}%" for s in S_grid])
ax.set_yticks(range(len(C_grid)))
ax.set_yticklabels(C_grid)
for i in range(len(C_grid)):
    for j in range(len(S_grid)):
        ax.text(j, i, f"{sens.values[i, j]:,.0f}", ha="center", va="center", fontsize=9)
ax.set_xlabel("Retention offer success rate (assumption)")
ax.set_ylabel("Cost per offer (assumption)")
ax.set_title("Net benefit per 1,000 customers\n(expected-value rule, test set)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(FIGS / "ev_sensitivity.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(pol_df["policy"][::-1], pol_df["net_benefit_per_1000_customers"][::-1], color="#54a24b")
ax.set_xlabel("Net benefit per 1,000 customers (monetary units, assumptions in README)")
ax.set_title("Base case: success 20%, offer cost 15")
plt.tight_layout()
plt.savefig(FIGS / "policy_comparison.png", dpi=150)
plt.close()
print("listo")
