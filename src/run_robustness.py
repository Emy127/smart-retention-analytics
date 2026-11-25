import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED, RES, load, features, to_codes

df = load()
X = features(df)
y = df["churn"].values
SENSITIVE = ["ethnic", "income", "marital"]
BASE_COLS = ["eqpdays", "months", "mou_Mean", "rev_Mean", "change_mou"]
S0, C0, MARGIN, HORIZON = 0.20, 15.0, 0.30, 12      # mismos supuestos del caso base


def split_seed(seed):
    tr, tmp = train_test_split(df.index, test_size=0.4, random_state=seed, stratify=df["churn"])
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=df.loc[tmp, "churn"])
    return tr, va, te


def fit_lgb(cols, tr, va, te):
    A = X[cols]
    Ctr, Cva, Cte = (to_codes(A.loc[i], A.loc[tr]) for i in (tr, va, te))
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                           random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Ctr, df.loc[tr, "churn"], eval_set=[(Cva, df.loc[va, "churn"])], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(Cte)[:, 1]


rows, group_rows = [], []
for seed in [42, 1, 2, 3, 4]:
    tr, va, te = split_seed(seed)
    yte = df.loc[te, "churn"].values
    value = df.loc[te, "rev_Mean"].fillna(df.loc[tr, "rev_Mean"].median()).values * MARGIN * HORIZON
    n = len(yte)

    base = Pipeline([("pre", ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")),
                                                               ("s", StandardScaler())]), BASE_COLS)])),
                     ("m", LogisticRegression(max_iter=2000, random_state=SEED))])
    p_base = base.fit(X.loc[tr, BASE_COLS], df.loc[tr, "churn"]).predict_proba(X.loc[te, BASE_COLS])[:, 1]
    p_full = fit_lgb(list(X.columns), tr, va, te)
    p_fair = fit_lgb([c for c in X.columns if c not in SENSITIVE], tr, va, te)

    gain = lambda m: float(np.sum(yte[m] * S0 * value[m]) - C0 * m.sum()) / n * 1000
    rule_full = p_full * S0 * value > C0
    rule_fair = p_fair * S0 * value > C0
    top20_base = np.zeros(n, bool)
    top20_base[np.argsort(-p_base)[:int(0.2 * n)]] = True
    everyone = np.ones(n, bool)
    rows.append({"seed": seed,
                 "auc_baseline": roc_auc_score(yte, p_base),
                 "auc_lightgbm": roc_auc_score(yte, p_full),
                 "auc_without_sensitive": roc_auc_score(yte, p_fair),
                 "ev_rule": gain(rule_full), "ev_rule_without_sensitive": gain(rule_fair),
                 "contact_everyone": gain(everyone),
                 "rule_minus_everyone": gain(rule_full) - gain(everyone),
                 "rule_minus_baseline_top20": gain(rule_full) - gain(top20_base),
                 "same_decision_%": float(np.mean(rule_full == rule_fair) * 100)})
    if seed == SEED:
        grp = df.loc[te, "ethnic"].fillna("missing").values
        for g in pd.Series(grp).value_counts().loc[lambda s: s >= 500].index:
            m = grp == g
            group_rows.append({"ethnic_code": g, "customers": int(m.sum()), "churn_%": yte[m].mean() * 100,
                               "contacted_%_full": rule_full[m].mean() * 100,
                               "contacted_%_without_sensitive": rule_fair[m].mean() * 100,
                               "mean_value": value[m].mean()})
    print("semilla", seed, "lista")

res = pd.DataFrame(rows)
res.round(4).to_csv(RES / "robustness_by_seed.csv", index=False)
summary = res.drop(columns="seed").agg(["mean", "std", "min", "max"]).T
summary.round(4).to_csv(RES / "robustness_summary.csv")
groups = pd.DataFrame(group_rows)
groups.round(2).to_csv(RES / "contact_rate_by_ethnic_code.csv", index=False)
print(summary.round(4).to_string())
print(groups.round(2).to_string(index=False))
print("listo")
