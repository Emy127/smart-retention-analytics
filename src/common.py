import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA, RES, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"
for p in (RES, FIGS):
    p.mkdir(parents=True, exist_ok=True)


def load():
    client = pd.read_csv(DATA / "Client.csv")
    record = pd.read_csv(DATA / "Record.csv")
    return record.merge(client, on="Customer_ID", how="inner")


def split(df):
    # 60/20/20 estratificado: entrenamiento, validacion (elegir todo) y prueba (se evalua una vez)
    y = df["churn"]
    tr, tmp = train_test_split(df.index, test_size=0.4, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=y.loc[tmp])
    return tr, va, te


def features(df):
    return df.drop(columns=["churn", "Customer_ID"])


def to_codes(X, ref):
    # Categorias a enteros usando solo las categorias vistas en entrenamiento
    X = X.copy()
    for c in X.select_dtypes(include="object").columns:
        cats = {v: i for i, v in enumerate(sorted(ref[c].dropna().unique()))}
        X[c] = X[c].map(cats)
    return X


def boot_auc_diff(y, p_a, p_b, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return np.percentile(diffs, [2.5, 97.5])
