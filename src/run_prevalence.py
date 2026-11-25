import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RES, FIGS, load

df = load()
pt = pd.read_csv(RES / "preds_test.csv")
y = pt["churn"].values
p = pt["LightGBM"].values
rev = pt[["Customer_ID"]].merge(df[["Customer_ID", "rev_Mean"]], on="Customer_ID")["rev_Mean"]
rev = rev.fillna(df["rev_Mean"].median()).values
value = rev * 0.30 * 12          # supuesto: margen 30 %, horizonte 12 meses
base_prev = y.mean()
n = len(y)


def scenario(prev):
    # Supuesto: solo cambia la proporcion de churn; el orden del modelo se mantiene
    odds = p / (1 - p) * (prev / (1 - prev)) / (base_prev / (1 - base_prev))
    p_adj = odds / (1 + odds)
    w = np.where(y == 1, prev / base_prev, (1 - prev) / (1 - base_prev))
    return p_adj, w


def benefit(target, w, s, c):
    return float(np.sum(w * target * (y * s * value - c)) / w.sum() * 1000)


def top_share(p_adj, w, share):
    order = np.argsort(-p_adj)
    chosen = np.cumsum(w[order]) / w.sum() <= share
    m = np.zeros(n, bool)
    m[order[chosen]] = True
    return m


S0, C0 = 0.20, 15.0
rows = []
for prev in [0.01, 0.02, 0.05, base_prev]:
    p_adj, w = scenario(prev)
    rule = p_adj * S0 * value > C0
    top10 = top_share(p_adj, w, 0.10)
    rows.append({"churn_rate_%": prev * 100,
                 "contact_everyone": benefit(np.ones(n, bool), w, S0, C0),
                 "top_10pct": benefit(top10, w, S0, C0),
                 "ev_rule": benefit(rule, w, S0, C0),
                 "ev_rule_contacted_%": float(np.sum(w * rule) / w.sum() * 100),
                 "breakeven_offer_cost_top10": float(np.sum(w * top10 * y * S0 * value) / np.sum(w * top10))})
scen = pd.DataFrame(rows)
scen.round(2).to_csv(RES / "prevalence_scenarios.csv", index=False)
print("Ganancia por 1000 clientes (exito 20 %, costo 15):")
print(scen.round(2).to_string(index=False))

# Escenario de churn mensual 2 %: que costo de oferta tiene sentido
p_adj, w = scenario(0.02)
S_grid, C_grid = [0.05, 0.10, 0.20, 0.30], [0.5, 1, 2, 5, 15]
ben = pd.DataFrame([[benefit(p_adj * s * value > c, w, s, c) for s in S_grid] for c in C_grid],
                   index=C_grid, columns=S_grid)
share = pd.DataFrame([[np.sum(w * (p_adj * s * value > c)) / w.sum() * 100 for s in S_grid] for c in C_grid],
                     index=C_grid, columns=S_grid)
ben.round(1).to_csv(RES / "prev2_net_benefit_per_1000.csv")
share.round(1).to_csv(RES / "prev2_contacted_pct.csv")
print("Churn 2 %: ganancia por 1000 (filas costo, columnas exito)\n", ben.round(0))
print("Churn 2 %: % de clientes contactados\n", share.round(1))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(ben.values, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(S_grid)))
ax.set_xticklabels([f"{int(s * 100)}%" for s in S_grid])
ax.set_yticks(range(len(C_grid)))
ax.set_yticklabels(C_grid)
for i in range(len(C_grid)):
    for j in range(len(S_grid)):
        ax.text(j, i, f"{ben.values[i, j]:,.0f}\n({share.values[i, j]:.0f}% contacted)",
                ha="center", va="center", fontsize=8)
ax.set_xlabel("Retention offer success rate (assumption)")
ax.set_ylabel("Cost per offer (assumption)")
ax.set_title("Scenario: 2% churn per window\nNet benefit per 1,000 customers")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(FIGS / "prevalence_2pct.png", dpi=150)
plt.close()
print("listo")
