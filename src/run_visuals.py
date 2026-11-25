import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from common import RES, FIGS, load, split

VIS = FIGS / "visual"
VIS.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY, NAVY = "#739DC2", "#FFA24C", "#C9CED6", "#142D4C"
df = load()
tr, va, te = split(df)
d = df.loc[tr]


def clean_axes(a):
    a.spines[["top", "right"]].set_visible(False)


# 1. Quien se queda y quien se va, por variable (entrenamiento)
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
for a, (col, title, lo, hi) in zip(ax.ravel(), [
        ("eqpdays", "Equipment age (days)", 0, 1200), ("months", "Months in service", 6, 48),
        ("rev_Mean", "Average monthly revenue", 0, 150), ("mou_Mean", "Monthly minutes of use", 0, 2000)]):
    bins = np.linspace(lo, hi, 22)
    v = [d.loc[(d.churn == k) & d[col].between(lo, hi), col] for k in (1, 0)]
    a.hist(v, bins=bins, stacked=True, color=[ORANGE, BLUE], edgecolor="white")
    a.set_title(title, fontsize=13, color=NAVY)
    a.set_ylabel("Customers")
    a.set_xlabel(f"Shown range: {lo} to {hi}", fontsize=9, color="gray")
    clean_axes(a)
fig.legend(handles=[Patch(color=BLUE, label="Stayed"), Patch(color=ORANGE, label="Churned")],
           loc="upper center", ncol=2, frameon=False, fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(VIS / "1_customers_stayed_vs_churned.png", dpi=200)
plt.close()

# 2. Variables que mas usa el modelo, agrupadas por tema (SHAP)
names = {"avgqty": ("Lifetime average calls", "Usage"), "eqpdays": ("Equipment age", "Device"), "hnd_price": ("Handset price", "Device"),
         "phones": ("Handsets issued", "Device"), "models": ("Models issued", "Device"),
         "refurb_new": ("New or refurbished handset", "Device"), "hnd_webcap": ("Handset web capability", "Device"),
         "months": ("Months in service", "Relationship"), "uniqsubs": ("Subscribers in household", "Relationship"),
         "actvsubs": ("Active subscribers", "Relationship"),
         "change_mou": ("Change in minutes of use", "Usage"), "mou_Mean": ("Monthly minutes", "Usage"),
         "avgmou": ("Lifetime average minutes", "Usage"), "avg6mou": ("6-month average minutes", "Usage"),
         "avg3mou": ("3-month average minutes", "Usage"), "mou_cvce_Mean": ("Completed call minutes", "Usage"),
         "totmrc_Mean": ("Monthly recurring charge", "Billing"), "rev_Mean": ("Monthly revenue", "Billing"),
         "avgrev": ("Lifetime average revenue", "Billing"), "totrev": ("Total revenue", "Billing"),
         "change_rev": ("Change in revenue", "Billing"), "ovrmou_Mean": ("Overage minutes", "Billing"),
         "drop_vce_Mean": ("Dropped voice calls", "Network quality"),
         "lor": ("Length of residence", "Customer profile"), "area": ("Geographic area", "Customer profile"),
         "crclscod": ("Credit class", "Customer profile"), "ethnic": ("Ethnicity code", "Customer profile"),
         "income": ("Estimated income", "Customer profile")}
palette = {"Device": "#E4572E", "Relationship": "#F3A712", "Usage": "#4C78A8", "Billing": "#54A24B",
           "Network quality": "#B279A2", "Customer profile": "#9AA0A6", "Other": GRAY}
imp = pd.read_csv(RES / "shap_top15.csv", index_col=0)["mean_abs_shap"].head(12)[::-1]
labels = [names.get(c, (c, "Other"))[0] for c in imp.index]
groups = [names.get(c, (c, "Other"))[1] for c in imp.index]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(labels, imp.values, color=[palette[g] for g in groups])
ax.set_xlabel("Influence on the churn score (mean |SHAP|)")
ax.legend(handles=[Patch(color=palette[g], label=g) for g in dict.fromkeys(groups[::-1])],
          frameon=False, loc="lower right")
clean_axes(ax)
plt.tight_layout()
plt.savefig(VIS / "2_what_the_model_uses.png", dpi=200)
plt.close()

# 3. Proporcion de churn segun el puntaje del modelo (prueba)
pt = pd.read_csv(RES / "preds_test.csv")
score, y = pt["LightGBM"].values, pt["churn"].values
edges = np.arange(0, 1.0001, 0.025)
idx = np.clip(np.digitize(score, edges) - 1, 0, len(edges) - 2)
shown = [i for i in range(len(edges) - 1) if (idx == i).sum() >= 50]
rate = np.array([y[idx == i].mean() for i in shown]) * 100
x = edges[shown] + 0.0125
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.bar(x, rate, width=0.023, color=ORANGE, label="Churned")
ax.bar(x, 100 - rate, width=0.023, bottom=rate, color=BLUE, label="Stayed")
LOW, HIGH = 0.30, 0.70
for cut in (LOW, HIGH):
    ax.axvline(cut, color="black", linewidth=2, linestyle="--")
zones = [("Lower risk (score < 0.30)", score < LOW), ("Higher risk (score >= 0.70)", score >= HIGH)]
for k, (lab, m) in enumerate(zones):
    fig.text(0.08 + 0.48 * k, 0.04,
             f"{lab}\n{m.mean():.1%} of customers | {y[m].mean():.1%} churned",
             fontsize=12, color=NAVY, bbox=dict(boxstyle="round", fc="#DCE6F5", ec="none"))
ax.set_xlim(x.min() - 0.02, x.max() + 0.02)
ax.set_ylim(0, 100)
ax.set_xlabel("Model churn score (bins with at least 50 customers)")
ax.set_ylabel("Share of customers (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
clean_axes(ax)
plt.tight_layout(rect=(0, 0.13, 1, 1))
plt.savefig(VIS / "3_churn_by_score.png", dpi=200)
plt.close()
pd.DataFrame([{"zone": lab, "customers_%": m.mean() * 100, "churned_%": y[m].mean() * 100}
              for lab, m in zones]).round(2).to_csv(RES / "risk_zones.csv", index=False)
print(pd.read_csv(RES / "risk_zones.csv"))

# 4. Flujo del enfoque
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.2)
ax.axis("off")
steps = [("Customer data", "usage, billing,\ndevice, profile"), ("Machine learning", "churn score\nper customer"),
         ("Expected value", "score x success\nx value - cost"), ("Decision", "contact only if\nvalue is positive")]
for i, (head, body) in enumerate(steps):
    x0 = 0.15 + i * 3.05
    ax.add_patch(FancyBboxPatch((x0, 0.6), 2.4, 2.0, boxstyle="round,pad=0.05", fc="#DCE6F5", ec=NAVY, lw=1.5))
    ax.text(x0 + 1.2, 2.1, head, ha="center", fontsize=14, color=NAVY, fontweight="bold")
    ax.text(x0 + 1.2, 1.25, body, ha="center", va="center", fontsize=12, color=NAVY)
    if i < 3:
        ax.annotate("", xy=(x0 + 2.98, 1.6), xytext=(x0 + 2.52, 1.6),
                    arrowprops=dict(arrowstyle="-|>", lw=3, color=ORANGE))
plt.tight_layout()
plt.savefig(VIS / "4_approach_flow.png", dpi=200)
plt.close()

# 5. Comparacion de politicas (caso base)
pol = pd.read_csv(RES / "policy_comparison_base_case.csv").iloc[1:].iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 4.8))
colors = [ORANGE if p.startswith("Expected") else GRAY for p in pol["policy"]]
bars = ax.barh(pol["policy"], pol["net_benefit_per_1000_customers"], color=colors)
ax.bar_label(bars, labels=[f"{v:,.0f}  ({c:.0f}% contacted)" for v, c in
                           zip(pol["net_benefit_per_1000_customers"], pol["customers_contacted_%"])], padding=5)
ax.set_xlim(0, pol["net_benefit_per_1000_customers"].max() * 1.38)
ax.set_xticks([])
ax.set_title("Net benefit per 1,000 customers (test set)\nAssumed: success 20%, offer cost 15, margin 30%, 12 months",
             fontsize=12, color=NAVY)
ax.spines[["top", "right", "bottom"]].set_visible(False)
plt.tight_layout()
plt.savefig(VIS / "5_who_to_contact.png", dpi=200)
plt.close()
print("listo")
