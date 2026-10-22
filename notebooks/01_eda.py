"""
notebooks/01_eda.py
--------------------
Exploratory Data Analysis for the Medical Insurance dataset.
Run as a script to generate all plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ------------------------------------------------------------------ #
# Style
# ------------------------------------------------------------------ #
DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT1  = "#7c83fd"
ACCENT2  = "#4ecdc4"
WARN     = "#ff6b6b"
GOLD     = "#ffd93d"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   "#3d4058",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

os.makedirs("reports", exist_ok=True)


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ------------------------------------------------------------------ #
# 1. Load data
# ------------------------------------------------------------------ #
section("1. Basic Overview")
df = pd.read_csv("data/insurance.csv")
print(f"\nShape : {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nSummary statistics:")
print(df.describe().round(2))

# ------------------------------------------------------------------ #
# 2. Charges Distribution
# ------------------------------------------------------------------ #
section("2. Charges Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Insurance Charges Distribution", fontsize=16, fontweight="bold", y=1.02)

# Raw distribution
axes[0].hist(df["charges"], bins=40, color=ACCENT1, edgecolor=DARK_BG, alpha=0.85)
axes[0].set_xlabel("Annual Charges ($)", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)
axes[0].set_title("Raw Charges", fontsize=13, fontweight="bold")
axes[0].grid(True)

# Log-transformed
log_charges = np.log1p(df["charges"])
axes[1].hist(log_charges, bins=40, color=ACCENT2, edgecolor=DARK_BG, alpha=0.85)
axes[1].set_xlabel("log(1 + Annual Charges)", fontsize=12)
axes[1].set_ylabel("Count", fontsize=12)
axes[1].set_title("Log-Transformed Charges", fontsize=13, fontweight="bold")
axes[1].grid(True)

skew_raw = df["charges"].skew()
skew_log = log_charges.skew()
print(f"  Raw skewness  : {skew_raw:.3f}")
print(f"  Log skewness  : {skew_log:.3f}")

fig.tight_layout()
fig.savefig("reports/eda_charges_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/eda_charges_distribution.png")


# ------------------------------------------------------------------ #
# 3. Charges vs Age (coloured by smoker)
# ------------------------------------------------------------------ #
section("3. Charges vs Age & Smoker")

fig, ax = plt.subplots(figsize=(10, 7))
colors_smoker = {True: WARN, False: ACCENT1}
for smk, grp in df.groupby(df["smoker"] == "yes"):
    ax.scatter(
        grp["age"], grp["charges"],
        c=colors_smoker[smk], alpha=0.55, s=22, edgecolors="none",
        label="Smoker" if smk else "Non-Smoker",
    )
ax.set_xlabel("Age", fontsize=13)
ax.set_ylabel("Annual Charges ($)", fontsize=13)
ax.set_title("Charges vs Age (by Smoking Status)", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True)
fig.tight_layout()
fig.savefig("reports/eda_charges_vs_age.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/eda_charges_vs_age.png")


# ------------------------------------------------------------------ #
# 4. BMI vs Charges (coloured by smoker)
# ------------------------------------------------------------------ #
section("4. BMI vs Charges")

fig, ax = plt.subplots(figsize=(10, 7))
for smk, grp in df.groupby(df["smoker"] == "yes"):
    ax.scatter(
        grp["bmi"], grp["charges"],
        c=colors_smoker[smk], alpha=0.55, s=22, edgecolors="none",
        label="Smoker" if smk else "Non-Smoker",
    )
ax.axvline(30, color=GOLD, lw=2, linestyle="--", label="BMI=30 (Obese threshold)")
ax.set_xlabel("BMI", fontsize=13)
ax.set_ylabel("Annual Charges ($)", fontsize=13)
ax.set_title("Charges vs BMI (by Smoking Status)", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True)
fig.tight_layout()
fig.savefig("reports/eda_bmi_vs_charges.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/eda_bmi_vs_charges.png")


# ------------------------------------------------------------------ #
# 5. Categorical boxplots
# ------------------------------------------------------------------ #
section("5. Charges by Categorical Features")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Charges by Categorical Features", fontsize=15, fontweight="bold", y=1.02)

cat_palette = [ACCENT1, ACCENT2, WARN, GOLD, "#c77dff", "#ff9f40"]

for ax, col in zip(axes, ["smoker", "region", "sex"]):
    categories = df[col].unique()
    data_groups = [df.loc[df[col] == cat, "charges"].values for cat in categories]
    bp = ax.boxplot(
        data_groups,
        labels=categories,
        patch_artist=True,
        medianprops=dict(color=GOLD, linewidth=2),
        whiskerprops=dict(color="#a0a0b0"),
        capprops=dict(color="#a0a0b0"),
        flierprops=dict(markerfacecolor="#ff6b6b", marker="o", alpha=0.4, markersize=3),
    )
    for patch, color in zip(bp["boxes"], cat_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col.capitalize(), fontsize=13, fontweight="bold")
    ax.set_xlabel(col.capitalize(), fontsize=11)
    ax.set_ylabel("Charges ($)", fontsize=11)
    ax.grid(True, axis="y")

fig.tight_layout()
fig.savefig("reports/eda_categorical_boxplots.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/eda_categorical_boxplots.png")


# ------------------------------------------------------------------ #
# 6. Correlation Heatmap
# ------------------------------------------------------------------ #
section("6. Correlation Heatmap")

df_encoded = df.copy()
df_encoded["smoker_num"] = (df["smoker"] == "yes").astype(int)
df_encoded["sex_num"] = (df["sex"] == "male").astype(int)
corr_cols = ["age", "bmi", "children", "smoker_num", "sex_num", "charges"]
corr_df = df_encoded[corr_cols].rename(columns={"smoker_num": "smoker", "sex_num": "sex(male)"})
corr = corr_df.corr()

print(f"\n  Correlations with charges:\n{corr['charges'].drop('charges').sort_values(ascending=False).round(3)}")

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr, mask=mask, cmap=cmap,
    vmin=-1, vmax=1, center=0,
    square=True, linewidths=0.5, linecolor="#0f1117",
    annot=True, fmt=".2f", annot_kws={"size": 11, "color": "white"},
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
ax.tick_params(axis="both", labelsize=11)
fig.tight_layout()
fig.savefig("reports/eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> reports/eda_correlation_heatmap.png")


# ------------------------------------------------------------------ #
# 7. Smoker impact statistics
# ------------------------------------------------------------------ #
section("7. Smoking Impact Statistics")

smoker_charges    = df.loc[df["smoker"] == "yes",  "charges"]
nonsmoker_charges = df.loc[df["smoker"] == "no",   "charges"]
t_stat, p_val = stats.ttest_ind(smoker_charges, nonsmoker_charges)

print(f"  Smoker    mean charge: ${smoker_charges.mean():,.0f}")
print(f"  Non-smoker mean charge: ${nonsmoker_charges.mean():,.0f}")
print(f"  Ratio: {smoker_charges.mean() / nonsmoker_charges.mean():.2f}x")
print(f"  t-statistic: {t_stat:.2f}  |  p-value: {p_val:.2e}")


section("EDA Complete")
print("\nKey Insights:")
print("  1. Charges are highly right-skewed (skew=1.51); log-transform normalises them.")
print("  2. Smoking is the strongest predictor: smokers pay ~4x more on average.")
print("  3. Age and BMI have positive linear relationships with charges.")
print("  4. BMI > 30 interacts strongly with smoking to create a high-risk cluster.")
print("  5. Region and sex have minimal marginal impact.")
print("\nAll plots saved to reports/")
