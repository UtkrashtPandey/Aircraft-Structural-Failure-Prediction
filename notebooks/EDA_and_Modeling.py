"""
================================================
EDA & Modeling Notebook
================================================
Exploratory Data Analysis for Aerospace Structural
Failure Dataset — run cell-by-cell in Jupyter
or execute as a script.

Convert to notebook:
  pip install jupytext
  jupytext --to notebook notebooks/EDA_and_Modeling.py
================================================
"""

# %% [markdown]
# # Aircraft Structural Failure Prediction — EDA & Modeling
# **Aerospace AI Project | Winter Semester 2025-2026**
#
# This notebook explores the FEA-inspired structural dataset and walks through
# the complete ML pipeline from raw features to failure probability.

# %% Setup
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path().absolute().parent))

plt.style.use("dark_background")
sns.set_palette("husl")

# %% Load Dataset
df = pd.read_csv("../data/aerospace_structural_data.csv")
print(f"Dataset shape: {df.shape}")
df.head()

# %% Basic Statistics
print("=" * 60)
print("Dataset Overview")
print("=" * 60)
print(f"\nShape      : {df.shape}")
print(f"Failure %  : {df['failure_label'].mean()*100:.1f}%")
print(f"Risk dist  :\n{df['risk_class'].value_counts()}")
print("\nDescriptive Stats:")
df.describe().round(3)

# %% Missing Values
print("Missing values:", df.isnull().sum().sum())

# %% Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Binary failure
df["failure_label"].value_counts().plot(
    kind="bar", ax=axes[0], color=["#64FFDA", "#FF6B6B"],
    title="Failure Label Distribution")
axes[0].set_xticklabels(["No Failure", "Failure"], rotation=0)

# Risk class
risk_order = ["Low", "Medium", "High", "Critical"]
colors = ["#64FFDA", "#FFD700", "#FFA033", "#FF6B6B"]
df["risk_class"].value_counts()[risk_order].plot(
    kind="bar", ax=axes[1], color=colors,
    title="Risk Class Distribution")
axes[1].set_xticklabels(risk_order, rotation=0)

plt.tight_layout()
plt.savefig("../visuals/target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Stress Distribution by Component
fig, ax = plt.subplots(figsize=(12, 5))
df.boxplot(column="von_mises_stress_MPa", by="component",
           ax=ax, patch_artist=True)
ax.set_title("Von Mises Stress Distribution by Component")
ax.set_xlabel("Component"); ax.set_ylabel("Stress (MPa)")
plt.suptitle("")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../visuals/stress_by_component.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Correlation Heatmap
num_cols = [
    "von_mises_stress_MPa", "effective_stress_MPa", "strain",
    "fatigue_cycles", "crack_length_mm", "temperature_C",
    "cumulative_damage_index", "utilization_ratio",
    "structural_health_score", "rul_cycles"
]
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("../visuals/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Fatigue S-N Curve Approximation
fig, ax = plt.subplots(figsize=(9, 5))
for material, group in df.groupby("material"):
    sample = group.sample(min(200, len(group)))
    ax.scatter(np.log10(sample["fatigue_cycles"]),
               sample["von_mises_stress_MPa"],
               label=material, alpha=0.5, s=15)
ax.set_xlabel("log₁₀(Fatigue Cycles)")
ax.set_ylabel("Von Mises Stress (MPa)")
ax.set_title("S-N Curve Approximation by Material")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../visuals/sn_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Failure Probability vs Utilization
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 2.5, 25)
df["util_bin"] = pd.cut(df["utilization_ratio"], bins=bins)
fail_rate = df.groupby("util_bin")["failure_label"].mean()
fail_rate.plot(kind="bar", ax=ax, color="#FF6B6B", width=0.85)
ax.set_xlabel("Utilization Ratio Bin (σ_eff / σ_yield)")
ax.set_ylabel("Failure Rate")
ax.set_title("Failure Rate vs Stress Utilization Ratio")
ax.set_xticklabels([f"{b.left:.1f}" for b in fail_rate.index], rotation=45)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../visuals/failure_vs_utilization.png", dpi=150, bbox_inches="tight")
plt.show()

# %% RUL Distribution
fig, ax = plt.subplots(figsize=(9, 4))
df["rul_cycles"].hist(bins=60, ax=ax, color="#64FFDA", edgecolor="#0A0E1A", alpha=0.8)
ax.set_xlabel("Remaining Useful Life (cycles)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Remaining Useful Life")
ax.axvline(df["rul_cycles"].mean(), color="#FFD700", lw=2,
           label=f"Mean={df['rul_cycles'].mean():.0f}")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../visuals/rul_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Quick Model Benchmark
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

df_model = df.copy()
for c in ["component","material","load_case"]:
    df_model[c] = LabelEncoder().fit_transform(df_model[c])

drop = ["sample_id","failure_label","risk_class","structural_health_score","rul_cycles"]
X = df_model.drop(columns=drop)
y = df_model["failure_label"]

print("Running 5-fold CV...")
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost":       xgb.XGBClassifier(n_estimators=100, random_state=42,
                                        eval_metric="logloss", use_label_encoder=False),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  {name:20s}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

print("\n✅ EDA complete. Visualizations saved to /visuals/")
