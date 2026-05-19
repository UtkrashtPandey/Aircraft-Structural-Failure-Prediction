"""
================================================
Aircraft Structural Failure — Model Training
================================================
Trains three ML models:
  1. Random Forest Classifier
  2. XGBoost Classifier
  3. Deep Neural Network (TensorFlow/Keras)

Saves best models, metrics, and SHAP explainer.

Author  : Aerospace AI System
Version : 1.0.0
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
MODEL_DIR= BASE / "models"
VIS_DIR  = BASE / "visuals"
for d in [MODEL_DIR, VIS_DIR]:
    d.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────
def load_data():
    csv = DATA_DIR / "aerospace_structural_data.csv"
    if not csv.exists():
        print("⚠  Dataset not found — generating now...")
        sys.path.insert(0, str(DATA_DIR))
        import generate_dataset  # noqa: F401
    df = pd.read_csv(csv)
    print(f"✅ Loaded dataset: {df.shape}")
    return df


def preprocess(df: pd.DataFrame):
    drop_cols = ["sample_id", "failure_label", "risk_class",
                 "structural_health_score", "rul_cycles"]

    # Encode categoricals
    cat_cols = ["component", "material", "load_case"]
    le_map = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_map[c] = le

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["failure_label"].copy()

    return X, y, le_map, feature_cols


# ─────────────────────────────────────────────
# 2. Evaluation Helper
# ─────────────────────────────────────────────
def evaluate(name, model, X_test, y_test, results: dict):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="weighted")
    roc   = roc_auc_score(y_test, y_proba)
    report= classification_report(y_test, y_pred, output_dict=True)

    results[name] = {"accuracy": acc, "f1_score": f1, "roc_auc": roc}
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Fail", "Fail"],
                yticklabels=["No Fail", "Fail"])
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"cm_{name.replace(' ','_')}.png", dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr, roc


# ─────────────────────────────────────────────
# 3. ROC Combined Plot
# ─────────────────────────────────────────────
def plot_roc_combined(roc_data):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#00B4D8", "#F77F00", "#06D6A0"]
    for (name, fpr, tpr, auc_score), color in zip(roc_data, colors):
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name} (AUC={auc_score:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "roc_combined.png", dpi=150)
    plt.close()
    print("✅ ROC curve saved.")


# ─────────────────────────────────────────────
# 4. Feature Importance Plot
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return
    idx  = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(np.array(feature_names)[idx],
                   importances[idx], color="#00B4D8")
    ax.set_title(f"Top 20 Feature Importances — {model_name}", fontsize=12)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"feat_imp_{model_name.replace(' ','_')}.png", dpi=150)
    plt.close()
    print(f"✅ Feature importance plot saved for {model_name}.")


# ─────────────────────────────────────────────
# 5. SHAP Analysis
# ─────────────────────────────────────────────
def run_shap(model, X_test, feature_names, model_name):
    try:
        print(f"\n🔍 Running SHAP for {model_name}...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:500])

        # Handle binary output shape
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv, X_test[:500], feature_names=feature_names,
                          show=False, plot_type="bar")
        plt.title(f"SHAP Summary — {model_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"shap_{model_name.replace(' ','_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Save explainer
        joblib.dump(explainer, MODEL_DIR / f"shap_{model_name.replace(' ','_')}.pkl")
        print(f"✅ SHAP saved for {model_name}.")
    except Exception as e:
        print(f"⚠  SHAP failed for {model_name}: {e}")


# ─────────────────────────────────────────────
# 6. Neural Network (Keras)
# ─────────────────────────────────────────────
def build_neural_network(input_dim: int):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        tf.random.set_seed(42)

        model = Sequential([
            Dense(256, activation="relu", input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.25),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=Adam(1e-3),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        return model, True
    except ImportError:
        print("⚠  TensorFlow not installed. Skipping NN.")
        return None, False


class KerasSklearnWrapper:
    """Thin wrapper so Keras model has predict_proba interface."""
    def __init__(self, keras_model):
        self.model = keras_model

    def predict(self, X):
        proba = self.model.predict(X, verbose=0).flatten()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        proba = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])


# ─────────────────────────────────────────────
# 7. RUL Estimation (Regression)
# ─────────────────────────────────────────────
def train_rul_model(df_orig: pd.DataFrame):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    df = df_orig.copy()
    cat_cols = ["component", "material", "load_case"]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    drop_cols = ["sample_id", "failure_label", "risk_class",
                 "rul_cycles", "structural_health_score"]
    X = df.drop(columns=drop_cols)
    y = df["rul_cycles"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                    max_depth=5, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n📐 RUL Model — MAE: {mae:.0f} cycles  R²: {r2:.4f}")

    # RUL scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test[:500], y_pred[:500], alpha=0.3, s=10, color="#06D6A0")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            "r--", lw=2)
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"RUL Estimation  (R²={r2:.3f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "rul_estimation.png", dpi=150)
    plt.close()

    joblib.dump(reg, MODEL_DIR / "rul_model.pkl")
    print("✅ RUL model saved.")
    return {"mae": mae, "r2": r2}


# ─────────────────────────────────────────────
# 8. Model Comparison Chart
# ─────────────────────────────────────────────
def plot_model_comparison(results: dict):
    names   = list(results.keys())
    metrics = ["accuracy", "f1_score", "roc_auc"]
    labels  = ["Accuracy", "F1-Score", "ROC-AUC"]
    colors  = ["#00B4D8", "#F77F00", "#06D6A0"]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=label, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print("✅ Model comparison chart saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Aircraft Structural Failure — Model Training")
    print("=" * 60)

    df       = load_data()
    X, y, le_map, feature_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    results  = {}
    roc_data = []

    # ── Random Forest ──────────────────────────
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                min_samples_split=5, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    fpr, tpr, auc = evaluate("Random Forest", rf, X_test, y_test, results)
    roc_data.append(("Random Forest", fpr, tpr, auc))
    plot_feature_importance(rf, feature_cols, "Random Forest")
    run_shap(rf, X_test, feature_cols, "Random Forest")
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")

    # ── XGBoost ───────────────────────────────
    print("\n⚡ Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=7,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42, n_jobs=-1,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    fpr, tpr, auc = evaluate("XGBoost", xgb_model, X_test, y_test, results)
    roc_data.append(("XGBoost", fpr, tpr, auc))
    plot_feature_importance(xgb_model, feature_cols, "XGBoost")
    run_shap(xgb_model, X_test, feature_cols, "XGBoost")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost.pkl")

    # ── Neural Network ─────────────────────────
    keras_model, tf_available = build_neural_network(X_train_s.shape[1])
    if tf_available:
        print("\n🧠 Training Neural Network...")
        from tensorflow.keras.callbacks import EarlyStopping
        es = EarlyStopping(patience=10, restore_best_weights=True)
        keras_model.fit(X_train_s, y_train,
                        validation_split=0.15,
                        epochs=80, batch_size=256,
                        callbacks=[es], verbose=0)
        wrapper = KerasSklearnWrapper(keras_model)
        fpr, tpr, auc = evaluate("Neural Network", wrapper, X_test_s, y_test, results)
        roc_data.append(("Neural Network", fpr, tpr, auc))
        keras_model.save(MODEL_DIR / "neural_network.h5")

    # ── Plots ──────────────────────────────────
    plot_roc_combined(roc_data)
    plot_model_comparison(results)

    # ── RUL Model ──────────────────────────────
    rul_metrics = train_rul_model(pd.read_csv(DATA_DIR / "aerospace_structural_data.csv"))

    # ── Save metrics ───────────────────────────
    all_metrics = {"classification": results, "rul": rul_metrics}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Save feature list & encoders ───────────
    joblib.dump(le_map,       MODEL_DIR / "label_encoders.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    print("\n" + "=" * 60)
    print("  ✅ Training complete. All models saved to /models/")
    print("=" * 60)

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"  🏆 Best model by ROC-AUC: {best} "
          f"({results[best]['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
