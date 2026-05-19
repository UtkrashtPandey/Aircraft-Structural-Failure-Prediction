"""
================================================
Aircraft Structural Failure — Prediction Module
================================================
Loads trained models and runs inference on new
structural data. Returns failure probability,
risk class, and RUL estimate.

Author  : Aerospace AI System
Version : 1.0.0
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Optional

warnings.filterwarnings("ignore")

BASE      = Path(__file__).parent
MODEL_DIR = BASE / "models"

# ─────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────
_MODELS = {}

def _load_models():
    """Lazy-load all models into memory once."""
    global _MODELS
    if _MODELS:
        return _MODELS

    try:
        _MODELS["rf"]       = joblib.load(MODEL_DIR / "random_forest.pkl")
        _MODELS["xgb"]      = joblib.load(MODEL_DIR / "xgboost.pkl")
        _MODELS["scaler"]   = joblib.load(MODEL_DIR / "scaler.pkl")
        _MODELS["le_map"]   = joblib.load(MODEL_DIR / "label_encoders.pkl")
        _MODELS["features"] = joblib.load(MODEL_DIR / "feature_cols.pkl")
        _MODELS["rul"]      = joblib.load(MODEL_DIR / "rul_model.pkl")
        print("✅ Models loaded.")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model files not found. Run train_model.py first.\n{e}"
        )

    # Try loading Keras NN
    try:
        from tensorflow.keras.models import load_model
        _MODELS["nn"] = load_model(MODEL_DIR / "neural_network.h5", compile=False)
        _MODELS["nn_available"] = True
    except Exception:
        _MODELS["nn_available"] = False

    return _MODELS


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def _preprocess_input(data: pd.DataFrame, models: dict) -> tuple:
    """Encode categoricals and align features with training schema."""
    df = data.copy()
    le_map = models["le_map"]

    for col, le in le_map.items():
        if col in df.columns:
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col].astype(str))

    feature_cols = models["features"]
    # Fill any missing columns with 0
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    return df[feature_cols], models["scaler"].transform(df[feature_cols])


# ─────────────────────────────────────────────
# Risk Label
# ─────────────────────────────────────────────
def _proba_to_risk(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.70:
        return "High"
    else:
        return "Critical"


# ─────────────────────────────────────────────
# Health Score
# ─────────────────────────────────────────────
def _health_score(prob: float, rul: float, max_rul: float = 100000) -> float:
    rul_score = min(rul / max_rul, 1.0) * 100
    fail_score = (1 - prob) * 100
    return round(0.6 * fail_score + 0.4 * rul_score, 1)


# ─────────────────────────────────────────────
# Single Prediction
# ─────────────────────────────────────────────
def predict_single(input_dict: dict,
                   model_name: str = "xgb") -> dict:
    """
    Predict failure probability for a single component.

    Parameters
    ----------
    input_dict  : dict — feature values (matches dataset columns)
    model_name  : 'rf' | 'xgb' | 'nn'

    Returns
    -------
    dict with:
        failure_probability, risk_class, structural_health_score,
        rul_cycles, model_used, confidence_flag
    """
    models = _load_models()
    df = pd.DataFrame([input_dict])
    X_raw, X_scaled = _preprocess_input(df, models)

    # Choose model
    if model_name == "nn" and models.get("nn_available"):
        proba = float(models["nn"].predict(X_scaled, verbose=0).flatten()[0])
    elif model_name == "rf":
        proba = float(models["rf"].predict_proba(X_raw)[:, 1][0])
    else:  # default xgb
        proba = float(models["xgb"].predict_proba(X_raw)[:, 1][0])

    rul = int(np.clip(models["rul"].predict(X_raw)[0], 0, 100000))

    risk  = _proba_to_risk(proba)
    score = _health_score(proba, rul)

    return {
        "failure_probability":    round(proba, 4),
        "failure_probability_pct":round(proba * 100, 2),
        "risk_class":             risk,
        "structural_health_score":score,
        "rul_cycles":             rul,
        "model_used":             model_name,
        "confidence_flag":        "High" if abs(proba - 0.5) > 0.25 else "Low",
    }


# ─────────────────────────────────────────────
# Batch Prediction
# ─────────────────────────────────────────────
def predict_batch(df: pd.DataFrame,
                  model_name: str = "xgb") -> pd.DataFrame:
    """
    Run predictions on a full DataFrame.

    Returns the input DataFrame with added prediction columns.
    """
    models = _load_models()
    X_raw, X_scaled = _preprocess_input(df, models)

    if model_name == "nn" and models.get("nn_available"):
        probas = models["nn"].predict(X_scaled, verbose=0).flatten()
    elif model_name == "rf":
        probas = models["rf"].predict_proba(X_raw)[:, 1]
    else:
        probas = models["xgb"].predict_proba(X_raw)[:, 1]

    ruls = np.clip(models["rul"].predict(X_raw), 0, 100000).astype(int)

    result = df.copy()
    result["failure_probability"]     = probas.round(4)
    result["risk_class_predicted"]    = [_proba_to_risk(p) for p in probas]
    result["structural_health_score"] = [_health_score(p, r)
                                          for p, r in zip(probas, ruls)]
    result["rul_predicted"]           = ruls

    return result


# ─────────────────────────────────────────────
# Fleet Summary
# ─────────────────────────────────────────────
def fleet_summary(result_df: pd.DataFrame) -> dict:
    """Compute aggregate fleet health statistics from batch predictions."""
    risk_counts = result_df["risk_class_predicted"].value_counts().to_dict()

    return {
        "total_components":   len(result_df),
        "critical_count":     int(risk_counts.get("Critical", 0)),
        "high_count":         int(risk_counts.get("High", 0)),
        "medium_count":       int(risk_counts.get("Medium", 0)),
        "low_count":          int(risk_counts.get("Low", 0)),
        "avg_health_score":   float(result_df["structural_health_score"].mean()),
        "avg_rul":            int(result_df["rul_predicted"].mean()),
        "near_failure_pct":   float((result_df["failure_probability"] > 0.5).mean() * 100),
        "top_risky_component":result_df.loc[
            result_df["failure_probability"].idxmax(), "component"
        ] if "component" in result_df.columns else "N/A",
        "top_risky_material": result_df.loc[
            result_df["failure_probability"].idxmax(), "material"
        ] if "material" in result_df.columns else "N/A",
    }


# ─────────────────────────────────────────────
# Load Saved Metrics
# ─────────────────────────────────────────────
def load_metrics() -> dict:
    """Load saved model performance metrics."""
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "component":                "Wing_Spar",
        "material":                 "Al_7075-T6",
        "load_case":                "2.5g_Pull_Up",
        "thickness_mm":             12.5,
        "cross_section_area_cm2":   45.0,
        "applied_load_kN":          320.0,
        "load_factor_g":            2.5,
        "von_mises_stress_MPa":     387.0,
        "effective_stress_MPa":     580.5,
        "stress_concentration_Kt":  1.5,
        "stress_ratio_R":           0.1,
        "strain":                   0.0054,
        "deformation_mm":           2.34,
        "elastic_modulus_GPa":      71.7,
        "yield_strength_MPa":       503.0,
        "density_g_cm3":            2.81,
        "temperature_C":            87.0,
        "fatigue_cycles":           62000,
        "fatigue_limit_MPa":        159.0,
        "crack_length_mm":          0.0821,
        "crack_growth_rate":        2.1e-9,
        "cumulative_damage_index":  0.68,
        "safety_margin":            0.87,
        "utilization_ratio":        1.15,
    }

    print("=" * 50)
    print("  Prediction Test")
    print("=" * 50)
    result = predict_single(sample, model_name="xgb")
    for k, v in result.items():
        print(f"  {k:<30}: {v}")
