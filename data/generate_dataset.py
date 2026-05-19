"""
================================================
Aircraft Structural FEA Dataset Generator
================================================
Generates synthetic but realistic aerospace-grade
Finite Element Analysis (FEA) structural data for
ML model training and validation.

Author  : Aerospace AI System
Version : 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Seed for reproducibility
# ─────────────────────────────────────────────
np.random.seed(42)
N = 12000  # Total samples

# ─────────────────────────────────────────────
# 1. Component & Material Definitions
# ─────────────────────────────────────────────
COMPONENTS = [
    "Wing_Spar", "Fuselage_Frame", "Landing_Gear",
    "Tail_Assembly", "Engine_Mount", "Bulkhead",
    "Rib_Section", "Skin_Panel"
]

MATERIALS = {
    "Al_7075-T6":   {"E": 71.7,  "yield": 503, "density": 2.81, "fatigue_limit": 159},
    "Ti-6Al-4V":    {"E": 113.8, "yield": 880, "density": 4.43, "fatigue_limit": 510},
    "CFRP":         {"E": 135.0, "yield": 600, "density": 1.60, "fatigue_limit": 300},
    "Steel_4340":   {"E": 205.0, "yield": 470, "density": 7.85, "fatigue_limit": 413},
    "Al_2024-T3":   {"E": 73.1,  "yield": 345, "density": 2.78, "fatigue_limit": 138},
    "Inconel_718":  {"E": 200.0, "yield": 1034,"density": 8.19, "fatigue_limit": 620},
}

LOAD_CASES = [
    "1g_Level_Flight", "2.5g_Pull_Up", "Neg1g_Push_Over",
    "Ground_Taxi", "Hard_Landing", "Gust_Load", "Pressurization"
]

# ─────────────────────────────────────────────
# 2. Generate Base Parameters
# ─────────────────────────────────────────────
component      = np.random.choice(COMPONENTS, N)
material_names = np.random.choice(list(MATERIALS.keys()), N)
load_case      = np.random.choice(LOAD_CASES, N)

# Pull material properties per sample
E_modulus    = np.array([MATERIALS[m]["E"]           for m in material_names])
yield_stress = np.array([MATERIALS[m]["yield"]        for m in material_names])
fatigue_lim  = np.array([MATERIALS[m]["fatigue_limit"]for m in material_names])
density      = np.array([MATERIALS[m]["density"]      for m in material_names])

# ─────────────────────────────────────────────
# 3. Physics-informed Feature Generation
# ─────────────────────────────────────────────
# Applied load (kN)
applied_load = np.random.uniform(10, 850, N)

# Cross-section area (cm²)
cross_section_area = np.random.uniform(5, 120, N)

# Stress = Force / Area (MPa)
von_mises_stress = (applied_load * 1000) / (cross_section_area * 1e-4) / 1e6
von_mises_stress += np.random.normal(0, 15, N)
von_mises_stress = np.clip(von_mises_stress, 5, 1200)

# Strain = Stress / E
strain = von_mises_stress / (E_modulus * 1000)
strain += np.random.normal(0, 0.0001, N)
strain = np.clip(strain, 0.00001, 0.05)

# Temperature effects (°C) — wing leading edge / engine mount get hotter
temperature = np.where(
    np.isin(component, ["Engine_Mount", "Wing_Spar"]),
    np.random.uniform(80, 320, N),
    np.random.uniform(-55, 120, N)
)

# Fatigue cycles (flight cycles)
fatigue_cycles = np.random.exponential(scale=15000, size=N).astype(int)
fatigue_cycles = np.clip(fatigue_cycles, 100, 120000)

# Deformation (mm)
deformation = (von_mises_stress * np.random.uniform(100, 800, N)) / (E_modulus * 1000)
deformation = np.clip(deformation + np.random.normal(0, 0.2, N), 0.001, 25.0)

# Stress concentration factor (Kt) — accounts for holes, notches
stress_concentration = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], N,
                                         p=[0.25, 0.30, 0.20, 0.13, 0.08, 0.04])

# Effective stress with concentration
effective_stress = von_mises_stress * stress_concentration
effective_stress = np.clip(effective_stress, 5, 2500)

# Crack propagation indicator (mm) — Paris law inspired
delta_K = effective_stress * np.sqrt(np.pi * np.random.uniform(0.001, 0.05, N))
crack_growth_rate = 1.5e-11 * (delta_K ** 3.8)
crack_length = crack_growth_rate * fatigue_cycles * 1000
crack_length = np.clip(crack_length, 0.0, 50.0)

# Stress ratio R = min_stress / max_stress
stress_ratio = np.random.uniform(-1.0, 0.8, N)

# Load factor (g)
load_factor = np.random.uniform(-1.0, 3.5, N)

# Thickness (mm)
thickness = np.random.uniform(1.5, 35.0, N)

# Safety margin
safety_margin = yield_stress / np.clip(effective_stress, 1, 9999)
safety_margin = np.clip(safety_margin, 0.1, 10.0)

# Utilization ratio (how close to yield)
utilization_ratio = np.clip(effective_stress / yield_stress, 0.0, 5.0)

# Remaining life fraction (Miner's rule inspired)
damage_per_cycle = (von_mises_stress / fatigue_lim) ** 4
cumulative_damage = damage_per_cycle * fatigue_cycles / 1e6
cumulative_damage = np.clip(cumulative_damage, 0.0, 5.0)

# Remaining Useful Life (RUL) in flight cycles
rul = np.clip((1.0 - cumulative_damage) * 50000, 0, 100000).astype(int)

# ─────────────────────────────────────────────
# 4. Target Variable Engineering
# ─────────────────────────────────────────────
# Failure label (binary)
failure_score = (
    0.35 * np.clip(utilization_ratio, 0, 3) / 3 +
    0.25 * np.clip(cumulative_damage, 0, 2) / 2 +
    0.20 * np.clip(crack_length / 50, 0, 1) +
    0.10 * np.clip((temperature + 55) / 375, 0, 1) +
    0.10 * np.clip((stress_concentration - 1) / 2.5, 0, 1)
)
failure_score += np.random.normal(0, 0.05, N)
failure_label = (failure_score > 0.55).astype(int)

# Risk Classification (multi-class)
risk_class = np.where(failure_score < 0.25, "Low",
             np.where(failure_score < 0.50, "Medium",
             np.where(failure_score < 0.70, "High", "Critical")))

# ─────────────────────────────────────────────
# 5. Assemble DataFrame
# ─────────────────────────────────────────────
df = pd.DataFrame({
    # Identifiers
    "sample_id":             range(1, N + 1),
    "component":             component,
    "material":              material_names,
    "load_case":             load_case,

    # Geometry
    "thickness_mm":          np.round(thickness, 2),
    "cross_section_area_cm2":np.round(cross_section_area, 2),

    # Loads & Stress
    "applied_load_kN":       np.round(applied_load, 2),
    "load_factor_g":         np.round(load_factor, 3),
    "von_mises_stress_MPa":  np.round(von_mises_stress, 2),
    "effective_stress_MPa":  np.round(effective_stress, 2),
    "stress_concentration_Kt": stress_concentration,
    "stress_ratio_R":        np.round(stress_ratio, 3),

    # Strain & Deformation
    "strain":                np.round(strain, 6),
    "deformation_mm":        np.round(deformation, 4),

    # Material Properties
    "elastic_modulus_GPa":   E_modulus,
    "yield_strength_MPa":    yield_stress,
    "density_g_cm3":         density,

    # Thermal
    "temperature_C":         np.round(temperature, 1),

    # Fatigue & Damage
    "fatigue_cycles":        fatigue_cycles,
    "fatigue_limit_MPa":     fatigue_lim,
    "crack_length_mm":       np.round(crack_length, 4),
    "crack_growth_rate":     np.round(crack_growth_rate, 12),
    "cumulative_damage_index":np.round(cumulative_damage, 4),

    # Derived Health Metrics
    "safety_margin":         np.round(safety_margin, 3),
    "utilization_ratio":     np.round(utilization_ratio, 4),
    "structural_health_score": np.round((1 - np.clip(failure_score, 0, 1)) * 100, 2),

    # Remaining Useful Life
    "rul_cycles":            rul,

    # Targets
    "failure_label":         failure_label,
    "risk_class":            risk_class,
})

# ─────────────────────────────────────────────
# 6. Save Dataset
# ─────────────────────────────────────────────
output_path = Path(__file__).parent / "aerospace_structural_data.csv"
df.to_csv(output_path, index=False)

print(f"✅ Dataset generated: {output_path}")
print(f"   Shape     : {df.shape}")
print(f"   Failures  : {df['failure_label'].sum()} ({df['failure_label'].mean()*100:.1f}%)")
print(f"   Risk dist :\n{df['risk_class'].value_counts()}")

if __name__ == "__main__":
    pass
