"""
================================================
GenAI Engine — Aerospace Engineering Insights
================================================
Uses OpenAI GPT (or fallback local logic) to:
  • Analyze ML prediction outputs
  • Generate engineering recommendations
  • Produce maintenance reports
  • Explain failure causes in plain language

Author  : Aerospace AI System
Version : 1.0.0
"""

import os
import json
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
# OpenAI client (graceful fallback if not set)
# ─────────────────────────────────────────────
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False
    _client = None

MODEL_NAME = "gpt-4o-mini"

# ─────────────────────────────────────────────
# System Prompt — Aerospace Engineering Expert
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a Senior Structural Integrity Engineer at an aerospace OEM (think Airbus, Boeing, Rolls-Royce).
You specialize in fatigue analysis, FEA interpretation, damage tolerance assessment, and
predictive maintenance of aircraft primary structures.

When given ML prediction results and sensor data, you:
1. Interpret structural health metrics using aerospace industry standards (FAR 25, MIL-HDBK-5, ASTM E647)
2. Identify root-cause failure mechanisms (fatigue, creep, stress corrosion, fretting)
3. Provide actionable maintenance recommendations aligned with MSG-3 methodology
4. Use precise engineering terminology while remaining accessible to maintenance crews
5. Always prioritize airworthiness and safety margins

Output must be structured, professional, and formatted for engineering reports.
""".strip()


# ─────────────────────────────────────────────
# 1. Core LLM Call
# ─────────────────────────────────────────────
def _llm_call(prompt: str, max_tokens: int = 800) -> str:
    """Call OpenAI API or return fallback engineering text."""
    if OPENAI_AVAILABLE and _client:
        try:
            resp = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return _fallback_response(prompt, str(e))
    else:
        return _fallback_response(prompt)


# ─────────────────────────────────────────────
# 2. Fallback (No API Key)
# ─────────────────────────────────────────────
def _fallback_response(prompt: str, error: str = "") -> str:
    """Rule-based engineering response when OpenAI is unavailable."""
    p = prompt.lower()

    if "wing spar" in p or "spar" in p:
        component_note = ("Wing spar section identified as primary load path. "
                          "Cyclic bending loads near root attachment are the dominant failure driver.")
    elif "engine mount" in p:
        component_note = ("Engine mount subject to combined thermal and vibratory loading. "
                          "High-cycle fatigue and fretting corrosion are leading failure modes.")
    elif "landing gear" in p:
        component_note = ("Landing gear structure experiences impact loads up to 2.5g. "
                          "Stress corrosion cracking in high-strength steel is a primary concern.")
    elif "fuselage" in p:
        component_note = ("Fuselage frame subject to pressurization cycles and bending loads. "
                          "Fatigue crack initiation at fastener holes is the dominant mechanism.")
    else:
        component_note = ("Structural component under combined mechanical and thermal loads. "
                          "Multi-axial fatigue assessment recommended per ASTM E2207.")

    if "critical" in p or "failure" in p:
        urgency = "⚠️  IMMEDIATE ACTION REQUIRED — Component approaches structural limit."
        action = ("Remove from service for detailed NDE inspection. "
                  "Perform fluorescent penetrant inspection (FPI) and eddy current testing. "
                  "Do not return to service without signed-off engineering disposition.")
    elif "high" in p:
        urgency = "🔶  ELEVATED RISK — Schedule inspection within next maintenance window."
        action = ("Reduce inspection interval by 50%. Perform borescope inspection. "
                  "Monitor crack indicators at next scheduled C-check.")
    else:
        urgency = "✅  ACCEPTABLE — Continue normal operation with standard monitoring."
        action = ("Maintain standard inspection intervals per AMM Chapter 51. "
                  "Log structural health scores for trend analysis.")

    return (
        f"**Structural Assessment Report**\n\n"
        f"{component_note}\n\n"
        f"**Risk Level:** {urgency}\n\n"
        f"**Recommended Action:** {action}\n\n"
        f"**Standard Reference:** MIL-HDBK-1823A (NDE), MSG-3 Rev 2021, FAR 25.571\n\n"
        f"*Note: This assessment uses rule-based inference. "
        f"Set OPENAI_API_KEY for AI-powered analysis.*"
    )


# ─────────────────────────────────────────────
# 3. Prediction Insight
# ─────────────────────────────────────────────
def generate_prediction_insight(prediction_data: dict) -> str:
    """
    Analyzes a single prediction result and returns engineering insight.

    Parameters
    ----------
    prediction_data : dict with keys:
        component, material, failure_probability, risk_class,
        von_mises_stress_MPa, utilization_ratio, fatigue_cycles,
        crack_length_mm, temperature_C, rul_cycles
    """
    prompt = f"""
Analyze the following structural health data for an aircraft component and provide
a concise engineering assessment (3-4 paragraphs):

Component       : {prediction_data.get('component', 'N/A')}
Material        : {prediction_data.get('material', 'N/A')}
Failure Probability: {prediction_data.get('failure_probability', 0)*100:.1f}%
Risk Class      : {prediction_data.get('risk_class', 'N/A')}
Von Mises Stress: {prediction_data.get('von_mises_stress_MPa', 0):.1f} MPa
Yield Strength  : {prediction_data.get('yield_strength_MPa', 0):.1f} MPa
Utilization Ratio: {prediction_data.get('utilization_ratio', 0):.3f}
Fatigue Cycles  : {prediction_data.get('fatigue_cycles', 0):,}
Crack Length    : {prediction_data.get('crack_length_mm', 0):.4f} mm
Temperature     : {prediction_data.get('temperature_C', 0):.1f} °C
Estimated RUL   : {prediction_data.get('rul_cycles', 0):,} cycles

Provide:
1. Failure mechanism identification
2. Physics-based root cause analysis
3. Specific maintenance recommendation with inspection method
4. Airworthiness disposition (safe/conditional/grounded)
""".strip()

    return _llm_call(prompt, max_tokens=700)


# ─────────────────────────────────────────────
# 4. Batch Fleet Report
# ─────────────────────────────────────────────
def generate_fleet_report(fleet_summary: dict) -> str:
    """
    Generates a fleet-level structural health report.

    Parameters
    ----------
    fleet_summary : dict with aggregate statistics
    """
    prompt = f"""
Generate a Fleet Structural Health Summary Report for an airline/MRO operator.

Fleet Statistics:
- Total components analyzed : {fleet_summary.get('total_components', 0)}
- Critical risk components  : {fleet_summary.get('critical_count', 0)}
- High risk components      : {fleet_summary.get('high_count', 0)}
- Average health score      : {fleet_summary.get('avg_health_score', 0):.1f}/100
- Components near failure   : {fleet_summary.get('near_failure_pct', 0):.1f}%
- Average RUL               : {fleet_summary.get('avg_rul', 0):,} cycles
- Most common risk component: {fleet_summary.get('top_risky_component', 'N/A')}
- Most common material issue: {fleet_summary.get('top_risky_material', 'N/A')}

Write a formal engineering report including:
1. Executive Summary (3 sentences)
2. Fleet Risk Assessment
3. Priority Maintenance Actions (numbered list)
4. Recommended Inspection Schedule
5. Cost/Safety Implications
""".strip()

    return _llm_call(prompt, max_tokens=900)


# ─────────────────────────────────────────────
# 5. Maintenance Recommendations
# ─────────────────────────────────────────────
def generate_maintenance_recommendations(component: str,
                                          risk_class: str,
                                          rul_cycles: int,
                                          material: str) -> str:
    """Generates specific MSG-3 aligned maintenance recommendations."""
    prompt = f"""
Provide specific, actionable MSG-3 maintenance recommendations for:

Component  : {component}
Material   : {material}
Risk Level : {risk_class}
RUL        : {rul_cycles:,} flight cycles remaining

Format as:
1. Immediate Actions (if any)
2. Inspection Method & Technique (FPI, UT, EC, Visual)
3. Inspection Interval (flight hours/cycles)
4. Acceptance Criteria
5. Disposition if Defect Found
6. Applicable AMM/SRM Reference (general)

Be specific to {component} and {material} failure modes.
""".strip()

    return _llm_call(prompt, max_tokens=600)


# ─────────────────────────────────────────────
# 6. Failure Cause Explanation (plain language)
# ─────────────────────────────────────────────
def explain_failure_cause(features: dict) -> str:
    """
    Explains why a component is predicted to fail in simple terms
    suitable for maintenance crew briefings.
    """
    prompt = f"""
Explain to a maintenance technician (non-engineer) WHY this aircraft component
is flagged as high-risk. Use simple language, avoid heavy math.

Key indicators:
- Stress level is {features.get('utilization_ratio',0)*100:.0f}% of material yield strength
- Component has seen {features.get('fatigue_cycles',0):,} fatigue cycles
- Small cracks of {features.get('crack_length_mm',0):.3f} mm detected
- Operating temperature: {features.get('temperature_C',0):.0f}°C
- Component: {features.get('component','N/A')}

Write 2-3 short paragraphs. Use an analogy if helpful.
Start with the most important risk factor.
""".strip()

    return _llm_call(prompt, max_tokens=400)


# ─────────────────────────────────────────────
# 7. Technical Report Generator
# ─────────────────────────────────────────────
def generate_technical_report(model_metrics: dict,
                               dataset_summary: dict,
                               output_path: Optional[str] = None) -> str:
    """
    Generates a complete AI-written technical report in Markdown.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    prompt = f"""
Write a formal Technical Report titled:
"Aircraft Structural Failure Prediction using Machine Learning"

Report Date: {timestamp}

System Performance:
- Random Forest  : Accuracy={model_metrics.get('Random Forest',{}).get('accuracy',0):.3f}, 
                   F1={model_metrics.get('Random Forest',{}).get('f1_score',0):.3f},
                   AUC={model_metrics.get('Random Forest',{}).get('roc_auc',0):.3f}
- XGBoost        : Accuracy={model_metrics.get('XGBoost',{}).get('accuracy',0):.3f},
                   F1={model_metrics.get('XGBoost',{}).get('f1_score',0):.3f},
                   AUC={model_metrics.get('XGBoost',{}).get('roc_auc',0):.3f}
- Neural Network : Accuracy={model_metrics.get('Neural Network',{}).get('accuracy',0):.3f}

Dataset:
- Total samples  : {dataset_summary.get('total_samples', 12000)}
- Failure rate   : {dataset_summary.get('failure_rate', 0):.1f}%
- Components     : {dataset_summary.get('n_components', 8)}
- Materials      : {dataset_summary.get('n_materials', 6)}

Include sections:
1. Abstract
2. Introduction & Problem Statement
3. Methodology (FEA data, ML pipeline)
4. Results & Model Performance
5. Engineering Significance
6. Limitations
7. Conclusion & Future Work

Write in IEEE technical paper style. ~600 words.
""".strip()

    report = _llm_call(prompt, max_tokens=1000)

    if output_path:
        with open(output_path, "w") as f:
            f.write(f"# Technical Report\n\n")
            f.write(f"*Generated: {timestamp}*\n\n")
            f.write("---\n\n")
            f.write(report)
        print(f"✅ Technical report saved to {output_path}")

    return report


# ─────────────────────────────────────────────
# 8. Chatbot Response
# ─────────────────────────────────────────────
def engineering_chatbot(user_query: str,
                         context: Optional[dict] = None) -> str:
    """
    Answers engineering queries in context of the current analysis.
    Used by the Streamlit chatbot widget.
    """
    context_str = ""
    if context:
        context_str = f"\n\nCurrent analysis context:\n{json.dumps(context, indent=2)}"

    prompt = f"""
You are answering a query from an aerospace structural engineer.
Answer concisely and accurately. If the answer requires calculation,
show the key formula and numbers.

Query: {user_query}{context_str}

Answer in 2-4 sentences unless a detailed calculation is needed.
Always cite the relevant aerospace standard (FAR, MIL-SPEC, ASTM, etc.).
""".strip()

    return _llm_call(prompt, max_tokens=500)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "component":           "Wing_Spar",
        "material":            "Al_7075-T6",
        "failure_probability": 0.82,
        "risk_class":          "Critical",
        "von_mises_stress_MPa":420,
        "yield_strength_MPa":  503,
        "utilization_ratio":   0.835,
        "fatigue_cycles":      85000,
        "crack_length_mm":     0.142,
        "temperature_C":       95.0,
        "rul_cycles":          3200,
    }

    print("=" * 60)
    print("  GenAI Engine — Test Run")
    print("=" * 60)
    print("\n📋 Prediction Insight:\n")
    print(generate_prediction_insight(sample))

    print("\n🔧 Maintenance Recommendations:\n")
    print(generate_maintenance_recommendations(
        "Wing_Spar", "Critical", 3200, "Al_7075-T6"))

    print("\n💬 Chatbot Query:\n")
    print(engineering_chatbot(
        "What is the Paris law equation for crack growth?"))
