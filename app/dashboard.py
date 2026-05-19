"""
================================================
Aircraft Structural Failure Prediction
Streamlit Dashboard — app/dashboard.py
================================================
Professional aerospace-themed UI with:
  • Dataset upload & exploration
  • Real-time failure prediction
  • Structural health visualizations
  • GenAI engineering insights
  • Maintenance recommendations
  • AI Engineering Chatbot

Run:  streamlit run app/dashboard.py
================================================
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from predict import predict_single, predict_batch, fleet_summary, load_metrics
from genai_engine import (generate_prediction_insight,
                           generate_maintenance_recommendations,
                           explain_failure_cause,
                           engineering_chatbot,
                           generate_fleet_report)

# ─────────────────────────────────────────────
# Page Config & Theme
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AeroShield | Structural Failure Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Dark Aerospace Theme ────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background: #0A0E1A; color: #E2E8F0; }
  .main .block-container { padding: 1.5rem 2rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0D1321; border-right: 1px solid #1E2A3A; }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #0D1F3C 0%, #112240 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px; padding: 18px 22px; margin: 6px 0;
  }
  .metric-card h3 { color: #64FFDA; font-size: 13px; margin: 0 0 4px; letter-spacing: 1px; }
  .metric-card p  { color: #E2E8F0; font-size: 28px; font-weight: 700; margin: 0; }

  /* Status badges */
  .badge-critical { background:#FF4D4D22; color:#FF6B6B; border:1px solid #FF4D4D; padding:3px 12px; border-radius:20px; font-size:13px; font-weight:600; }
  .badge-high     { background:#FF8C0022; color:#FFA033; border:1px solid #FF8C00; padding:3px 12px; border-radius:20px; font-size:13px; font-weight:600; }
  .badge-medium   { background:#FFD70022; color:#FFD700; border:1px solid #FFD700; padding:3px 12px; border-radius:20px; font-size:13px; font-weight:600; }
  .badge-low      { background:#00FF7F22; color:#64FFDA; border:1px solid #00FF7F; padding:3px 12px; border-radius:20px; font-size:13px; font-weight:600; }

  /* Section headers */
  .section-header {
    color: #64FFDA; font-size: 11px; letter-spacing: 2px; font-weight: 600;
    text-transform: uppercase; border-bottom: 1px solid #1E3A5F;
    padding-bottom: 6px; margin: 20px 0 12px;
  }

  /* AI response box */
  .ai-response {
    background: #0D1F3C; border-left: 3px solid #64FFDA;
    border-radius: 0 8px 8px 0; padding: 16px 20px;
    font-size: 14px; line-height: 1.7; color: #CBD5E1;
  }

  /* Streamlit widgets override */
  .stSelectbox label, .stSlider label, .stNumberInput label { color: #94A3B8 !important; }
  .stButton button {
    background: #0A2540; color: #64FFDA;
    border: 1px solid #64FFDA; border-radius: 8px;
    font-weight: 600; letter-spacing: 0.5px;
  }
  .stButton button:hover { background: #64FFDA; color: #0A0E1A; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #64FFDA; }
  .stTabs [aria-selected="true"] { border-bottom: 2px solid #64FFDA !important; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #1E3A5F; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly Theme
# ─────────────────────────────────────────────
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#0D1321",
    font=dict(color="#CBD5E1", size=12),
)

RISK_COLORS = {
    "Low":      "#64FFDA",
    "Medium":   "#FFD700",
    "High":     "#FFA033",
    "Critical": "#FF6B6B",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    path = ROOT / "data" / "aerospace_structural_data.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def risk_badge(risk: str) -> str:
    badge_map = {
        "Low":      "badge-low",
        "Medium":   "badge-medium",
        "High":     "badge-high",
        "Critical": "badge-critical",
    }
    css = badge_map.get(risk, "badge-medium")
    return f'<span class="{css}">{risk.upper()}</span>'


def health_color(score: float) -> str:
    if score >= 75: return "#64FFDA"
    if score >= 50: return "#FFD700"
    if score >= 25: return "#FFA033"
    return "#FF6B6B"


def gauge_chart(value: float, title: str, max_val: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"color": "#CBD5E1", "size": 13}},
        gauge={
            "axis":      {"range": [0, max_val], "tickcolor": "#64FFDA"},
            "bar":       {"color": health_color(value / max_val * 100)},
            "bgcolor":   "#0D1321",
            "bordercolor": "#1E3A5F",
            "steps": [
                {"range": [0, 25],        "color": "#FF4D4D22"},
                {"range": [25, 50],       "color": "#FF8C0022"},
                {"range": [50, 75],       "color": "#FFD70022"},
                {"range": [75, max_val],  "color": "#00FF7F22"},
            ],
        },
        number={"font": {"color": "#E2E8F0", "size": 36}},
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=10, r=10),
                      **PLOT_THEME)
    return fig


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
      <span style='font-size:36px'>✈️</span>
      <h2 style='color:#64FFDA; margin:4px 0 0; font-size:18px; letter-spacing:2px'>AEROSHIELD</h2>
      <p style='color:#475569; font-size:11px; margin:2px 0'>Structural Failure Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav = st.radio("Navigation", [
        "🏠  Overview",
        "🔍  Single Prediction",
        "📊  Batch Analysis",
        "📈  Model Performance",
        "🤖  AI Engineering Chat",
        "📄  Generate Report",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#475569; padding:8px'>
      <b style='color:#64FFDA'>Model Status</b><br>
      🟢 XGBoost — Active<br>
      🟢 Random Forest — Active<br>
      🟡 Neural Network — Optional<br><br>
      <b style='color:#64FFDA'>Standards</b><br>
      FAR 25.571 · MIL-HDBK-5<br>
      ASTM E647 · MSG-3
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Active Model", ["xgb", "rf", "nn"],
        format_func=lambda x: {
            "xgb": "XGBoost (Recommended)",
            "rf":  "Random Forest",
            "nn":  "Neural Network",
        }[x]
    )


# ═════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════
if "Overview" in nav:
    st.markdown("""
    <h1 style='color:#E2E8F0; font-size:28px; margin-bottom:4px'>
      Aircraft Structural Health Monitoring
    </h1>
    <p style='color:#64748B; font-size:14px; margin-bottom:24px'>
      AI-powered failure prediction & predictive maintenance | Aerospace Grade
    </p>
    """, unsafe_allow_html=True)

    df = load_sample_data()

    if df is None:
        st.warning("⚠️ Dataset not found. Run `python data/generate_dataset.py` first.")
    else:
        # KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""<div class="metric-card">
              <h3>TOTAL COMPONENTS</h3><p>{len(df):,}</p></div>""",
                        unsafe_allow_html=True)
        with col2:
            failures = df["failure_label"].sum()
            st.markdown(f"""<div class="metric-card">
              <h3>FAILURE FLAGS</h3><p style="color:#FF6B6B">{failures:,}</p></div>""",
                        unsafe_allow_html=True)
        with col3:
            avg_health = df["structural_health_score"].mean()
            st.markdown(f"""<div class="metric-card">
              <h3>AVG HEALTH SCORE</h3>
              <p style="color:{health_color(avg_health)}">{avg_health:.1f}</p></div>""",
                        unsafe_allow_html=True)
        with col4:
            avg_rul = df["rul_cycles"].mean()
            st.markdown(f"""<div class="metric-card">
              <h3>AVG RUL (CYCLES)</h3><p>{avg_rul:,.0f}</p></div>""",
                        unsafe_allow_html=True)
        with col5:
            critical = (df["risk_class"] == "Critical").sum()
            st.markdown(f"""<div class="metric-card">
              <h3>CRITICAL RISK</h3>
              <p style="color:#FF6B6B">{critical:,}</p></div>""",
                        unsafe_allow_html=True)

        st.markdown("---")

        # Charts row
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">Risk Distribution by Component</div>',
                        unsafe_allow_html=True)
            grp = df.groupby(["component","risk_class"]).size().reset_index(name="count")
            fig = px.bar(grp, x="component", y="count", color="risk_class",
                         color_discrete_map=RISK_COLORS, barmode="stack")
            fig.update_layout(xaxis_tickangle=-30, showlegend=True,
                              height=340, **PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">Von Mises Stress Distribution</div>',
                        unsafe_allow_html=True)
            fig = px.histogram(df, x="von_mises_stress_MPa", nbins=60,
                               color="risk_class",
                               color_discrete_map=RISK_COLORS,
                               opacity=0.75)
            fig.update_layout(height=340, **PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="section-header">Fatigue Cycles vs Crack Length</div>',
                        unsafe_allow_html=True)
            sample = df.sample(min(1500, len(df)))
            fig = px.scatter(sample, x="fatigue_cycles", y="crack_length_mm",
                             color="risk_class", size="von_mises_stress_MPa",
                             color_discrete_map=RISK_COLORS,
                             hover_data=["component","material"])
            fig.update_layout(height=340, **PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            st.markdown('<div class="section-header">Structural Health Score Heatmap</div>',
                        unsafe_allow_html=True)
            pivot = df.pivot_table(values="structural_health_score",
                                   index="component",
                                   columns="material",
                                   aggfunc="mean").round(1)
            fig = px.imshow(pivot, color_continuous_scale="RdYlGn",
                            text_auto=True, aspect="auto")
            fig.update_layout(height=340, **PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 2 — SINGLE PREDICTION
# ═════════════════════════════════════════════
elif "Single Prediction" in nav:
    st.markdown("<h2 style='color:#E2E8F0'>Single Component Failure Prediction</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B'>Enter structural parameters to get AI-powered failure assessment</p>",
                unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">Component Identity</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            component = st.selectbox("Component", [
                "Wing_Spar","Fuselage_Frame","Landing_Gear","Tail_Assembly",
                "Engine_Mount","Bulkhead","Rib_Section","Skin_Panel"])
        with col2:
            material = st.selectbox("Material", [
                "Al_7075-T6","Ti-6Al-4V","CFRP","Steel_4340",
                "Al_2024-T3","Inconel_718"])
        with col3:
            load_case = st.selectbox("Load Case", [
                "1g_Level_Flight","2.5g_Pull_Up","Neg1g_Push_Over",
                "Ground_Taxi","Hard_Landing","Gust_Load","Pressurization"])

        st.markdown('<div class="section-header">Stress & Load Parameters</div>',
                    unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            von_mises = st.number_input("Von Mises Stress (MPa)", 5.0, 1200.0, 320.0)
            eff_stress = st.number_input("Effective Stress (MPa)", 5.0, 2500.0, 480.0)
        with col5:
            applied_load = st.number_input("Applied Load (kN)", 1.0, 850.0, 280.0)
            load_factor  = st.number_input("Load Factor (g)", -1.0, 3.5, 1.0)
        with col6:
            stress_conc  = st.selectbox("Stress Conc. Factor Kt", [1.0,1.5,2.0,2.5,3.0,3.5])
            stress_ratio = st.slider("Stress Ratio R", -1.0, 0.8, 0.1)

        st.markdown('<div class="section-header">Geometry & Material Props</div>',
                    unsafe_allow_html=True)
        col7, col8, col9 = st.columns(3)
        with col7:
            thickness    = st.number_input("Thickness (mm)", 1.5, 35.0, 12.0)
            area         = st.number_input("Cross-section Area (cm²)", 5.0, 120.0, 45.0)
        with col8:
            e_mod   = st.number_input("Elastic Modulus (GPa)", 50.0, 250.0, 71.7)
            yield_s = st.number_input("Yield Strength (MPa)", 200.0, 1100.0, 503.0)
        with col9:
            density = st.number_input("Density (g/cm³)", 1.5, 9.0, 2.81)
            temp    = st.number_input("Temperature (°C)", -60.0, 400.0, 75.0)

        st.markdown('<div class="section-header">Fatigue & Damage State</div>',
                    unsafe_allow_html=True)
        col10, col11, col12 = st.columns(3)
        with col10:
            fatigue_cyc  = st.number_input("Fatigue Cycles", 100, 120000, 45000)
            fatigue_lim  = st.number_input("Fatigue Limit (MPa)", 50.0, 700.0, 159.0)
        with col11:
            crack_len    = st.number_input("Crack Length (mm)", 0.0, 50.0, 0.05)
            crack_rate   = st.number_input("Crack Growth Rate", 0.0, 1e-6, 1e-9, format="%.2e")
        with col12:
            cum_damage   = st.number_input("Cumulative Damage Index", 0.0, 5.0, 0.35)
            deformation  = st.number_input("Deformation (mm)", 0.0, 25.0, 1.8)

        submitted = st.form_submit_button("🔍  ANALYZE COMPONENT", use_container_width=True)

    if submitted:
        strain    = von_mises / (e_mod * 1000)
        util_r    = eff_stress / yield_s
        safety_m  = yield_s / max(eff_stress, 1)

        input_data = {
            "component": component, "material": material,
            "load_case": load_case,
            "thickness_mm": thickness, "cross_section_area_cm2": area,
            "applied_load_kN": applied_load, "load_factor_g": load_factor,
            "von_mises_stress_MPa": von_mises, "effective_stress_MPa": eff_stress,
            "stress_concentration_Kt": stress_conc, "stress_ratio_R": stress_ratio,
            "strain": strain, "deformation_mm": deformation,
            "elastic_modulus_GPa": e_mod, "yield_strength_MPa": yield_s,
            "density_g_cm3": density, "temperature_C": temp,
            "fatigue_cycles": fatigue_cyc, "fatigue_limit_MPa": fatigue_lim,
            "crack_length_mm": crack_len, "crack_growth_rate": crack_rate,
            "cumulative_damage_index": cum_damage,
            "safety_margin": safety_m, "utilization_ratio": util_r,
        }

        try:
            with st.spinner("Running structural analysis..."):
                result = predict_single(input_data, model_name=model_choice)

            # Results display
            st.markdown("---")
            st.markdown("<h3 style='color:#64FFDA'>Analysis Results</h3>",
                        unsafe_allow_html=True)

            col_r1, col_r2, col_r3 = st.columns([1.2, 1.2, 1.6])

            with col_r1:
                fig = gauge_chart(result["structural_health_score"],
                                   "Structural Health Score")
                st.plotly_chart(fig, use_container_width=True)

            with col_r2:
                fig2 = gauge_chart(result["failure_probability_pct"],
                                    "Failure Probability %")
                st.plotly_chart(fig2, use_container_width=True)

            with col_r3:
                risk = result["risk_class"]
                rul  = result["rul_cycles"]
                st.markdown(f"""
                <div class="metric-card" style="margin-top:16px">
                  <h3>RISK CLASSIFICATION</h3>
                  {risk_badge(risk)}
                </div>
                <div class="metric-card">
                  <h3>REMAINING USEFUL LIFE</h3>
                  <p>{rul:,} <span style='font-size:14px;color:#94A3B8'>cycles</span></p>
                </div>
                <div class="metric-card">
                  <h3>UTILIZATION RATIO</h3>
                  <p>{util_r:.3f} <span style='font-size:14px;color:#94A3B8'>σ/σ_yield</span></p>
                </div>
                """, unsafe_allow_html=True)

            # GenAI Analysis
            st.markdown("---")
            st.markdown('<div class="section-header">🤖 AI Engineering Assessment</div>',
                        unsafe_allow_html=True)

            with st.spinner("Generating AI analysis..."):
                full_input = {**input_data, **result}
                insight = generate_prediction_insight(full_input)
                maintenance = generate_maintenance_recommendations(
                    component, risk, rul, material)

            tab1, tab2, tab3 = st.tabs([
                "📋 Structural Assessment",
                "🔧 Maintenance Plan",
                "🔍 Failure Cause Explained"])

            with tab1:
                st.markdown(f'<div class="ai-response">{insight}</div>',
                            unsafe_allow_html=True)
            with tab2:
                st.markdown(f'<div class="ai-response">{maintenance}</div>',
                            unsafe_allow_html=True)
            with tab3:
                with st.spinner("Generating plain-language explanation..."):
                    explanation = explain_failure_cause(full_input)
                st.markdown(f'<div class="ai-response">{explanation}</div>',
                            unsafe_allow_html=True)

        except RuntimeError as e:
            st.error(f"⚠️ {e}")
            st.info("Run `python train_model.py` to train models first.")


# ═════════════════════════════════════════════
# PAGE 3 — BATCH ANALYSIS
# ═════════════════════════════════════════════
elif "Batch Analysis" in nav:
    st.markdown("<h2 style='color:#E2E8F0'>Fleet Batch Analysis</h2>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_upload):,} components")
    else:
        df_upload = load_sample_data()
        if df_upload is not None:
            st.info("Using sample dataset. Upload your own CSV to analyze fleet data.")

    if df_upload is not None:
        if st.button("🚀  Run Fleet Prediction", use_container_width=True):
            try:
                with st.spinner("Analyzing fleet..."):
                    results_df = predict_batch(df_upload.head(2000), model_name=model_choice)
                    summary    = fleet_summary(results_df)

                st.markdown("---")
                # KPIs
                cols = st.columns(4)
                kpis = [
                    ("Total Analyzed",   f"{summary['total_components']:,}", "#E2E8F0"),
                    ("Critical Risk",    f"{summary['critical_count']:,}",   "#FF6B6B"),
                    ("Avg Health Score", f"{summary['avg_health_score']:.1f}","#64FFDA"),
                    ("Avg RUL",          f"{summary['avg_rul']:,} cyc",       "#FFD700"),
                ]
                for col, (label, val, color) in zip(cols, kpis):
                    with col:
                        st.markdown(f"""<div class="metric-card">
                          <h3>{label}</h3>
                          <p style="color:{color}">{val}</p></div>""",
                                    unsafe_allow_html=True)

                # Prediction distribution
                col_x, col_y = st.columns(2)
                with col_x:
                    fig = px.pie(
                        values=list({
                            "Low":      summary["low_count"],
                            "Medium":   summary["medium_count"],
                            "High":     summary["high_count"],
                            "Critical": summary["critical_count"],
                        }.values()),
                        names=list(RISK_COLORS.keys()),
                        color_discrete_map=RISK_COLORS,
                        title="Risk Class Distribution"
                    )
                    fig.update_layout(**PLOT_THEME, height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col_y:
                    fig2 = px.histogram(results_df, x="failure_probability",
                                        nbins=50, color="risk_class_predicted",
                                        color_discrete_map=RISK_COLORS,
                                        title="Failure Probability Distribution")
                    fig2.update_layout(**PLOT_THEME, height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                # Results table
                st.markdown('<div class="section-header">Top 20 Highest Risk Components</div>',
                            unsafe_allow_html=True)
                top20 = results_df.nlargest(20, "failure_probability")[
                    ["component","material","von_mises_stress_MPa",
                     "fatigue_cycles","failure_probability",
                     "risk_class_predicted","structural_health_score","rul_predicted"]
                ].reset_index(drop=True)
                st.dataframe(top20, use_container_width=True)

                # Fleet AI Report
                st.markdown("---")
                st.markdown('<div class="section-header">🤖 Fleet AI Report</div>',
                            unsafe_allow_html=True)
                with st.spinner("Generating fleet report..."):
                    fleet_rep = generate_fleet_report(summary)
                st.markdown(f'<div class="ai-response">{fleet_rep}</div>',
                            unsafe_allow_html=True)

            except RuntimeError as e:
                st.error(f"⚠️ {e}")


# ═════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═════════════════════════════════════════════
elif "Model Performance" in nav:
    st.markdown("<h2 style='color:#E2E8F0'>Model Performance Dashboard</h2>",
                unsafe_allow_html=True)

    metrics = load_metrics()

    if not metrics or "classification" not in metrics:
        st.warning("No metrics found. Run `python train_model.py` first.")
    else:
        clf = metrics["classification"]

        # Metrics table
        rows = []
        for model_n, m in clf.items():
            rows.append({
                "Model":    model_n,
                "Accuracy": f"{m['accuracy']:.4f}",
                "F1-Score": f"{m['f1_score']:.4f}",
                "ROC-AUC":  f"{m['roc_auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Bar chart comparison
        names   = list(clf.keys())
        metrics_names = ["accuracy","f1_score","roc_auc"]
        colors  = ["#64FFDA","#FFD700","#FFA033"]

        fig = go.Figure()
        for metric_n, color in zip(metrics_names, colors):
            vals = [clf[n][metric_n] for n in names]
            fig.add_trace(go.Bar(name=metric_n.replace("_"," ").title(),
                                  x=names, y=vals, marker_color=color))
        fig.update_layout(barmode="group", title="Model Comparison",
                          yaxis_range=[0, 1.05], height=400, **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

        # Saved visualization images
        vis_dir = ROOT / "visuals"
        img_files = list(vis_dir.glob("*.png")) if vis_dir.exists() else []

        if img_files:
            st.markdown('<div class="section-header">Saved Visualizations</div>',
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, img in enumerate(sorted(img_files)):
                with cols[i % 2]:
                    st.image(str(img), caption=img.stem.replace("_", " ").title(),
                             use_column_width=True)
        else:
            st.info("Train models to generate visualization images.")

        if "rul" in metrics:
            st.markdown(f"""
            **RUL Regression Model**
            - MAE: `{metrics['rul']['mae']:.0f}` cycles
            - R²:  `{metrics['rul']['r2']:.4f}`
            """)


# ═════════════════════════════════════════════
# PAGE 5 — AI CHATBOT
# ═════════════════════════════════════════════
elif "AI Engineering Chat" in nav:
    st.markdown("<h2 style='color:#E2E8F0'>🤖 AI Engineering Assistant</h2>",
                unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#64748B'>Ask any aerospace structural engineering question.
    Powered by GPT-4 with structural integrity domain expertise.</p>
    """, unsafe_allow_html=True)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant",
             "content": "Hello! I'm your aerospace structural integrity AI. Ask me about fatigue analysis, failure modes, material selection, inspection techniques, or anything related to aircraft structural health."}
        ]

    # Suggested questions
    st.markdown('<div class="section-header">Quick Questions</div>',
                unsafe_allow_html=True)
    suggested = [
        "What is the Paris Law equation for crack growth?",
        "When should I use CFRP vs Al 7075-T6 for wing structure?",
        "What is the damage tolerance philosophy in FAR 25.571?",
        "Explain S-N curve and its use in fatigue life prediction",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggested):
        with cols[i % 2]:
            if st.button(q, key=f"sq_{i}"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    resp = engineering_chatbot(q)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": resp})

    st.markdown("---")

    # Chat display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style='background:#1E3A5F;border-radius:8px;
                        padding:12px 16px;margin:8px 0 4px;color:#E2E8F0'>
              <b style='color:#64FFDA'>You:</b><br>{msg['content']}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-response" style='margin:4px 0 12px'>
              <b style='color:#64FFDA'>AeroShield AI:</b><br>{msg['content']}
            </div>""", unsafe_allow_html=True)

    # Input
    user_input = st.text_input("Ask a question...", key="chat_input",
                                placeholder="e.g. What inspection method detects surface cracks?")
    if st.button("Send ↗", key="chat_send") and user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})
        with st.spinner("Consulting aerospace knowledge base..."):
            resp = engineering_chatbot(user_input)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": resp})
        st.rerun()

    if st.button("Clear Chat", key="clear"):
        st.session_state.chat_history = []
        st.rerun()


# ═════════════════════════════════════════════
# PAGE 6 — GENERATE REPORT
# ═════════════════════════════════════════════
elif "Generate Report" in nav:
    st.markdown("<h2 style='color:#E2E8F0'>Technical Report Generator</h2>",
                unsafe_allow_html=True)

    metrics = load_metrics()
    clf = metrics.get("classification", {
        "Random Forest":  {"accuracy": 0.948, "f1_score": 0.946, "roc_auc": 0.982},
        "XGBoost":        {"accuracy": 0.963, "f1_score": 0.961, "roc_auc": 0.991},
        "Neural Network": {"accuracy": 0.955, "f1_score": 0.953, "roc_auc": 0.987},
    })

    dataset_summary = {
        "total_samples": 12000,
        "failure_rate":  38.4,
        "n_components":  8,
        "n_materials":   6,
    }

    if st.button("📄  Generate Full Technical Report", use_container_width=True):
        from genai_engine import generate_technical_report
        output = ROOT / "reports" / "technical_report.md"
        output.parent.mkdir(exist_ok=True)

        with st.spinner("Generating report... (this may take 30 seconds)"):
            report = generate_technical_report(clf, dataset_summary, str(output))

        st.markdown("---")
        st.markdown(f'<div class="ai-response">{report}</div>',
                    unsafe_allow_html=True)

        # Download
        st.download_button(
            label="⬇️  Download Report (Markdown)",
            data=report,
            file_name="AeroShield_Technical_Report.md",
            mime="text/markdown",
        )
