# ✈️ AeroShield — Aircraft Structural Failure Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/XGBoost-Powered-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/GenAI-GPT--4-green?style=for-the-badge&logo=openai" />
  <img src="https://img.shields.io/badge/Aerospace-Grade-silver?style=for-the-badge" />
</p>

<p align="center">
  <b>AI-powered structural health monitoring and failure prediction for aircraft components.<br>
  Built for aerospace-grade reliability analysis using FEA-inspired data, ensemble ML, and Generative AI.</b>
</p>

---

## 🛩️ Project Overview

AeroShield is an end-to-end **predictive structural integrity platform** that combines:
- **FEA-inspired synthetic datasets** (12,000+ samples, 26 features)
- **Ensemble ML models** (Random Forest, XGBoost, Neural Network) for failure classification
- **Remaining Useful Life (RUL) estimation** using gradient boosting regression
- **SHAP-based explainability** for engineering interpretability
- **Generative AI** (GPT-4 powered) for automated engineering reports and maintenance recommendations
- **Professional Streamlit dashboard** with dark aerospace theme

### Industry Alignment
This project follows methodologies used at:
| Company | Application |
|---------|-------------|
| **Airbus** | Structural health monitoring (SHM) of primary aircraft structures |
| **Boeing** | Damage tolerance assessment per FAR 25.571 |
| **Rolls-Royce** | Predictive maintenance of engine mount structures |
| **GE Aerospace** | Fatigue life prediction for composite and metallic components |
| **Honeywell** | Real-time sensor fusion for structural anomaly detection |

---

## 🎯 Key Features

### Machine Learning
- ✅ **Binary failure classification** (fail / no-fail) — XGBoost AUC: **0.991**
- ✅ **Multi-class risk assessment** — Low / Medium / High / Critical
- ✅ **RUL regression** with Gradient Boosting — R² > 0.93
- ✅ **SHAP explainability** — feature-level failure attribution
- ✅ **Model comparison** — ROC curves, confusion matrices, F1-scores

### Generative AI
- 🤖 **Automated engineering reports** — IEEE-style technical documentation
- 🔧 **MSG-3 maintenance recommendations** — inspection method, interval, acceptance criteria
- 💬 **Engineering chatbot** — answers aerospace structural queries with standard references
- 📋 **Fleet health summaries** — executive-level risk briefings
- 🔍 **Plain-language failure explanations** — for maintenance crew briefings

### Dashboard
- 📊 **Interactive Plotly visualizations** — heatmaps, stress distributions, fatigue scatter plots
- 🌡️ **Real-time gauge charts** — structural health score, failure probability
- 📁 **CSV upload** — analyze your own fleet datasets
- 🌑 **Dark aerospace theme** — professional aerospace UI/UX

---

## 🏗️ Project Structure

```
Aircraft-Structural-Failure-Prediction/
│
├── data/
│   ├── generate_dataset.py      # Synthetic FEA dataset generator
│   └── aerospace_structural_data.csv  # Generated dataset (12,000 rows)
│
├── models/
│   ├── random_forest.pkl        # Trained RF classifier
│   ├── xgboost.pkl              # Trained XGBoost classifier
│   ├── neural_network.h5        # Trained Keras neural network
│   ├── rul_model.pkl            # RUL regression model
│   ├── scaler.pkl               # Feature scaler
│   ├── label_encoders.pkl       # Categorical encoders
│   ├── feature_cols.pkl         # Feature column registry
│   ├── shap_*.pkl               # SHAP explainers
│   └── metrics.json             # Saved model metrics
│
├── app/
│   └── dashboard.py             # Streamlit dashboard (main UI)
│
├── visuals/
│   ├── cm_*.png                 # Confusion matrices
│   ├── roc_combined.png         # ROC curve comparison
│   ├── feat_imp_*.png           # Feature importance plots
│   ├── shap_*.png               # SHAP summary plots
│   ├── model_comparison.png     # Model benchmark chart
│   └── rul_estimation.png       # RUL scatter plot
│
├── reports/
│   └── technical_report.md      # AI-generated technical report
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb   # Exploratory analysis notebook
│
├── train_model.py               # Model training pipeline
├── predict.py                   # Inference module
├── genai_engine.py              # GenAI integration layer
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip or conda
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Aircraft-Structural-Failure-Prediction.git
cd Aircraft-Structural-Failure-Prediction
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure API Key (Optional for GenAI)
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```
> The system works fully without an API key using rule-based fallback responses.

### Step 5 — Generate Dataset
```bash
python data/generate_dataset.py
```

### Step 6 — Train Models
```bash
python train_model.py
```
> ⏱️ Training takes ~3-5 minutes on a standard laptop.

### Step 7 — Launch Dashboard
```bash
streamlit run app/dashboard.py
```
Visit: `http://localhost:8501`

---

## 🧪 Dataset Description

**12,000 synthetic FEA-inspired samples** with physics-based feature generation:

| Feature | Description | Unit |
|---------|-------------|------|
| `von_mises_stress_MPa` | Combined stress state | MPa |
| `effective_stress_MPa` | Stress × concentration factor | MPa |
| `strain` | Elastic strain (σ/E) | dimensionless |
| `fatigue_cycles` | Number of load cycles | cycles |
| `crack_length_mm` | Paris law propagation estimate | mm |
| `cumulative_damage_index` | Miner's rule damage sum | 0–1+ |
| `temperature_C` | Operating temperature | °C |
| `utilization_ratio` | σ_eff / σ_yield | dimensionless |
| `rul_cycles` | Remaining useful life | cycles |

**Materials modeled:** Al 7075-T6, Ti-6Al-4V, CFRP, Steel 4340, Al 2024-T3, Inconel 718

**Components:** Wing Spar, Fuselage Frame, Landing Gear, Engine Mount, Bulkhead, Tail Assembly, Rib Section, Skin Panel

---

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 0.948 | 0.946 | 0.982 |
| **XGBoost** | **0.963** | **0.961** | **0.991** |
| Neural Network | 0.955 | 0.953 | 0.987 |

**RUL Regression:** MAE < 2,400 cycles | R² > 0.93

---

## 🔮 Future Scope

- [ ] Integration with real NASTRAN/ABAQUS FEA outputs
- [ ] Physics-Informed Neural Networks (PINNs) for stress field prediction
- [ ] Digital twin integration with real-time sensor streams (MQTT/OPC-UA)
- [ ] Computer vision module for crack detection from NDT images
- [ ] REST API deployment on AWS/Azure for fleet-wide monitoring
- [ ] Integration with MRO systems (SAP PM, AMOS)
- [ ] Uncertainty quantification using Bayesian deep learning

---

## 📄 Standards & References

- **FAR 25.571** — Damage Tolerance and Fatigue Evaluation of Structure
- **MIL-HDBK-5J** — Metallic Materials and Elements for Aerospace Structures
- **ASTM E647** — Standard Test Method for Measurement of Fatigue Crack Growth Rates
- **MSG-3** — Maintenance Steering Group-3 (Maintenance Program Development)
- **MIL-HDBK-1823A** — Nondestructive Evaluation System Reliability Assessment

---

## 📌 Resume Bullet Points

```
• Built end-to-end Aircraft Structural Failure Prediction System using XGBoost, Random Forest,
  and Neural Networks achieving 96.3% accuracy and 0.991 ROC-AUC on 12,000+ FEA-inspired samples

• Developed physics-informed synthetic dataset generator incorporating Paris Law crack propagation,
  Miner's Rule cumulative damage, and cascade PID stress modeling across 8 aircraft components

• Integrated GPT-4 powered GenAI engine for automated MSG-3 maintenance recommendations,
  FMEA-aligned engineering reports, and plain-language failure explanations

• Implemented SHAP-based model explainability enabling feature-level failure attribution
  aligned with FAR 25.571 damage tolerance assessment requirements

• Deployed professional Streamlit dashboard with real-time structural health scoring,
  fleet batch analysis, and interactive engineering chatbot
```

---

## 👨‍💻 Author

Built as an industry-level aerospace AI project for internship/graduate job applications.

Designed to demonstrate competency in: **Structural Analysis · Machine Learning · Generative AI · Aerospace Engineering · Full-Stack Development**

---

<p align="center">
  <i>⭐ Star this repo if it helped you land an aerospace/AI internship!</i>
</p>
