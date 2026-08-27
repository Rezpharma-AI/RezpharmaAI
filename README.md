# 🏥 Rezpharma: Hybrid AI Clinical Decision Support System (CDSS)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-orange)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey)

> **⚠️ MEDICAL DISCLAIMER:** This project is a research prototype and portfolio piece. It is **NOT** intended for direct clinical diagnosis, treatment, or use in real patient care without regulatory approval (FDA/CE) and validation by a licensed pharmacist or physician.

## 🏗️ Architecture: The Hybrid AI Approach
Modern healthcare AI cannot rely solely on probabilistic machine learning. This CDSS uses a **Hybrid Architecture**:

1. **Deterministic Rule Engine (SQLite):** Queries a relational database of clinical DDI rules, severity grades, and management protocols.
2. **Pharmacokinetic (PK) Engine:** Uses Cockcroft-Gault, IBW, and Adjusted Body Weight formulas to simulate drug clearance and generate time-concentration graphs.
3. **Probabilistic Deep Learning:** A PyTorch Neural Network (with BatchNorm & Dropout) analyzes serum biomarkers (CRP, IL-6, LDH) to predict disease risk.
4. **Multi-Modal Imaging:** Mock AI outputs for Radiology (CXR) and Histology (IHC scoring).

## 📂 Project Structure
```text
RezpharmaCDSS/
├── app.py                  # Main Streamlit Dashboard
├── setup_demo_data.py      # Fetches open-source NLM data to build demo DB
├── scripts/
│   └── parse_drugbank.py   # Advanced parser for full DrugBank XML (Academic use)
├── data/                   # Auto-generated (ignored by Git)
└── requirements.txt