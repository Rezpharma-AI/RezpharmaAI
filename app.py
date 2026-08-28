import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sqlite3
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# FOLDER SETUP
# ============================================================

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
IMAGES_DIR = BASE_DIR / "images"
TISSUE_DIR = IMAGES_DIR / "tissue"
RADIOLOGY_DIR = IMAGES_DIR / "radiology"


def ensure_dirs():
    for p in [DATA_DIR, PROCESSED_DIR, IMAGES_DIR, TISSUE_DIR, RADIOLOGY_DIR]:
        p.mkdir(parents=True, exist_ok=True)


# ============================================================
# DUMMY KNOWLEDGE BASE GENERATORS
# ============================================================

def make_ddi_db():
    rows = [
        {
            "drug1": "Warfarin",
            "drug2": "Amiodarone",
            "severity": "Major",
            "mechanism": "Amiodarone inhibits CYP2C9 metabolism of warfarin, increasing INR.",
            "management": "Reduce warfarin dose by 30-50%. Monitor INR closely.",
            "onset": "Delayed"
        },
        {
            "drug1": "Warfarin",
            "drug2": "Aspirin",
            "severity": "Major",
            "mechanism": "Pharmacodynamic synergism increases bleeding risk.",
            "management": "Avoid combination if possible. If required, use lowest aspirin dose and monitor bleeding.",
            "onset": "Rapid"
        },
        {
            "drug1": "Simvastatin",
            "drug2": "Amlodipine",
            "severity": "Moderate",
            "mechanism": "Amlodipine may increase simvastatin exposure via CYP3A4 inhibition.",
            "management": "Limit simvastatin to 20 mg/day when used with amlodipine.",
            "onset": "Delayed"
        },
        {
            "drug1": "Digoxin",
            "drug2": "Furosemide",
            "severity": "Major",
            "mechanism": "Furosemide-induced hypokalemia increases digoxin toxicity risk.",
            "management": "Monitor potassium, magnesium, and digoxin level. Replace electrolytes if needed.",
            "onset": "Delayed"
        },
        {
            "drug1": "Metformin",
            "drug2": "Contrast Dye",
            "severity": "Contraindicated",
            "mechanism": "Contrast-induced kidney injury may cause metformin accumulation and lactic acidosis.",
            "management": "Hold metformin before contrast and for 48 hours after. Restart only if renal function is stable.",
            "onset": "Delayed"
        },
        {
            "drug1": "Ondansetron",
            "drug2": "Amiodarone",
            "severity": "Major",
            "mechanism": "Additive QTc prolongation risk.",
            "management": "Avoid if possible. Monitor ECG and electrolytes if combination is necessary.",
            "onset": "Rapid"
        },
        {
            "drug1": "Lisinopril",
            "drug2": "Spironolactone",
            "severity": "Major",
            "mechanism": "Combined RAAS blockade increases hyperkalemia risk.",
            "management": "Monitor potassium and renal function closely. Avoid in severe renal impairment.",
            "onset": "Delayed"
        },
        {
            "drug1": "Ciprofloxacin",
            "drug2": "Tizanidine",
            "severity": "Contraindicated",
            "mechanism": "Ciprofloxacin strongly inhibits CYP1A2, raising tizanidine levels.",
            "management": "Do not use together. Risk of severe hypotension and sedation.",
            "onset": "Rapid"
        },
        {
            "drug1": "Digoxin",
            "drug2": "Amiodarone",
            "severity": "Major",
            "mechanism": "Amiodarone increases digoxin concentration.",
            "management": "Reduce digoxin dose by approximately 50% and monitor digoxin level.",
            "onset": "Delayed"
        },
        {
            "drug1": "Morphine",
            "drug2": "Midazolam",
            "severity": "Major",
            "mechanism": "Additive CNS depression and respiratory depression.",
            "management": "Use only with monitoring. Reduce doses and monitor respiratory status.",
            "onset": "Rapid"
        }
    ]

    return pd.DataFrame(rows)


def make_pk_db():
    rows = [
        {
            "drug": "Vancomycin",
            "vd_l_per_kg": 0.7,
            "renal_fraction": 0.9,
            "half_life_h": 6,
            "tdm_target": "AUC/MIC 400-600 or trough-based monitoring per protocol",
            "metabolism": "Renal elimination",
            "protein_binding": "30-55%",
            "note": "Adjust dose or interval according to renal function and TDM."
        },
        {
            "drug": "Gentamicin",
            "vd_l_per_kg": 0.25,
            "renal_fraction": 0.95,
            "half_life_h": 2.5,
            "tdm_target": "Peak/trough or extended-interval nomogram",
            "metabolism": "Renal elimination",
            "protein_binding": "<10%",
            "note": "High nephrotoxicity and ototoxicity risk. Monitor levels."
        },
        {
            "drug": "Warfarin",
            "vd_l_per_kg": 0.14,
            "renal_fraction": 0.0,
            "half_life_h": 40,
            "tdm_target": "INR usually 2.0-3.0 depending on indication",
            "metabolism": "CYP2C9, CYP3A4, CYP1A2",
            "protein_binding": "99%",
            "note": "Highly protein bound and many interactions."
        },
        {
            "drug": "Phenytoin",
            "vd_l_per_kg": 0.7,
            "renal_fraction": 0.05,
            "half_life_h": 22,
            "tdm_target": "Total 10-20 mcg/mL; free 1-2 mcg/mL",
            "metabolism": "CYP2C9, CYP2C19",
            "protein_binding": "90%",
            "note": "Correct total phenytoin for low albumin or renal failure."
        },
        {
            "drug": "Meropenem",
            "vd_l_per_kg": 0.3,
            "renal_fraction": 0.8,
            "half_life_h": 1,
            "tdm_target": "Time above MIC; consider extended infusion",
            "metabolism": "Renal elimination",
            "protein_binding": "2%",
            "note": "Adjust interval in renal impairment."
        },
        {
            "drug": "Digoxin",
            "vd_l_per_kg": 7.0,
            "renal_fraction": 0.7,
            "half_life_h": 36,
            "tdm_target": "Usually 0.5-0.9 ng/mL for heart failure",
            "metabolism": "Renal elimination",
            "protein_binding": "25%",
            "note": "Narrow therapeutic index. Monitor K+, Mg++, renal function."
        },
        {
            "drug": "Levetiracetam",
            "vd_l_per_kg": 0.5,
            "renal_fraction": 0.66,
            "half_life_h": 7,
            "tdm_target": "Context-dependent; often 10-40 mcg/mL",
            "metabolism": "Renal elimination",
            "protein_binding": "<10%",
            "note": "Adjust dose in renal impairment."
        }
    ]

    return pd.DataFrame(rows)


def make_serum_db():
    np.random.seed(42)
    n = 120
    group = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

    df = pd.DataFrame({
        "ID": [f"P{str(i).zfill(3)}" for i in range(1, n + 1)],
        "GROUP": group,
        "AGE": np.random.randint(35, 75, n),
        "CRP": np.round(np.exp(np.random.normal(1 + 0.8 * group, 0.5, n)), 2),
        "IL6": np.round(np.exp(np.random.normal(0.5 + 1.0 * group, 0.6, n)), 2),
        "TNFa": np.round(np.exp(np.random.normal(0.3 + 0.5 * group, 0.5, n)), 2),
        "ALB": np.round(np.random.normal(4.2 - 0.6 * group, 0.4, n), 2),
        "LDH": np.round(np.random.normal(200 + 60 * group, 30, n), 1)
    })

    return df


def make_tissue_db():
    rows = [
        {
            "ImageFile": "tissue_01.jpg",
            "PatientID": "T001",
            "Biomarker": "HER2",
            "Score": "Negative",
            "GROUP": 0
        },
        {
            "ImageFile": "tissue_02.jpg",
            "PatientID": "T002",
            "Biomarker": "HER2",
            "Score": "Strong (3+)",
            "GROUP": 1
        },
        {
            "ImageFile": "tissue_03.jpg",
            "PatientID": "T003",
            "Biomarker": "Ki-67",
            "Score": "Low (10%)",
            "GROUP": 0
        },
        {
            "ImageFile": "tissue_04.jpg",
            "PatientID": "T004",
            "Biomarker": "Ki-67",
            "Score": "High (85%)",
            "GROUP": 1
        },
        {
            "ImageFile": "tissue_05.jpg",
            "PatientID": "T005",
            "Biomarker": "PD-L1",
            "Score": "Positive",
            "GROUP": 1
        }
    ]

    return pd.DataFrame(rows)


def generate_tissue_images():
    for i in range(1, 6):
        path = TISSUE_DIR / f"tissue_{i:02d}.jpg"
        if not path.exists():
            img = Image.new("RGB", (420, 420), color=(243, 226, 226))
            draw = ImageDraw.Draw(img)

            for _ in range(170):
                x = int(np.random.randint(15, 405))
                y = int(np.random.randint(15, 405))
                r = int(np.random.randint(5, 16))

                if i in [2, 4, 5] and np.random.rand() > 0.35:
                    color = (120, 60, 30)
                else:
                    color = (180, 150, 150)

                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

            img.save(path)


def generate_radiology_image():
    path = RADIOLOGY_DIR / "chest_xray.jpg"
    if not path.exists():
        img = Image.new("RGB", (600, 600), color=(10, 10, 10))
        draw = ImageDraw.Draw(img)

        # Spine
        draw.rectangle([270, 60, 330, 540], fill=(90, 90, 90))

        # Lungs
        draw.ellipse([80, 120, 260, 480], outline=(140, 140, 140), width=6)
        draw.ellipse([340, 120, 520, 480], outline=(140, 140, 140), width=6)

        # Simulated lower-zone edema/opacities
        draw.ellipse([120, 300, 230, 460], fill=(170, 170, 170))
        draw.ellipse([370, 300, 480, 460], fill=(170, 170, 170))

        img = img.filter(ImageFilter.GaussianBlur(radius=4))
        img.save(path)

# ============================================================
# CLOUD AUTO-BUILDER (Fetches real NLM data for the web demo)
# ============================================================
def build_cloud_database():
    """Fetches lightweight NLM data for the Streamlit Cloud web demo."""
    import requests
    import sqlite3
    import time
    
    db_path = DATA_DIR / "cdss.db"
    
    # If a massive local database already exists, skip this (for local users)
    if db_path.exists() and db_path.stat().st_size > 500000:
        return
        
    print("☁️ Cloud Environment Detected. Building lightweight NLM database...")
    
    acute_drugs = ["warfarin", "aspirin", "amiodarone", "digoxin", "furosemide", 
                   "simvastatin", "lisinopril", "metformin", "ondansetron", "ciprofloxacin"]
    
    rxcuis, mapping = [], {}
    for drug in acute_drugs:
        try:
            r = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", params={"name": drug, "search": 2}, timeout=5)
            ids = r.json().get("idGroup", {}).get("rxnormId")
            if ids:
                rxcuis.append(ids[0])
                mapping[ids[0]] = drug.title()
        except: pass
        time.sleep(0.1)
        
    if not rxcuis: return
    
    try:
        r = requests.get(f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={'+'.join(rxcuis)}", timeout=10)
        data = r.json()
    except: return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS ddi_rules (drug1 TEXT, drug2 TEXT, severity TEXT, mechanism TEXT, management TEXT, onset TEXT)")
    
    groups = data.get("interactionTypeGroup", [])
    if isinstance(groups, dict): groups = [groups]
    
    for group in groups:
        types = group.get("interactionType", [])
        if isinstance(types, dict): types = [types]
        for itype in types:
            pairs = itype.get("interactionPair", [])
            if isinstance(pairs, dict): pairs = [pairs]
            for pair in pairs:
                concepts = pair.get("interactionConcept", [])
                if len(concepts) < 2: continue
                d1, d2 = concepts[0].get("minConceptItem", {}).get("rxcui"), concepts[1].get("minConceptItem", {}).get("rxcui")
                if d1 in mapping and d2 in mapping:
                    sev = pair.get("severity", "N")
                    sev_map = {"4": "CONTRAINDICATED", "3": "MAJOR", "2": "MODERATE", "1": "MINOR"}
                    sev_text = sev_map.get(str(sev), "MODERATE")
                    desc = pair.get("description", "NLM DDI")
                    cur.execute("INSERT INTO ddi_rules VALUES (?, ?, ?, ?, ?, ?)", 
                                (mapping[d1], mapping[d2], sev_text, f"NLM DDI: {desc}", "Consult pharmacist.", "Unknown"))
    conn.commit()
    conn.close()
    print("✅ Cloud NLM database built!")

# Call the generators
generate_all_data()
build_cloud_database() # <--- ADD THIS LINE
def generate_all_data():
    ensure_dirs()

    ddi_path = PROCESSED_DIR / "ddi_database.csv"
    if not ddi_path.exists():
        make_ddi_db().to_csv(ddi_path, index=False)

    pk_path = PROCESSED_DIR / "pk_database.csv"
    if not pk_path.exists():
        make_pk_db().to_csv(pk_path, index=False)

    serum_path = DATA_DIR / "serum.csv"
    if not serum_path.exists():
        make_serum_db().to_csv(serum_path, index=False)

    tissue_path = DATA_DIR / "tissue_metadata.csv"
    if not tissue_path.exists():
        make_tissue_db().to_csv(tissue_path, index=False)

    generate_tissue_images()
    generate_radiology_image()


generate_all_data()


# ============================================================
# AI MODEL
# ============================================================

class DeepClinicalNet(nn.Module):
    """
    Advanced Deep Learning Architecture for Tabular Clinical Data.
    Uses BatchNorm to stabilize training and Dropout to prevent overfitting.
    """
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Randomly drops 30% of neurons to prevent overfitting
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="RezpharmaCDSS",
    page_icon="🏥",
    layout="wide"
)

st.markdown(
    """
    <style>
    .severity-contraindicated {
        background-color: #000000;
        color: white;
        padding: 6px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .severity-major {
        background-color: #ff4b4b;
        color: white;
        padding: 6px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .severity-moderate {
        background-color: #ffa500;
        color: black;
        padding: 6px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .severity-minor {
        background-color: #4CAF50;
        color: white;
        padding: 6px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("🏥 RezpharmaCDSS")
st.sidebar.markdown("---")
st.sidebar.info("v4.1 - Clean Basic Structure")
st.sidebar.warning("⚠️ Research prototype only. Not for direct clinical use.")

st.title("🏥 Rezpharma Clinical Decision Support System")
st.markdown(
    "Structured into Clinical/DDI, PK/PD, Serum Biomarker AI, and Imaging/Histology."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🩺 Clinical & DDI",
        "💊 PK/PD Engine",
        "🩸 Serum Biomarkers AI",
        "🧫 Imaging & Histology"
    ]
)


# ============================================================
# HELPERS
# ============================================================

# ============================================================
# HELPERS
# ============================================================

def severity_html(severity):
    sev_class = severity.strip().lower()
    return f'<span class="severity-{sev_class}">{severity.upper()}</span>'


def get_ddi_match(ddi_df, drug_a, drug_b):
    a = drug_a.strip().lower()
    b = drug_b.strip().lower()

    mask = (
        ((ddi_df["drug1"].str.lower() == a) & (ddi_df["drug2"].str.lower() == b)) |
        ((ddi_df["drug1"].str.lower() == b) & (ddi_df["drug2"].str.lower() == a))
    )

    matched = ddi_df[mask]

    if matched.empty:
        return None

    return matched.iloc[0]


def database_has_ddi_data():
    db_path = DATA_DIR / "cdss.db"
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM ddi_rules").fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def get_ddi_match_db(drug_a, drug_b):
    db_path = DATA_DIR / "cdss.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT *
            FROM ddi_rules
            WHERE lower(drug1) = lower(?) AND lower(drug2) = lower(?)
               OR lower(drug1) = lower(?) AND lower(drug2) = lower(?)
            LIMIT 1
            """,
            conn,
            params=(drug_a.strip(), drug_b.strip(), drug_b.strip(), drug_a.strip())
        )
        conn.close()
    except Exception:
        return None

    if df.empty:
        return None

    row = df.iloc[0].to_dict()
    row.setdefault("severity", "MODERATE")
    row.setdefault("mechanism", "")
    row.setdefault("management", "")
    row.setdefault("onset", "Unknown")
    return row



# ============================================================
# TAB 1: CLINICAL & DDI
# ============================================================

with tab1:
    st.subheader("🩺 Clinical Assessment & Drug-Drug Interaction")

    st.markdown("##### A. Vital Signs")

    v1, v2, v3, v4, v5, v6 = st.columns(6)

    hr = v1.number_input("Heart Rate", 0, 250, 85, step=1)
    sbp = v2.number_input("SBP", 0, 300, 120, step=1)
    dbp = v3.number_input("DBP", 0, 200, 80, step=1)
    temp = v4.number_input("Temp °C", 30.0, 45.0, 37.0, step=0.1)
    rr = v5.number_input("Resp Rate", 0, 60, 16, step=1)
    spo2 = v6.number_input("SpO2 %", 0, 100, 98, step=1)

    st.markdown("##### B. Symptoms / Problems")

    symptom_options = [
        "Active bleeding",
        "Syncope / Dizziness",
        "Chest pain",
        "Shortness of breath",
        "Confusion / Delirium",
        "Fever / Sepsis",
        "Low urine output / AKI",
        "Palpitations"
    ]

    selected_symptoms = st.multiselect("Common symptoms/problems", symptom_options)
    unique_symptom = st.text_input("Optional unique symptom/problem")

    symptoms_full = selected_symptoms.copy()
    if unique_symptom.strip():
        symptoms_full.append(unique_symptom.strip())

    st.markdown("##### C. Medications")

    meds_text = st.text_area(
        "Current medications, comma-separated",
        value="Warfarin, Amiodarone, Aspirin, Furosemide, Digoxin, Ondansetron"
    )

    analyze_clinical = st.button("🔍 Analyze Clinical & DDI", type="primary")

    if analyze_clinical:
                       # --- GENERATE CLINICAL REPORT FOR EHR EXPORT ---
        st.markdown("---")
        st.markdown("#### 📋 Clinical Summary Export")
        
        # Safely grab variables (prevents crashes if placed outside the button block)
        ews_val = locals().get('ews', 'N/A')
        symptoms_val = ', '.join(locals().get('symptoms_full', [])) if locals().get('symptoms_full') else 'None reported'
        drugs_val = ', '.join(locals().get('drugs', [])) if locals().get('drugs') else 'None entered'
        
        report_text = "=== REZPHARMA CDSS CLINICAL REPORT ===\n"
        report_text += f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        report_text += "--- VITALS & STATUS ---\n"
        report_text += f"Early Warning Score (EWS): {ews_val}\n"
        report_text += f"Active Symptoms: {symptoms_val}\n\n"
        report_text += "--- MEDICATIONS ---\n"
        report_text += f"{drugs_val}\n\n"
        report_text += "--- DDI ALERTS ---\n"
        
        # Re-run quick check to populate the text report safely
        ddi_summary = []
        if 'drugs' in locals() and len(drugs) > 1:
            use_db = database_has_ddi_data()
            ddi_df = None if use_db else pd.read_csv(PROCESSED_DIR / "ddi_database.csv")
            
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    match = get_ddi_match_db(drugs[i], drugs[j]) if use_db else get_ddi_match(ddi_df, drugs[i], drugs[j])
                    if match is not None:
                        sev = match.get('severity', 'UNKNOWN')
                        mgmt = match.get('management', 'Monitor patient')
                        ddi_summary.append(f"[{sev.upper()}] {drugs[i]} + {drugs[j]}: {mgmt}")
                        
        if ddi_summary:
            report_text += "\n".join(ddi_summary)
        else:
            report_text += "No major interactions detected.\n"
            
        st.download_button(
            label="📥 Download Clinical Summary (.txt)",
            data=report_text,
            file_name=f"cdss_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            key="ehr_export_btn"   # <--- ADD THIS LINE
        )
            
        st.download_button(
            label="📥 Download Clinical Summary (.txt)",
            data=report_text,
            file_name=f"cdss_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

        st.markdown("---")

        # Vital alerts
        ews = 0
        vital_alerts = []

        map_value = dbp + (sbp - dbp) / 3

        if hr < 50 or hr > 110:
            ews += 2
            vital_alerts.append(f"Abnormal heart rate: {hr} bpm")

        if sbp < 90 or sbp > 180:
            ews += 2
            vital_alerts.append(f"Abnormal systolic BP: {sbp} mmHg")

        if map_value < 65:
            ews += 2
            vital_alerts.append(f"MAP is low: {map_value:.1f} mmHg")

        if temp < 35.5 or temp > 38.5:
            ews += 1
            vital_alerts.append(f"Abnormal temperature: {temp} °C")

        if rr > 20 or rr < 8:
            ews += 1
            vital_alerts.append(f"Abnormal respiratory rate: {rr}")

        if spo2 < 92:
            ews += 2
            vital_alerts.append(f"Low SpO2: {spo2}%")

        if ews >= 3:
            st.error(f"🚨 Early warning score: {ews}. Patient may be clinically unstable.")
        else:
            st.success(f"✅ Early warning score: {ews}.")

        for alert in vital_alerts:
            st.warning(alert)

        # Medication list
        cleaned_meds = meds_text.replace("\n", ",")
        drugs = [d.strip() for d in cleaned_meds.split(",") if d.strip()]
        drugs_lower = [d.lower() for d in drugs]

        if len(drugs) == 0:
            st.warning("No medications entered.")

        if len(drugs) >= 5:
            st.warning(f"⚠️ Polypharmacy: {len(drugs)} medications. Higher adverse event risk.")

        # Symptom-drug context
        symptoms_lower = [s.lower() for s in symptoms_full]

        antithrombotic = [
            "warfarin", "aspirin", "clopidogrel", "ticagrelor", "prasugrel",
            "heparin", "enoxaparin", "apixaban", "rivaroxaban",
            "dabigatran", "edoxaban", "fondaparinux"
        ]

        qt_drugs = [
            "amiodarone", "ondansetron", "haloperidol", "azithromycin",
            "levofloxacin", "moxifloxacin", "methadone", "sotalol",
            "dofetilide", "quetiapine", "escitalopram", "citalopram"
        ]

        sedatives = [
            "midazolam", "lorazepam", "diazepam", "morphine",
            "fentanyl", "hydromorphone", "propofol", "quetiapine",
            "haloperidol", "zolpidem"
        ]

        if any("active bleeding" in s for s in symptoms_lower):
            if any(d in antithrombotic for d in drugs_lower):
                st.error("🚨 Active bleeding while on anticoagulant/antiplatelet therapy. Urgent review required.")

        if any("syncope" in s for s in symptoms_lower):
            if any(d in qt_drugs for d in drugs_lower):
                st.warning("⚠️ Syncope/dizziness with QT-prolonging drugs. Consider ECG and QTc assessment.")

        if any("confusion" in s or "delirium" in s for s in symptoms_lower):
            if any(d in sedatives for d in drugs_lower):
                st.warning("⚠️ Confusion/delirium with sedating medications. Consider dose reduction or discontinuation.")

        if any("fever" in s or "sepsis" in s for s in symptoms_lower):
            st.info("ℹ️ Fever/sepsis selected. Consider cultures, lactate, source control, and antimicrobial review.")

        # DDI check
        st.markdown("#### 🚨 Drug-Drug Interaction Results")

        use_db = database_has_ddi_data()
        ddi_df = None

        if use_db:
            st.caption("Using SQLite clinical DDI database.")
        else:
            ddi_df = pd.read_csv(PROCESSED_DIR / "ddi_database.csv")
            if "onset" not in ddi_df.columns:
                ddi_df["onset"] = "Unknown"

        found = False

        if len(drugs) > 1:
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    if use_db:
                        match = get_ddi_match_db(drugs[i], drugs[j])
                    else:
                        match = get_ddi_match(ddi_df, drugs[i], drugs[j])

                    if match is not None:
                        found = True

                        severity_val = str(match["severity"]) if "severity" in match else "MODERATE"
                        mechanism_val = match["mechanism"] if "mechanism" in match else ""
                        management_val = match["management"] if "management" in match else ""
                        onset_val = match["onset"] if "onset" in match else "Unknown"

                        with st.expander(f"**{drugs[i]} + {drugs[j]}** — {severity_val}", expanded=True):
                            st.markdown(severity_html(severity_val), unsafe_allow_html=True)
                            st.write(f"**Mechanism:** {mechanism_val}")
                            st.write(f"**Management:** {management_val}")
                            st.write(f"**Onset:** {onset_val}")

        if not found:
            if use_db:
                st.success("✅ No matching interactions found in the SQLite database.")
            else:
                st.success("✅ No major interactions found in the sample DDI database.")


# ============================================================
# TAB 2: PK/PD ENGINE
# ============================================================

with tab2:
    st.subheader("💊 PK/PD Engine")

    st.markdown("Enter patient covariates to estimate renal function and review PK/PD flags.")

    c1, c2, c3, c4 = st.columns(4)

    age = c1.number_input("Age", 18, 110, 65, step=1)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    weight = c3.number_input("Actual weight kg", 20.0, 300.0, 75.0, step=0.5)
    height_cm = c4.number_input("Height cm", 100.0, 230.0, 170.0, step=0.5)

    c5, c6, c7 = st.columns(3)

    scr = c5.number_input("Serum creatinine mg/dL", 0.0, 15.0, 1.0, step=0.1)
    albumin = c6.number_input("Albumin g/dL", 0.5, 6.0, 4.0, step=0.1)
    crcl_override = c7.number_input("CrCl override mL/min", 0.0, 250.0, 0.0, step=1.0)

    pk_df = pd.read_csv(PROCESSED_DIR / "pk_database.csv")

    selected_drugs = st.multiselect(
        "Select drugs for PK/PD review",
        pk_df["drug"].tolist(),
        default=["Vancomycin", "Phenytoin"]
    )

    calculate_pk = st.button("🧮 Calculate PK/PD", type="primary")

    if calculate_pk:
        st.markdown("---")

        if scr <= 0 and crcl_override <= 0:
            st.error("Serum creatinine must be greater than 0, or provide a CrCl override.")
        else:
            height_in = height_cm / 2.54
            inches_over_60 = max(height_in - 60, 0)

            if sex == "Male":
                ibw = 50 + 2.3 * inches_over_60
            else:
                ibw = 45.5 + 2.3 * inches_over_60

            if weight <= ibw:
                dosing_weight = weight
            elif weight <= ibw * 1.3:
                dosing_weight = weight
            else:
                dosing_weight = ibw + 0.4 * (weight - ibw)

            if crcl_override > 0:
                crcl = crcl_override
            else:
                crcl = ((140 - age) * dosing_weight) / (72 * scr)
                if sex == "Female":
                    crcl = crcl * 0.85

            if crcl < 15:
                renal_status = "Kidney failure"
            elif crcl < 30:
                renal_status = "Severe impairment"
            elif crcl < 60:
                renal_status = "Moderate impairment"
            else:
                renal_status = "Normal / mild"

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("IBW", f"{ibw:.1f} kg")
            m2.metric("Dosing weight", f"{dosing_weight:.1f} kg")
            m3.metric("CrCl", f"{crcl:.1f} mL/min")
            m4.metric("Renal status", renal_status)

            if crcl < 30:
                st.error("🚨 CrCl < 30 mL/min. Major renal dose adjustments may be required.")
            elif crcl < 60:
                st.warning("⚠️ CrCl < 60 mL/min. Review renally cleared drugs.")

            st.markdown("#### Selected Drug PK/PD Profiles")

            if len(selected_drugs) == 0:
                st.info("Select at least one drug.")

            for drug in selected_drugs:
                row_df = pk_df[pk_df["drug"].str.lower() == drug.lower()]

                if row_df.empty:
                    continue

                r = row_df.iloc[0]

                with st.expander(f"**{drug}**", expanded=True):
                    cc1, cc2, cc3 = st.columns(3)

                    cc1.metric("Vd", f"{r['vd_l_per_kg']} L/kg")
                    cc2.metric("Half-life", f"{r['half_life_h']} h")
                    cc3.metric("Renal fraction", f"{int(float(r['renal_fraction']) * 100)}%")

                    st.write(f"**TDM target:** {r['tdm_target']}")
                    st.write(f"**Metabolism:** {r['metabolism']}")
                    st.write(f"**Protein binding:** {r['protein_binding']}")
                    st.write(f"**Note:** {r['note']}")

                    renal_fraction = float(r["renal_fraction"])

                    if renal_fraction >= 0.5 and crcl < 50:
                        st.error(f"⚠️ {drug} is significantly renally cleared. Consider dose or interval adjustment.")

                    if drug.lower() == "phenytoin" and albumin < 4.0:
                        st.info("ℹ️ Phenytoin: low albumin increases free fraction. Consider free phenytoin level or corrected total level.")
                                            # --- ADVANCED PK SIMULATION GRAPH ---
                    if drug.lower() in ["vancomycin", "gentamicin"]:
                        st.markdown("**📈 Predicted 24-Hour Concentration Profile**")
                        
                        # Calculate adjusted half-life based on patient's renal function
                        normal_crcl = 100.0
                        adjusted_t_half = r['half_life_h'] * (normal_crcl / max(crcl, 10.0))
                        k_elim = 0.693 / adjusted_t_half
                        
                        Vd_total = r['vd_l_per_kg'] * dosing_weight
                        dose_mg = 1500 if drug.lower() == "vancomycin" else 500 # Standard demo doses
                        
                        # Simulate 24 hours (q12h dosing -> 2 doses)
                        times = np.linspace(0, 24, 500)
                        conc = np.zeros_like(times)
                        
                        # Simple 1-compartment IV bolus model for 2 doses (at t=0 and t=12)
                        for t_dose in [0, 12]:
                            mask = times >= t_dose
                            t_rel = times[mask] - t_dose
                            conc[mask] += (dose_mg / Vd_total) * np.exp(-k_elim * t_rel)
                            
                        fig_pk, ax_pk = plt.subplots(figsize=(8, 4))
                        ax_pk.plot(times, conc, label="Predicted Serum Concentration", color="purple", linewidth=2)
                        
                        if drug.lower() == "vancomycin":
                            ax_pk.axhline(20, color="r", linestyle="--", alpha=0.7, label="Target Peak (~20 mg/L)")
                            ax_pk.axhline(15, color="g", linestyle="--", alpha=0.7, label="Target Trough (10-15 mg/L)")
                        else:
                            ax_pk.axhline(8, color="r", linestyle="--", alpha=0.7, label="Target Peak (5-10 mg/L)")
                            ax_pk.axhline(2, color="g", linestyle="--", alpha=0.7, label="Target Trough (< 2 mg/L)")
                            
                        ax_pk.set_xlabel("Time (hours)")
                        ax_pk.set_ylabel("Concentration (mg/L)")
                        ax_pk.set_title(f"{drug} PK Profile (Adjusted t½: {adjusted_t_half:.1f}h | CrCl: {crcl:.0f} mL/min)")
                        ax_pk.legend()
                        st.pyplot(fig_pk)

                    if drug.lower() == "digoxin" and crcl < 50:
                        st.warning("⚠️ Digoxin: reduced renal clearance increases toxicity risk. Monitor level and electrolytes.")

                    if drug.lower() == "vancomycin" and crcl < 30:
                        st.error("🚨 Vancomycin: significant renal impairment. Use AUC-based monitoring and consult TDM protocol.")


# ============================================================
# TAB 3: SERUM BIOMARKERS AI
# ============================================================

with tab3:
    st.subheader("🩸 Serum Biomarker AI")

    uploaded = st.file_uploader(
        "Upload serum CSV. It needs a GROUP column with values 0 and 1.",
        type=["csv"]
    )

    if uploaded is not None:
        df_serum = pd.read_csv(uploaded)
        df_serum.columns = df_serum.columns.str.strip()
        st.success("Custom dataset loaded.")
    else:
        df_serum = pd.read_csv(DATA_DIR / "serum.csv")
        st.info("Using generated dummy serum dataset.")

    st.dataframe(df_serum.head())

    biomarkers = [
        c for c in df_serum.select_dtypes(include=[np.number]).columns
        if c.upper() != "GROUP" and "ID" not in c.upper()
    ]

    if "GROUP" not in df_serum.columns:
        st.error("Dataset must contain a GROUP column.")
    elif len(biomarkers) == 0:
        st.error("No numeric biomarker columns found.")
    else:
        if st.button("🚀 Train AI Model", key="train_ai"):
            y = pd.to_numeric(df_serum["GROUP"], errors="coerce").fillna(0).astype(int).values

            if len(np.unique(y)) < 2:
                st.error("GROUP must contain both 0 and 1.")
            else:
                with st.spinner("Training model..."):
                    X_df = df_serum[biomarkers].apply(pd.to_numeric, errors="coerce")
                    medians = X_df.median()
                    X = X_df.fillna(medians).values

                    try:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                    except ValueError:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                    scaler = StandardScaler().fit(X_tr)
                    X_tr_s = scaler.transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    lr = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)

                    model = SimpleNN(X_tr_s.shape[1])
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    loss_fn = nn.BCELoss()

                    X_t = torch.tensor(X_tr_s, dtype=torch.float32)
                    y_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)

                    for _ in range(120):
                        optimizer.zero_grad()
                        loss = loss_fn(model(X_t), y_t)
                        loss.backward()
                        optimizer.step()

                    model.eval()

                    with torch.no_grad():
                        dl_probs = model(torch.tensor(X_te_s, dtype=torch.float32)).numpy().flatten()

                    lr_probs = lr.predict_proba(X_te_s)[:, 1]

                    st.session_state["trained_model"] = model
                    st.session_state["scaler"] = scaler
                    st.session_state["biomarkers"] = biomarkers
                    st.session_state["defaults"] = medians.to_dict()

                    try:
                        auc_lr = roc_auc_score(y_te, lr_probs)
                        auc_dl = roc_auc_score(y_te, dl_probs)

                        fpr_lr, tpr_lr, _ = roc_curve(y_te, lr_probs)
                        fpr_dl, tpr_dl, _ = roc_curve(y_te, dl_probs)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(fpr_lr, tpr_lr, "--", label=f"Baseline LR AUC = {auc_lr:.2f}")
                        ax.plot(fpr_dl, tpr_dl, "r-", linewidth=2, label=f"Neural Net AUC = {auc_dl:.2f}")
                        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve")
                        ax.legend()
                        st.pyplot(fig)
                    except ValueError:
                        st.warning("Could not compute ROC curve because test set contains only one class.")

                    importance_df = pd.DataFrame({
                        "Biomarker": biomarkers,
                        "Importance": np.abs(lr.coef_[0])
                    }).sort_values("Importance", ascending=False)

                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=importance_df, x="Importance", y="Biomarker", color="#2ca02c", ax=ax2)
                    ax2.set_title("Biomarker Importance from Logistic Regression")
                    st.pyplot(fig2)

        if "trained_model" in st.session_state:
            st.markdown("---")
            st.markdown("#### 🧮 Single Patient Prediction")

            inputs = {}
            cols = st.columns(3)

            for i, bio in enumerate(st.session_state["biomarkers"]):
                with cols[i % 3]:
                    default_val = float(st.session_state["defaults"].get(bio, 0.0))
                    inputs[bio] = st.number_input(bio, value=default_val)

            if st.button("Calculate Risk", key="calculate_risk"):
                input_df = pd.DataFrame([inputs])
                input_df = input_df[st.session_state["biomarkers"]]

                X_scaled = st.session_state["scaler"].transform(input_df.values.astype(float))

                st.session_state["trained_model"].eval()

                with torch.no_grad():
                    prob = st.session_state["trained_model"](
                        torch.tensor(X_scaled, dtype=torch.float32)
                    ).item()

                st.metric("Predicted probability", f"{prob:.1%}")

                if prob > 0.7:
                    st.error("High risk")
                elif prob > 0.4:
                    st.warning("Intermediate risk")
                else:
                    st.success("Low risk")


# ============================================================
# TAB 4: IMAGING & HISTOLOGY
# ============================================================

with tab4:
    st.subheader("🧫 Imaging & Histology")

    mode = st.radio(
        "Select module",
        ["Radiology", "Histology"],
        horizontal=True
    )

    if mode == "Radiology":
        st.markdown("#### 🫁 Radiology AI Demo")

        uploaded_img = st.file_uploader(
            "Upload chest X-ray image (optional)",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_img is not None:
            st.image(uploaded_img, caption="Uploaded image", width="stretch")
        else:
            cxr_path = RADIOLOGY_DIR / "chest_xray.jpg"
            if cxr_path.exists():
                st.image(str(cxr_path), caption="Generated demo chest X-ray", width="stretch")

        st.markdown("#### Simulated AI Findings")

        r1, r2, r3 = st.columns(3)

        r1.metric("Pulmonary edema", "82%")
        r2.metric("Pneumonia", "34%")
        r3.metric("Pleural effusion", "57%")

        st.warning(
            "⚠️ Simulated radiology output. In production this should connect to a validated imaging model."
        )

    else:
        st.markdown("#### 🔬 Histology / IHC Viewer")

        tissue_meta_path = DATA_DIR / "tissue_metadata.csv"

        if tissue_meta_path.exists():
            df_tissue = pd.read_csv(tissue_meta_path)

            selected_file = st.selectbox(
                "Select tissue sample",
                df_tissue["ImageFile"].tolist()
            )

            img_path = TISSUE_DIR / selected_file

            if img_path.exists():
                meta = df_tissue[df_tissue["ImageFile"] == selected_file].iloc[0]

                c1, c2, c3 = st.columns(3)

                c1.metric("Patient ID", meta["PatientID"])
                c2.metric("Biomarker", meta["Biomarker"])
                c3.metric("Score", meta["Score"])

                st.image(str(img_path), caption=f"Sample: {selected_file}", width="stretch")
            else:
                st.error("Selected tissue image file is missing.")
        else:
            st.warning("Tissue metadata not found.")
