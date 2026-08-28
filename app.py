import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sqlite3
import requests
import time
from pathlib import Path
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

ensure_dirs()

# ============================================================
# DUMMY KNOWLEDGE BASE GENERATORS
# ============================================================
def make_ddi_db():
    rows = [
        {"drug1": "Warfarin", "drug2": "Amiodarone", "severity": "Major", "mechanism": "Amiodarone inhibits CYP2C9 metabolism of warfarin, increasing INR.", "management": "Reduce warfarin dose by 30-50%. Monitor INR closely.", "onset": "Delayed"},
        {"drug1": "Warfarin", "drug2": "Aspirin", "severity": "Major", "mechanism": "Pharmacodynamic synergism increases bleeding risk.", "management": "Avoid combination if possible. If required, use lowest aspirin dose and monitor bleeding.", "onset": "Rapid"},
        {"drug1": "Simvastatin", "drug2": "Amlodipine", "severity": "Moderate", "mechanism": "Amlodipine may increase simvastatin exposure via CYP3A4 inhibition.", "management": "Limit simvastatin to 20 mg/day when used with amlodipine.", "onset": "Delayed"},
        {"drug1": "Digoxin", "drug2": "Furosemide", "severity": "Major", "mechanism": "Furosemide-induced hypokalemia increases digoxin toxicity risk.", "management": "Monitor potassium, magnesium, and digoxin level. Replace electrolytes if needed.", "onset": "Delayed"},
        {"drug1": "Metformin", "drug2": "Contrast Dye", "severity": "Contraindicated", "mechanism": "Contrast-induced kidney injury may cause metformin accumulation and lactic acidosis.", "management": "Hold metformin before contrast and for 48 hours after. Restart only if renal function is stable.", "onset": "Delayed"},
        {"drug1": "Ondansetron", "drug2": "Amiodarone", "severity": "Major", "mechanism": "Additive QTc prolongation risk.", "management": "Avoid if possible. Monitor ECG and electrolytes if combination is necessary.", "onset": "Rapid"},
        {"drug1": "Lisinopril", "drug2": "Spironolactone", "severity": "Major", "mechanism": "Combined RAAS blockade increases hyperkalemia risk.", "management": "Monitor potassium and renal function closely. Avoid in severe renal impairment.", "onset": "Delayed"},
        {"drug1": "Ciprofloxacin", "drug2": "Tizanidine", "severity": "Contraindicated", "mechanism": "Ciprofloxacin strongly inhibits CYP1A2, raising tizanidine levels.", "management": "Do not use together. Risk of severe hypotension and sedation.", "onset": "Rapid"},
        {"drug1": "Digoxin", "drug2": "Amiodarone", "severity": "Major", "mechanism": "Amiodarone increases digoxin concentration.", "management": "Reduce digoxin dose by approximately 50% and monitor digoxin level.", "onset": "Delayed"},
        {"drug1": "Morphine", "drug2": "Midazolam", "severity": "Major", "mechanism": "Additive CNS depression and respiratory depression.", "management": "Use only with monitoring. Reduce doses and monitor respiratory status.", "onset": "Rapid"}
    ]
    return pd.DataFrame(rows)

def make_pk_db():
    rows = [
        {"drug": "Vancomycin", "vd_l_per_kg": 0.7, "renal_fraction": 0.9, "half_life_h": 6, "tdm_target": "AUC/MIC 400-600", "metabolism": "Renal elimination", "protein_binding": "30-55%", "note": "Adjust dose or interval according to renal function and TDM."},
        {"drug": "Gentamicin", "vd_l_per_kg": 0.25, "renal_fraction": 0.95, "half_life_h": 2.5, "tdm_target": "Peak/trough nomogram", "metabolism": "Renal elimination", "protein_binding": "<10%", "note": "High nephrotoxicity and ototoxicity risk. Monitor levels."},
        {"drug": "Warfarin", "vd_l_per_kg": 0.14, "renal_fraction": 0.0, "half_life_h": 40, "tdm_target": "INR 2.0-3.0", "metabolism": "CYP2C9, CYP3A4", "protein_binding": "99%", "note": "Highly protein bound and many interactions."},
        {"drug": "Phenytoin", "vd_l_per_kg": 0.7, "renal_fraction": 0.05, "half_life_h": 22, "tdm_target": "Total 10-20 mcg/mL", "metabolism": "CYP2C9, CYP2C19", "protein_binding": "90%", "note": "Correct total phenytoin for low albumin or renal failure."},
        {"drug": "Meropenem", "vd_l_per_kg": 0.3, "renal_fraction": 0.8, "half_life_h": 1, "tdm_target": "Time above MIC", "metabolism": "Renal elimination", "protein_binding": "2%", "note": "Adjust interval in renal impairment."},
        {"drug": "Digoxin", "vd_l_per_kg": 7.0, "renal_fraction": 0.7, "half_life_h": 36, "tdm_target": "0.5-0.9 ng/mL", "metabolism": "Renal elimination", "protein_binding": "25%", "note": "Narrow therapeutic index. Monitor K+, Mg++, renal function."}
    ]
    return pd.DataFrame(rows)

def make_serum_db():
    np.random.seed(42)
    n = 120
    group = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    df = pd.DataFrame({
        "ID": [f"P{str(i).zfill(3)}" for i in range(1, n + 1)],
        "GROUP": group, "AGE": np.random.randint(35, 75, n),
        "CRP": np.round(np.exp(np.random.normal(1 + 0.8 * group, 0.5, n)), 2),
        "IL6": np.round(np.exp(np.random.normal(0.5 + 1.0 * group, 0.6, n)), 2),
        "TNFa": np.round(np.exp(np.random.normal(0.3 + 0.5 * group, 0.5, n)), 2),
        "ALB": np.round(np.random.normal(4.2 - 0.6 * group, 0.4, n), 2),
        "LDH": np.round(np.random.normal(200 + 60 * group, 30, n), 1)
    })
    return df

def make_tissue_db():
    rows = [
        {"ImageFile": "tissue_01.jpg", "PatientID": "T001", "Biomarker": "HER2", "Score": "Negative", "GROUP": 0},
        {"ImageFile": "tissue_02.jpg", "PatientID": "T002", "Biomarker": "HER2", "Score": "Strong (3+)", "GROUP": 1},
        {"ImageFile": "tissue_03.jpg", "PatientID": "T003", "Biomarker": "Ki-67", "Score": "Low (10%)", "GROUP": 0},
        {"ImageFile": "tissue_04.jpg", "PatientID": "T004", "Biomarker": "Ki-67", "Score": "High (85%)", "GROUP": 1},
        {"ImageFile": "tissue_05.jpg", "PatientID": "T005", "Biomarker": "PD-L1", "Score": "Positive", "GROUP": 1}
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
                color = (120, 60, 30) if i in [2, 4, 5] and np.random.rand() > 0.35 else (180, 150, 150)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            img.save(path)

def generate_radiology_image():
    path = RADIOLOGY_DIR / "chest_xray.jpg"
    if not path.exists():
        img = Image.new("RGB", (600, 600), color=(10, 10, 10))
        draw = ImageDraw.Draw(img)
        draw.rectangle([270, 60, 330, 540], fill=(90, 90, 90))
        draw.ellipse([80, 120, 260, 480], outline=(140, 140, 140), width=6)
        draw.ellipse([340, 120, 520, 480], outline=(140, 140, 140), width=6)
        draw.ellipse([120, 300, 230, 460], fill=(170, 170, 170))
        draw.ellipse([370, 300, 480, 460], fill=(170, 170, 170))
        img = img.filter(ImageFilter.GaussianBlur(radius=4))
        img.save(path)

def generate_all_data():
    if not (PROCESSED_DIR / "ddi_database.csv").exists(): make_ddi_db().to_csv(PROCESSED_DIR / "ddi_database.csv", index=False)
    if not (PROCESSED_DIR / "pk_database.csv").exists(): make_pk_db().to_csv(PROCESSED_DIR / "pk_database.csv", index=False)
    if not (DATA_DIR / "serum.csv").exists(): make_serum_db().to_csv(DATA_DIR / "serum.csv", index=False)
    if not (DATA_DIR / "tissue_metadata.csv").exists(): make_tissue_db().to_csv(DATA_DIR / "tissue_metadata.csv", index=False)
    generate_tissue_images()
    generate_radiology_image()

generate_all_data()

# ============================================================
# AI MODEL
# ============================================================
class DeepClinicalNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="RezpharmaCDSS", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.severity-contraindicated { background-color: #000000; color: white; padding: 6px; border-radius: 6px; font-weight: bold; display: inline-block; }
.severity-major { background-color: #ff4b4b; color: white; padding: 6px; border-radius: 6px; font-weight: bold; display: inline-block; }
.severity-moderate { background-color: #ffa500; color: black; padding: 6px; border-radius: 6px; font-weight: bold; display: inline-block; }
.severity-minor { background-color: #4CAF50; color: white; padding: 6px; border-radius: 6px; font-weight: bold; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏥 RezpharmaCDSS")
st.sidebar.markdown("---")
st.sidebar.success("🚀 v6.0 - Master Cloud Builder ACTIVE!")
st.sidebar.warning("⚠️ Research prototype only. Not for direct clinical use.")

# ============================================================
# CLOUD AUTO-BUILDER (Runs after page setup)
# ============================================================
def build_cloud_database():
    db_path = DATA_DIR / "cdss.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            count = conn.execute("SELECT COUNT(*) FROM ddi_rules").fetchone()[0]
            conn.close()
            if count > 10:
                st.sidebar.success(f"✅ Cloud DB Active ({count} rules)")
                return
        except: pass

    st.sidebar.info("☁️ Building Cloud Database from NLM API...")
    acute_drugs = ["warfarin", "aspirin", "amiodarone", "digoxin", "furosemide", "simvastatin", "lisinopril", "metformin", "ondansetron", "ciprofloxacin"]
    rxcuis, mapping = [], {}
    for drug in acute_drugs:
        try:
            r = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", params={"name": drug, "search": 2}, timeout=5)
            ids = r.json().get("idGroup", {}).get("rxnormId")
            if ids:
                rxcuis.append(ids[0])
                mapping[ids[0]] = drug.title()
        except Exception as e: st.sidebar.warning(f"API Error: {str(e)[:30]}")
        time.sleep(0.2)
        
    if not rxcuis:
        st.sidebar.error("❌ Could not reach NLM API.")
        return
        
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={'+'.join(rxcuis)}"
        r = requests.get(url, timeout=15)
        data = r.json()
    except Exception as e:
        st.sidebar.error(f"❌ API Fetch Failed")
        return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS ddi_rules (drug1 TEXT, drug2 TEXT, severity TEXT, mechanism TEXT, management TEXT, onset TEXT)")
        cur.execute("DELETE FROM ddi_rules WHERE mechanism LIKE '%NLM DDI%'")
        count = 0
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
                    d1 = concepts[0].get("minConceptItem", {}).get("rxcui")
                    d2 = concepts[1].get("minConceptItem", {}).get("rxcui")
                    if d1 in mapping and d2 in mapping:
                        sev = pair.get("severity", "N")
                        sev_map = {"4": "CONTRAINDICATED", "3": "MAJOR", "2": "MODERATE", "1": "MINOR"}
                        sev_text = sev_map.get(str(sev), "MODERATE")
                        desc = pair.get("description", "NLM DDI")
                        cur.execute("INSERT INTO ddi_rules VALUES (?, ?, ?, ?, ?, ?)", (mapping[d1], mapping[d2], sev_text, f"NLM DDI: {desc}", "Consult pharmacist.", "Unknown"))
                        count += 1
        conn.commit()
        conn.close()
        st.sidebar.success(f"✅ Built Cloud DB! ({count} NLM rules)")
    except Exception as e:
        st.sidebar.error(f"❌ DB Write Failed")

build_cloud_database()

# ============================================================
# HELPERS
# ============================================================
def severity_html(severity):
    sev_class = severity.strip().lower().replace(" ", "")
    return f'<span class="severity-{sev_class}">{severity.upper()}</span>'

def get_ddi_match(ddi_df, drug_a, drug_b):
    a, b = drug_a.strip().lower(), drug_b.strip().lower()
    mask = (((ddi_df["drug1"].str.lower() == a) & (ddi_df["drug2"].str.lower() == b)) | ((ddi_df["drug1"].str.lower() == b) & (ddi_df["drug2"].str.lower() == a)))
    matched = ddi_df[mask]
    return matched.iloc[0] if not matched.empty else None

def database_has_ddi_data():
    db_path = DATA_DIR / "cdss.db"
    if not db_path.exists(): return False
    try:
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM ddi_rules").fetchone()[0]
        conn.close()
        return count > 0
    except: return False

def get_ddi_match_db(drug_a, drug_b):
    db_path = DATA_DIR / "cdss.db"
    if not db_path.exists(): return None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM ddi_rules WHERE lower(drug1) = lower(?) AND lower(drug2) = lower(?) OR lower(drug1) = lower(?) AND lower(drug2) = lower(?) LIMIT 1", conn, params=(drug_a.strip(), drug_b.strip(), drug_b.strip(), drug_a.strip()))
        conn.close()
    except: return None
    if df.empty: return None
    row = df.iloc[0].to_dict()
    row.setdefault("severity", "MODERATE")
    row.setdefault("mechanism", "")
    row.setdefault("management", "")
    row.setdefault("onset", "Unknown")
    return row

st.title("🏥 Rezpharma Clinical Decision Support System")
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Clinical & DDI", "💊 PK/PD Engine", "🩸 Serum Biomarkers AI", "🧫 Imaging & Histology"])

# ============================================================
# TAB 1: CLINICAL & DDI
# ============================================================
with tab1:
    st.subheader("🩺 Clinical Assessment & Drug-Drug Interaction")
    st.markdown("##### A. Vital Signs")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    hr, sbp, dbp = v1.number_input("HR", 0, 250, 85), v2.number_input("SBP", 0, 300, 120), v3.number_input("DBP", 0, 200, 80)
    temp, rr, spo2 = v4.number_input("Temp", 30.0, 45.0, 37.0), v5.number_input("RR", 0, 60, 16), v6.number_input("SpO2", 0, 100, 98)
    
    st.markdown("##### B. Symptoms / Problems")
    selected_symptoms = st.multiselect("Common symptoms", ["Active bleeding", "Syncope", "Chest pain", "Confusion", "Fever"])
    
    st.markdown("##### C. Medications")
    meds_text = st.text_area("Current medications", value="Warfarin, Amiodarone, Aspirin, Furosemide, Digoxin, Ondansetron")
    analyze_clinical = st.button("🔍 Analyze Clinical & DDI", type="primary")

    if analyze_clinical:
        st.markdown("---")
        ews = 0
        if hr < 50 or hr > 110: ews += 2
        if sbp < 90 or sbp > 180: ews += 2
        if temp < 35.5 or temp > 38.5: ews += 1
        if rr > 20 or rr < 8: ews += 1
        if spo2 < 92: ews += 2
        
        if ews >= 3: st.error(f"🚨 Early warning score: {ews}. Patient may be clinically unstable.")
        else: st.success(f"✅ Early warning score: {ews}.")
        
        cleaned_meds = meds_text.replace("\n", ",")
        drugs = [d.strip() for d in cleaned_meds.split(",") if d.strip()]
        if len(drugs) >= 5: st.warning(f"⚠️ Polypharmacy: {len(drugs)} medications.")
        
        st.markdown("#### 🚨 Drug-Drug Interaction Results")
        use_db = database_has_ddi_data()
        ddi_df = None
        if use_db: st.caption("Using SQLite clinical DDI database.")
        else: ddi_df = pd.read_csv(PROCESSED_DIR / "ddi_database.csv")
        
        found = False
        if len(drugs) > 1:
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    match = get_ddi_match_db(drugs[i], drugs[j]) if use_db else get_ddi_match(ddi_df, drugs[i], drugs[j])
                    if match is not None:
                        found = True
                        severity_val = str(match["severity"])
                        with st.expander(f"**{drugs[i]} + {drugs[j]}** — {severity_val}", expanded=True):
                            st.markdown(severity_html(severity_val), unsafe_allow_html=True)
                            st.write(f"**Mechanism:** {match.get('mechanism', '')}")
                            st.write(f"**Management:** {match.get('management', '')}")
        if not found: st.success("✅ No major interactions found.")

        st.markdown("---")
        report_text = f"=== REZPHARMA CDSS REPORT ===\nEWS: {ews}\nMeds: {', '.join(drugs)}\n"
        st.download_button(label="📥 Download Report", data=report_text, file_name="report.txt", mime="text/plain", key="ehr_export_btn")

# ============================================================
# TAB 2: PK/PD ENGINE
# ============================================================
with tab2:
    st.subheader("💊 PK/PD Engine")
    c1, c2, c3, c4 = st.columns(4)
    age, sex = c1.number_input("Age", 18, 110, 65), c2.selectbox("Sex", ["Male", "Female"])
    weight, height_cm = c3.number_input("Weight kg", 20.0, 300.0, 75.0), c4.number_input("Height cm", 100.0, 230.0, 170.0)
    c5, c6, c7 = st.columns(3)
    scr, albumin = c5.number_input("SCr mg/dL", 0.1, 15.0, 1.0), c6.number_input("Albumin g/dL", 0.5, 6.0, 4.0)
    crcl_override = c7.number_input("CrCl override", 0.0, 250.0, 0.0)
    pk_df = pd.read_csv(PROCESSED_DIR / "pk_database.csv")
    selected_drugs = st.multiselect("Select drugs", pk_df["drug"].tolist(), default=["Vancomycin", "Phenytoin"])
    
    if st.button("🧮 Calculate PK/PD", type="primary"):
        height_in = height_cm / 2.54
        ibw = (50 + 2.3 * (height_in - 60)) if sex == "Male" else (45.5 + 2.3 * (height_in - 60))
        dosing_weight = weight if weight <= ibw else ibw + 0.4 * (weight - ibw)
        crcl = crcl_override if crcl_override > 0 else (((140 - age) * dosing_weight) / (72 * scr)) * (0.85 if sex == "Female" else 1.0)
        st.metric("CrCl", f"{crcl:.1f} mL/min")
        
        for drug in selected_drugs:
            row_df = pk_df[pk_df["drug"].str.lower() == drug.lower()]
            if row_df.empty: continue
            r = row_df.iloc[0]
            with st.expander(f"**{drug}**", expanded=True):
                st.write(f"**Half-life:** {r['half_life_h']} h")
                if drug.lower() in ["vancomycin", "gentamicin"]:
                    adjusted_t_half = r['half_life_h'] * (100.0 / max(crcl, 10.0))
                    k_elim = 0.693 / adjusted_t_half
                    Vd_total = r['vd_l_per_kg'] * dosing_weight
                    dose_mg = 1500 if drug.lower() == "vancomycin" else 500
                    times = np.linspace(0, 24, 500)
                    conc = np.zeros_like(times)
                    for t_dose in [0, 12]:
                        mask = times >= t_dose
                        conc[mask] += (dose_mg / Vd_total) * np.exp(-k_elim * (times[mask] - t_dose))
                    fig_pk, ax_pk = plt.subplots(figsize=(8, 4))
                    ax_pk.plot(times, conc, label="Predicted Concentration", color="purple", linewidth=2)
                    ax_pk.set_title(f"{drug} PK Profile (Adjusted t½: {adjusted_t_half:.1f}h)")
                    st.pyplot(fig_pk)

# ============================================================
# TAB 3: SERUM BIOMARKERS AI
# ============================================================
with tab3:
    st.subheader("🩸 Serum Biomarker AI")
    uploaded = st.file_uploader("Upload serum CSV", type=["csv"])
    df_serum = pd.read_csv(uploaded) if uploaded else pd.read_csv(DATA_DIR / "serum.csv")
    biomarkers = [c for c in df_serum.select_dtypes(include=[np.number]).columns if c.upper() != "GROUP" and "ID" not in c.upper()]
    
    if st.button("🚀 Train AI Model"):
        y = df_serum["GROUP"].values
        X = df_serum[biomarkers].fillna(df_serum[biomarkers].median()).values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        model = DeepClinicalNet(X_tr_s.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()
        for _ in range(50):
            optimizer.zero_grad()
            loss = loss_fn(model(torch.tensor(X_tr_s, dtype=torch.float32)), torch.tensor(y_tr, dtype=torch.float32).view(-1, 1))
            loss.backward()
            optimizer.step()
        st.session_state["trained_model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["biomarkers"] = biomarkers
        st.success("✅ Model Trained!")

# ============================================================
# TAB 4: IMAGING & HISTOLOGY
# ============================================================
with tab4:
    st.subheader("🧫 Imaging & Histology")
    if RADIOLOGY_DIR.joinpath("chest_xray.jpg").exists():
        st.image(str(RADIOLOGY_DIR / "chest_xray.jpg"), use_container_width=True)
    st.warning("⚠️ Simulated radiology output.")