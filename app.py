import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3
import requests
import time
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. PHARMA.AI THEME & CSS INJECTION
# ============================================================
st.set_page_config(page_title="Rezpharma AI", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

pharma_css = """
<style>
    /* Global Background & Text */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        color: #e2e8f0;
    }
    h1, h2, h3, h4 { color: #f8fafc; font-family: 'Inter', sans-serif; font-weight: 600; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 2px solid #0ea5e9;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #38bdf8;
    }
    
    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.1);
    }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #38bdf8; font-weight: 700; }
    
    /* Modern Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.5);
        transform: translateY(-2px);
        color: white;
    }
    
    /* Expander (DDI Alerts) */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: 1px solid #0ea5e9;
    }
    
    /* Alerts Customization */
    div[data-testid="stAlert"] {
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 10px;
        border-left: 5px solid #0ea5e9;
    }
</style>
"""
st.markdown(pharma_css, unsafe_allow_html=True)

# ============================================================
# 2. FOLDER & DATABASE SETUP
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
IMAGES_DIR = BASE_DIR / "images"
TISSUE_DIR = IMAGES_DIR / "tissue"
RADIOLOGY_DIR = IMAGES_DIR / "radiology"

for p in [DATA_DIR, PROCESSED_DIR, IMAGES_DIR, TISSUE_DIR, RADIOLOGY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --- Authentication Database ---
def init_auth_db():
    conn = sqlite3.connect(DATA_DIR / "users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(username, password):
    conn = sqlite3.connect(DATA_DIR / "users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                  (username, hash_password(password), 'clinician'))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."

def login(username, password):
    conn = sqlite3.connect(DATA_DIR / "users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

init_auth_db()

# ============================================================
# 3. AUTHENTICATION GATEKEEPER
# ============================================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #38bdf8; font-size: 4rem;'>🧬 Rezpharma AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #94a3b8;'>Next-Generation Clinical Decision Support</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Login", "📝 Create Account"])
        
        with auth_tab1:
            with st.form("login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login to Platform", use_container_width=True)
                if submit:
                    if login(user, pwd):
                        st.session_state.authenticated = True
                        st.session_state.username = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                        
        with auth_tab2:
            with st.form("signup_form"):
                new_user = st.text_input("Create Username")
                new_pwd = st.text_input("Create Password", type="password")
                submit = st.form_submit_button("Create Secure Account", use_container_width=True)
                if submit:
                    if not new_user or not new_pwd:
                        st.error("Please fill in all fields.")
                    else:
                        success, msg = sign_up(new_user, new_pwd)
                        if success:
                            st.success(msg + " Please login.")
                        else:
                            st.error(msg)
    st.stop()

# ============================================================
# 4. MAIN APP (POST-AUTHENTICATION)
# ============================================================
# Sidebar
st.sidebar.title(f"🧬 Rezpharma AI")
st.sidebar.markdown(f"Welcome, **{st.session_state.username}**")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("v7.0 Enterprise Edition")
st.sidebar.warning("⚠️ Research prototype. Not for direct clinical use without validation.")

# --- Drug Autocomplete List ---
COMMON_DRUGS = sorted(list(set([
    "Warfarin", "Aspirin", "Clopidogrel", "Heparin", "Enoxaparin", "Apixaban", "Rivaroxaban",
    "Metoprolol", "Lisinopril", "Losartan", "Amlodipine", "Atorvastatin", "Simvastatin", "Rosuvastatin",
    "Amiodarone", "Digoxin", "Furosemide", "Spironolactone", "Hydrochlorothiazide",
    "Metformin", "Insulin", "Glipizide", "Levothyroxine",
    "Omeprazole", "Pantoprazole", "Famotidine", "Ondansetron", "Metoclopramide",
    "Albuterol", "Fluticasone", "Tiotropium", "Montelukast",
    "Azithromycin", "Ciprofloxacin", "Levofloxacin", "Amoxicillin", "Cephalexin", "Vancomycin", "Piperacillin-Tazobactam", "Meropenem",
    "Fluconazole", "Voriconazole",
    "Acetaminophen", "Ibuprofen", "Naproxen", "Meloxicam", "Celecoxib",
    "Morphine", "Fentanyl", "Hydromorphone", "Oxycodone", "Tramadol", "Gabapentin", "Pregabalin",
    "Midazolam", "Propofol", "Dexmedetomidine", "Lorazepam", "Diazepam",
    "Haloperidol", "Quetiapine", "Olanzapine", "Sertraline", "Escitalopram", "Fluoxetine", "Bupropion",
    "Phenytoin", "Levetiracetam", "Valproate", "Carbamazepine",
    "Prednisone", "Methylprednisolone", "Dexamethasone", "Hydrocortisone",
    "Epinephrine", "Norepinephrine", "Vasopressin", "Dobutamine", "Milrinone",
    "Potassium Chloride", "Magnesium Sulfate", "Calcium Gluconate", "Sodium Bicarbonate"
])))

# ============================================================
# 5. CLOUD NLM AUTO-BUILDER
# ============================================================
def build_cloud_database():
    db_path = DATA_DIR / "cdss.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            count = conn.execute("SELECT COUNT(*) FROM ddi_rules").fetchone()[0]
            conn.close()
            if count > 10: return
        except: pass

    acute_drugs = ["warfarin", "aspirin", "amiodarone", "digoxin", "furosemide", "simvastatin", "lisinopril", "metformin", "ondansetron", "ciprofloxacin"]
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
        url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={'+'.join(rxcuis)}"
        r = requests.get(url, timeout=15)
        data = r.json()
    except: return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS ddi_rules (drug1 TEXT, drug2 TEXT, severity TEXT, mechanism TEXT, management TEXT, onset TEXT)")
        cur.execute("DELETE FROM ddi_rules WHERE mechanism LIKE '%NLM DDI%'")
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
        conn.commit()
        conn.close()
    except: pass

build_cloud_database()

# Dummy PK Data Generator
def get_pk_df():
    rows = [
        {"drug": "Vancomycin", "vd_l_per_kg": 0.7, "renal_fraction": 0.9, "half_life_h": 6, "tdm_target": "AUC/MIC 400-600", "metabolism": "Renal", "protein_binding": "30-55%", "note": "Adjust per TDM."},
        {"drug": "Gentamicin", "vd_l_per_kg": 0.25, "renal_fraction": 0.95, "half_life_h": 2.5, "tdm_target": "Peak 5-10, Trough < 2", "metabolism": "Renal", "protein_binding": "<10%", "note": "Nephrotoxic."},
        {"drug": "Digoxin", "vd_l_per_kg": 7.0, "renal_fraction": 0.7, "half_life_h": 36, "tdm_target": "0.5-0.9 ng/mL", "metabolism": "Renal", "protein_binding": "25%", "note": "Narrow index."}
    ]
    return pd.DataFrame(rows)

# Helper for DDI matching
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
    return row

# ============================================================
# 6. MAIN UI TABS
# ============================================================
st.title("Clinical Decision Support Dashboard")
tab1, tab2, tab3 = st.tabs(["🩺 Clinical & DDI", "💊 PK/PD Engine", "🩸 AI Biomarkers"])

# --- TAB 1: CLINICAL & DDI ---
with tab1:
    st.markdown("##### A. Vital Signs")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    hr = v1.number_input("HR", 0, 250, 85)
    sbp = v2.number_input("SBP", 0, 300, 120)
    dbp = v3.number_input("DBP", 0, 200, 80)
    temp = v4.number_input("Temp", 30.0, 45.0, 37.0)
    rr = v5.number_input("RR", 0, 60, 16)
    spo2 = v6.number_input("SpO2", 0, 100, 98)
    
    st.markdown("##### B. Active Medications")
    selected_meds = st.multiselect("Search and add medications", options=COMMON_DRUGS, placeholder="Type to search (e.g., Warfarin)...")
    
    if st.button("🔍 Analyze Patient", type="primary", use_container_width=True):
        ews = 0
        if hr < 50 or hr > 110: ews += 2
        if sbp < 90 or sbp > 180: ews += 2
        if temp < 35.5 or temp > 38.5: ews += 1
        if rr > 20 or rr < 8: ews += 1
        if spo2 < 92: ews += 2
        
        if ews >= 3: st.error(f"🚨 Early Warning Score: {ews} (Unstable)")
        else: st.success(f"✅ Early Warning Score: {ews} (Stable)")
        
        st.markdown("#### 🚨 Interaction Results")
        if len(selected_meds) < 2:
            st.info("Add at least two medications to check for interactions.")
        else:
            found = False
            for i in range(len(selected_meds)):
                for j in range(i + 1, len(selected_meds)):
                    match = get_ddi_match_db(selected_meds[i], selected_meds[j])
                    if match:
                        found = True
                        sev = match['severity']
                        color = "red" if sev in ["MAJOR", "CONTRAINDICATED"] else "orange"
                        with st.expander(f"**{selected_meds[i]} + {selected_meds[j]}** — {sev}", expanded=True):
                            st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:1.2rem;'>{sev}</span>", unsafe_allow_html=True)
                            st.write(f"**Mechanism:** {match['mechanism']}")
                            st.write(f"**Management:** {match['management']}")
            if not found:
                st.success("✅ No major interactions found in the active database.")

# --- TAB 2: PK/PD ENGINE ---
with tab2:
    st.markdown("##### Renal Dosing & Pharmacokinetics")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", 18, 110, 65)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    weight = c3.number_input("Weight (kg)", 20.0, 300.0, 75.0)
    height = c4.number_input("Height (cm)", 100.0, 230.0, 170.0)
    scr = st.number_input("Serum Creatinine (mg/dL)", 0.1, 15.0, 1.0)
    
    if st.button("🧮 Calculate PK Profile", type="primary", use_container_width=True):
        height_in = height / 2.54
        ibw = (50 + 2.3 * (height_in - 60)) if sex == "Male" else (45.5 + 2.3 * (height_in - 60))
        dosing_weight = weight if weight <= ibw else ibw + 0.4 * (weight - ibw)
        crcl = (((140 - age) * dosing_weight) / (72 * scr)) * (0.85 if sex == "Female" else 1.0)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("CrCl (Cockcroft-Gault)", f"{crcl:.1f} mL/min")
        m2.metric("Ideal Body Weight", f"{ibw:.1f} kg")
        m3.metric("Dosing Weight", f"{dosing_weight:.1f} kg")
        
        pk_df = get_pk_df()
        for drug in ["Vancomycin", "Gentamicin"]:
            r = pk_df[pk_df["drug"] == drug].iloc[0]
            with st.expander(f"**{drug}** 24-Hour Simulation", expanded=True):
                adjusted_t_half = r['half_life_h'] * (100.0 / max(crcl, 10.0))
                k_elim = 0.693 / adjusted_t_half
                Vd = r['vd_l_per_kg'] * dosing_weight
                dose = 1500 if drug == "Vancomycin" else 500
                
                times = np.linspace(0, 24, 500)
                conc = np.zeros_like(times)
                for t_dose in [0, 12]:
                    mask = times >= t_dose
                    conc[mask] += (dose / Vd) * np.exp(-k_elim * (times[mask] - t_dose))
                    
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(times, conc, color="#0ea5e9", linewidth=3)
                ax.set_facecolor('#0f172a')
                fig.patch.set_facecolor('#0f172a')
                ax.tick_params(colors='#94a3b8')
                ax.set_title(f"{drug} Concentration (Adjusted t½: {adjusted_t_half:.1f}h)", color='white')
                st.pyplot(fig)

# --- TAB 3: AI BIOMARKERS ---
with tab3:
    st.markdown("##### Deep Learning Biomarker Analysis")
    st.info("Upload a CSV of patient labs (CRP, IL-6, LDH) to run the DeepClinicalNet.")
    
    class DeepClinicalNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid())
        def forward(self, x): return self.net(x)
        
    st.success("Neural Network Architecture Loaded.")