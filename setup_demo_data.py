import requests
import sqlite3
import time
import os

DB_PATH = "data/cdss.db"
ACUTE_DRUGS = [
    "warfarin", "heparin", "aspirin", "clopidogrel", "apixaban",
    "furosemide", "lisinopril", "metoprolol", "amiodarone", "digoxin",
    "metformin", "simvastatin", "atorvastatin", "amlodipine", "spironolactone",
    "vancomycin", "ciprofloxacin", "levofloxacin", "azithromycin", "clarithromycin",
    "ondansetron", "pantoprazole", "morphine", "fentanyl", "midazolam",
    "haloperidol", "quetiapine", "phenytoin", "levetiracetam", "gabapentin"
]

def map_severity(sev_code):
    if sev_code == "4" or "contraindicated" in str(sev_code).lower(): return "CONTRAINDICATED"
    elif sev_code == "3" or "major" in str(sev_code).lower(): return "MAJOR"
    elif sev_code == "2" or "moderate" in str(sev_code).lower(): return "MODERATE"
    return "MINOR"

def get_rxcui(name):
    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    try:
        r = requests.get(url, params={"name": name, "search": 2}, timeout=10)
        if r.status_code == 200:
            ids = r.json().get("idGroup", {}).get("rxnormId")
            if ids: return ids[0]
    except: pass
    return None

def setup_demo_data():
    os.makedirs("data", exist_ok=True)
    print("🔍 Fetching open-source NLM (National Library of Medicine) DDI data...")
    rxcuis, mapping = [], {}
    
    for drug in ACUTE_DRUGS:
        cui = get_rxcui(drug)
        if cui:
            rxcuis.append(cui)
            mapping[cui] = drug.title()
        time.sleep(0.2)
        
    cuis_str = "+".join(rxcuis)
    url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={cuis_str}"
    
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    conn = sqlite3.connect(DB_PATH)
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
                d1_cui = concepts[0].get("minConceptItem", {}).get("rxcui")
                d2_cui = concepts[1].get("minConceptItem", {}).get("rxcui")
                if d1_cui in mapping and d2_cui in mapping:
                    name1, name2 = mapping[d1_cui], mapping[d2_cui]
                    sev = map_severity(pair.get("severity", "N"))
                    desc = pair.get("description", "No description.")
                    cur.execute("INSERT INTO ddi_rules VALUES (?, ?, ?, ?, ?, ?)", (name1, name2, sev, f"NLM DDI: {desc}", "Consult pharmacist.", "Unknown"))
                    count += 1
                    
    conn.commit()
    conn.close()
    print(f"✅ SUCCESS! Created demo database with {count} NLM clinical DDIs.")

if __name__ == "__main__":
    setup_demo_data()