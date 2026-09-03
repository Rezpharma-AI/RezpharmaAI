import sqlite3
import csv
import hashlib
import os
from pathlib import Path

DB_PATH = "database/rezpharma.db"
RAW_DIR = Path("data/raw")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    # Add extra tables for side effects to support M1 Tier 2
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS side_effects (
            side_effect_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS drug_side_effects (
            drug_id TEXT REFERENCES drugs(drug_id),
            side_effect_id TEXT REFERENCES side_effects(side_effect_id),
            PRIMARY KEY (drug_id, side_effect_id)
        );
        CREATE INDEX IF NOT EXISTS idx_dse_drug ON drug_side_effects(drug_id);
    """)
    conn.commit()
    return conn

def infer_severity_and_lr(interaction_text, mechanism, action):
    """Infer severity and log-LR from the interaction description text."""
    text = (interaction_text + " " + mechanism).lower()
    
    if "contraindicated" in text or "fatal" in text or "life-threatening" in text:
        return "contraindicated", 4.0
    if "severe" in text or "toxicity" in text or "hemorrhage" in text or "bleeding" in text or "arrhythmia" in text or "torsades" in text or "prolongation" in text:
        return "severe", 2.5
    if "moderate" in text or "increase" in action or "decrease" in action:
        return "moderate", 1.5
        
    return "mild", 0.8

def load_drugs(conn):
    print("\n[1/4] Loading Drugs...")
    cursor = conn.cursor()
    
    # Load from ChEMBL mapping for clean names
    chembl_file = RAW_DIR / "drugs_chembl.csv"
    if chembl_file.exists():
        print(f"  Reading {chembl_file.name}...")
        with open(chembl_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_id = row.get("DrugBank_ID", "").strip()
                name = row.get("Drug_Name", "").strip().strip("'\"")
                if drug_id and name:
                    cursor.execute("INSERT OR IGNORE INTO drugs (drug_id, generic_name) VALUES (?, ?)", (drug_id, name))
        conn.commit()
        
    # Ensure all drugs from the massive DDI file are in the DB
    ddi_file = RAW_DIR / "drug_drug_edges.csv"
    if ddi_file.exists():
        print(f"  Extracting unique drugs from {ddi_file.name}...")
        with open(ddi_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id1, name1 = row.get("id1", "").strip(), row.get("name1", "").strip()
                id2, name2 = row.get("id2", "").strip(), row.get("name2", "").strip()
                if id1 and name1:
                    cursor.execute("INSERT OR IGNORE INTO drugs (drug_id, generic_name) VALUES (?, ?)", (id1, name1))
                if id2 and name2:
                    cursor.execute("INSERT OR IGNORE INTO drugs (drug_id, generic_name) VALUES (?, ?)", (id2, name2))
        conn.commit()
        
    cursor.execute("SELECT COUNT(*) FROM drugs")
    total = cursor.fetchone()[0]
    print(f"  Total drugs in database: {total:,}")
    return total

def load_ddis(conn):
    print("\n[2/4] Loading Drug-Drug Interactions (481 MB)...")
    cursor = conn.cursor()
    ddi_file = RAW_DIR / "drug_drug_edges.csv"
    if not ddi_file.exists():
        print(f"  ERROR: {ddi_file.name} not found!")
        return 0
        
    print(f"  Processing {ddi_file.name}... this may take 1-2 minutes.")
    interactions_loaded = 0
    
    with open(ddi_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id1 = row.get("id1", "").strip()
            id2 = row.get("id2", "").strip()
            interaction_text = row.get("interaction", "").strip()
            mechanism_short = row.get("mechanism", "").strip()
            action = row.get("action", "").strip()
            
            if not id1 or not id2:
                continue
                
            severity, log_lr = infer_severity_and_lr(interaction_text, mechanism_short, action)
            int_id = hashlib.md5(f"{id1}_{id2}_{mechanism_short}".encode()).hexdigest()[:16]
            full_mechanism = f"{mechanism_short} ({action}): {interaction_text}"[:500]
            
            try:
                cursor.execute(
                    """INSERT OR IGNORE INTO ddi_interactions 
                       (interaction_id, perpetrator_drug_id, victim_drug_id, mechanism, severity, log_lr, evidence_level) 
                       VALUES (?, ?, ?, ?, ?, ?, 'curated')""",
                    (int_id, id1, id2, full_mechanism, severity, log_lr)
                )
                interactions_loaded += 1
                if interactions_loaded % 100000 == 0:
                    conn.commit()
                    print(f"    ... processed {interactions_loaded:,} interactions")
            except Exception:
                pass
                
    conn.commit()
    print(f"  Loaded {interactions_loaded:,} DDI interactions.")
    return interactions_loaded

def load_side_effects(conn):
    print("\n[3/4] Loading Side Effects (for M1 Tier 2 Signal Mining)...")
    cursor = conn.cursor()
    se_file = RAW_DIR / "drug_side_effect_with_names.csv"
    if not se_file.exists():
        print(f"  ERROR: {se_file.name} not found!")
        return 0, 0
        
    print(f"  Reading {se_file.name}...")
    se_loaded = 0
    drug_se_loaded = 0
    
    with open(se_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_id = row.get("DRUG_ID", "").strip()
            se_id = row.get("SIDE_EFFECT_ID", "").strip()
            se_name = row.get("SIDE_EFFECT_NAME", "").strip()
            
            if not drug_id or not se_id:
                continue
                
            try:
                cursor.execute("INSERT OR IGNORE INTO side_effects (side_effect_id, name) VALUES (?, ?)", (se_id, se_name))
                se_loaded += 1
                cursor.execute("INSERT OR IGNORE INTO drug_side_effects (drug_id, side_effect_id) VALUES (?, ?)", (drug_id, se_id))
                drug_se_loaded += 1
            except Exception:
                pass
                
    conn.commit()
    print(f"  Loaded {se_loaded:,} unique side effects and {drug_se_loaded:,} drug-side effect links.")
    return se_loaded, drug_se_loaded

def show_stats(conn):
    print("\n" + "="*60)
    print("  M1 KNOWLEDGE BASE STATISTICS")
    print("="*60)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM drugs")
    print(f"  Total Drugs: {cursor.fetchone()[0]:,}")
    
    cursor.execute("SELECT COUNT(*) FROM ddi_interactions")
    print(f"  Total DDIs: {cursor.fetchone()[0]:,}")
    
    cursor.execute("SELECT severity, COUNT(*) FROM ddi_interactions GROUP BY severity ORDER BY COUNT(*) DESC")
    print("\n  DDI Severity Distribution:")
    for sev, count in cursor.fetchall():
        print(f"    {sev:<15}: {count:>8,}")
        
    cursor.execute("SELECT COUNT(DISTINCT side_effect_id) FROM side_effects")
    print(f"\n  Unique Side Effects (ADRs): {cursor.fetchone()[0]:,}")
    
    cursor.execute("""
        SELECT d1.generic_name, d2.generic_name, i.severity, i.mechanism
        FROM ddi_interactions i
        JOIN drugs d1 ON i.perpetrator_drug_id = d1.drug_id
        JOIN drugs d2 ON i.victim_drug_id = d2.drug_id
        WHERE i.severity IN ('severe', 'contraindicated')
        ORDER BY RANDOM() LIMIT 5
    """)
    print("\n  Sample High-Risk Interactions Loaded:")
    for d1, d2, sev, mech in cursor.fetchall():
        print(f"    [{sev.upper()}] {d1} + {d2}")
        print(f"      -> {mech[:90]}...")

if __name__ == "__main__":
    print("="*60)
    print("  RezpharmaCDSS - M1 Knowledge Base Loader")
    print("  Processing DrugBank & PrimeKG Data")
    print("="*60)
    
    if not os.path.exists(DB_PATH):
        print(f"\n  Creating database from schema...")
        conn = sqlite3.connect(DB_PATH)
        with open("database/schema.sql", "r") as f:
            conn.executescript(f.read())
        conn.close()
        
    conn = get_db()
    load_drugs(conn)
    load_ddis(conn)
    load_side_effects(conn)
    show_stats(conn)
    conn.close()
    print("\n  Done! M1 Knowledge Base is ready.")