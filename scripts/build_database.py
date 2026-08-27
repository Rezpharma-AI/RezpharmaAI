import sqlite3
import os

DB_PATH = "data/cdss.db"

def build_knowledge_base():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Create Tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ddi_rules (
        drug1 TEXT, drug2 TEXT, severity TEXT, mechanism TEXT, management TEXT
    )""")
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pk_parameters (
        drug TEXT, vd REAL, renal_clearance REAL, half_life REAL, tdm_target TEXT
    )""")

    # 2. Populate DDI Rules (Hybrid Rule-Based Engine)
    ddi_data = [
        ("Warfarin", "Amiodarone", "MAJOR", "CYP2C9 inhibition by amiodarone increases warfarin AUC.", "Reduce warfarin dose by 30-50%. Monitor INR daily for 1 week."),
        ("Warfarin", "Aspirin", "MAJOR", "Pharmacodynamic synergy; antiplatelet + anticoagulant.", "Avoid if possible. Monitor for GI bleeding."),
        ("Digoxin", "Furosemide", "MAJOR", "Furosemide causes hypokalemia, increasing digoxin toxicity.", "Monitor K+ and Mg++. Supplement if low."),
        ("Simvastatin", "Amlodipine", "MODERATE", "CYP3A4 weak inhibition increases statin exposure.", "Max simvastatin dose 20mg/day."),
        ("Metformin", "Contrast Dye", "CONTRAINDICATED", "Risk of contrast-induced nephropathy leading to lactic acidosis.", "Hold metformin 48h post-procedure. Check SCr before restarting."),
        ("Lisinopril", "Spironolactone", "MAJOR", "Dual RAAS blockade increases hyperkalemia risk.", "Monitor K+ weekly initially. Avoid if eGFR < 30.")
    ]
    cur.executemany("INSERT INTO ddi_rules VALUES (?, ?, ?, ?, ?)", ddi_data)

    # 3. Populate PK Parameters (Hybrid PK/PD Engine)
    pk_data = [
        ("Vancomycin", 0.7, 0.9, 6.0, "AUC/MIC 400-600"),
        ("Gentamicin", 0.25, 0.95, 2.5, "Peak 5-10, Trough < 2"),
        ("Warfarin", 0.14, 0.0, 40.0, "INR 2.0-3.0"),
        ("Phenytoin", 0.7, 0.05, 22.0, "Total 10-20, Free 1-2"),
        ("Meropenem", 0.3, 0.8, 1.0, "Time > MIC (Extended infusion)")
    ]
    cur.executemany("INSERT INTO pk_parameters VALUES (?, ?, ?, ?, ?)", pk_data)

    conn.commit()
    conn.close()
    print("✅ SQLite Knowledge Base (cdss.db) built successfully!")

if __name__ == "__main__":
    build_knowledge_base()
