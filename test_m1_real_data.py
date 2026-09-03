import sys
sys.path.insert(0, '.')  # Point to project root, NOT 'src'
from src.m1_ddi_adr.rule_engine import DDIRuleEngine
import sqlite3

print("=" * 50)
print(" M1 Rule Engine - Real Database Test")
print("=" * 50)

engine = DDIRuleEngine()

# Test 1: Known high-risk pair
print("\n[Test 1] Warfarin + Aspirin")
hits = engine.check_interactions(['Warfarin', 'Aspirin'])
print(f"Found {len(hits)} interactions.")
for h in hits[:2]:
    print(f"  [{h['severity'].upper()}] {h['mechanism'][:80]}...")

# Test 2: Fetch a real severe pair from the 2.8M database
conn = sqlite3.connect('database/rezpharma.db')
cur = conn.cursor()
cur.execute("""
    SELECT p.generic_name, v.generic_name 
    FROM ddi_interactions d
    JOIN drugs p ON d.perpetrator_drug_id = p.drug_id
    JOIN drugs v ON d.victim_drug_id = v.drug_id
    WHERE d.severity = 'severe' AND d.mechanism LIKE '%QTc%'
    LIMIT 1
""")
row = cur.fetchone()
if row:
    d1, d2 = row
    print(f"\n[Test 2] Real Severe Pair from DB: {d1} + {d2}")
    hits2 = engine.check_interactions([d1, d2])
    print(f"Found {len(hits2)} interactions.")
    if hits2:
        print(f"  Log-LR: {hits2[0]['log_lr']}")
        print(f"  Mechanism: {hits2[0]['mechanism'][:100]}...")
conn.close()
print("\nDone!")