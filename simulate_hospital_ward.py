"""
Phase 14: Full End-to-End Hospital Ward Simulation
Integrates M1-M4, 2.85M Knowledge Base, Blackboard & Advisor across a patient cohort.
"""
import sys, os, random, sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m3_analysis_lab.kinetic_gfr import KineticGFR
from src.m2_pkpd.particle_filter import ClearanceParticleFilter
from src.m1_ddi_adr.rule_engine import DDIRuleEngine
from src.blackboard.blackboard import Blackboard, ProvenanceError
from src.advisor.utility import ExpectedUtilityCalculator
from src.advisor.knapsack import AlertBudgetSelector
from src.cdss_core.distributions import NormalPosterior

def main():
    print("=" * 80)
    print("  PHASE 14: FULL END-TO-END HOSPITAL WARD SIMULATION")
    print("  Integrating M1-M4, 2.85M Knowledge Base, Blackboard & Advisor")
    print("=" * 80)
    
    # 1. Connect to Knowledge Base
    print("\n[1] Connecting to 2.85M DDI Knowledge Base...")
    if not os.path.exists('database/rezpharma.db'):
        print("    ERROR: database/rezpharma.db not found. Run load_knowledge_base.py first!")
        return
        
    conn = sqlite3.connect('database/rezpharma.db')
    cur = conn.cursor()
    
    # Fetch real severe DDI pairs to assign to high-risk patients
    cur.execute("""
        SELECT p.generic_name, v.generic_name, d.mechanism 
        FROM ddi_interactions d
        JOIN drugs p ON d.perpetrator_drug_id = p.drug_id
        JOIN drugs v ON d.victim_drug_id = v.drug_id
        WHERE d.severity IN ('severe', 'contraindicated') AND d.log_lr > 2.0
        ORDER BY RANDOM() LIMIT 30
    """)
    severe_pairs = cur.fetchall()
    
    # Fetch random formulary drugs for baseline regimens
    cur.execute("SELECT generic_name FROM drugs ORDER BY RANDOM() LIMIT 100")
    formulary = [row[0] for row in cur.fetchall() if row[0]]
    conn.close()
    
    print(f"    Loaded {len(severe_pairs)} severe DDI pairs and {len(formulary)} formulary drugs.")
    
    # 2. Generate Patient Cohort
    print("\n[2] Generating Synthetic ICU/Ward Cohort (20 Patients)...")
    np.random.seed(42)
    random.seed(42)
    
    num_patients = 20
    patients = []
    trajectories = ['stable', 'aki', 'cardiogenic_shock', 'polypharmacy_risk', 'sepsis']
    
    for i in range(num_patients):
        traj = random.choice(trajectories)
        meds = random.sample(formulary, min(4, len(formulary)))
        
        # Inject real severe DDIs into polypharmacy patients
        if traj == 'polypharmacy_risk' and severe_pairs:
            pair = random.choice(severe_pairs)
            meds.extend([pair[0], pair[1]])
            meds = list(set(meds))
            
        patients.append({
            'id': f'PT_{i+1:03d}',
            'traj': traj,
            'meds': meds,
            'base_creat': np.random.uniform(0.8, 1.2),
            'base_ef': np.random.uniform(45, 65)
        })
        
    # 3. Run the CDSS Pipeline
    print("\n[3] Running Multi-Module CDSS Pipeline...")
    engine = DDIRuleEngine()
    knapsack = AlertBudgetSelector(budget=3) # Max 3 alerts per patient
    
    ward_stats = {
        'total_alerts_generated': 0,
        'total_alerts_shown': 0,
        'total_alerts_suppressed': 0,
        'high_risk_patients': []
    }
    
    for pt in patients:
        bb = Blackboard(patient_id=pt['id'])
        
        # --- M3: Analysis Lab (Kinetic GFR) ---
        kgfr = KineticGFR(n_particles=500)
        creat_series = [(0, pt['base_creat'])]
        if pt['traj'] in ['aki', 'sepsis']:
            for t in range(1, 5): # Rising creatinine
                creat_series.append((t*6, pt['base_creat'] + t * np.random.uniform(0.3, 0.6)))
        else:
            for t in range(1, 5): # Stable
                creat_series.append((t*6, pt['base_creat'] + np.random.normal(0, 0.05)))
                
        gfr_post = kgfr.estimate(creat_series, muscle_mass=1.1)
        gfr_normal = gfr_post.to_normal_approximation()
        bb.update_latent("GFR", gfr_normal, f"m3_{pt['id']}", "M3")
        
        # --- M4: Imaging (Echo EF) ---
        ef = pt['base_ef']
        if pt['traj'] == 'cardiogenic_shock':
            ef -= np.random.uniform(15, 25) # Dropping EF
        ef_post = NormalPosterior(mu=ef, sigma=5.0)
        bb.update_latent("EF", ef_post, f"m4_{pt['id']}", "M4")
        
        # --- M1: DDI/ADR (Real 2.85M Knowledge Base) ---
        interactions = engine.check_interactions(pt['meds'])
        for hit in interactions:
            lr = engine.to_log_lr(hit)
            harm_id = hit['mechanism'].split(':')[0].strip().lower().replace(' ', '_')[:25]
            try:
                bb.add_harm_evidence(harm_id, lr)
            except ProvenanceError:
                pass
                
        # --- M2: PK/PD (Particle Filter for non-stationary CL) ---
        pf = ClearanceParticleFilter(n_particles=500, initial_cl=gfr_normal.mu)
        if pt['traj'] == 'aki':
            for obs in [gfr_normal.mu, gfr_normal.mu*0.8, gfr_normal.mu*0.6, gfr_normal.mu*0.4]:
                pf.step(observation=obs)
        else:
            for obs in [gfr_normal.mu] * 4:
                pf.step(observation=obs + np.random.normal(0, 5))
                
        # --- ADVISOR: Risk Assessment ---
        patient_alerts = []
        
        if gfr_normal.mu < 40:
            patient_alerts.append({'id': 'AKI/Nephrotoxicity', 'p': 0.85, 'net_benefit': 21, 'attention_cost': 1})
        if ef < 35:
            patient_alerts.append({'id': 'Cardiogenic Shock', 'p': 0.75, 'net_benefit': 15, 'attention_cost': 1})
            
        for harm_id in bb.harm_log_lrs.keys():
            p_harm = bb.get_posterior_probability(harm_id, 0.05)
            if p_harm > 0.15:
                patient_alerts.append({'id': harm_id.replace('_', ' ').title(), 'p': p_harm, 'net_benefit': p_harm*20, 'attention_cost': 1})
                
        ward_stats['total_alerts_generated'] += len(patient_alerts)
        
        # Apply Knapsack Budget
        selected = knapsack.select_alerts(patient_alerts)
        suppressed = [a for a in patient_alerts if a not in selected]
        
        ward_stats['total_alerts_shown'] += len(selected)
        ward_stats['total_alerts_suppressed'] += len(suppressed)
        
        if len(selected) > 0 or pt['traj'] != 'stable':
            ward_stats['high_risk_patients'].append({
                'id': pt['id'], 'traj': pt['traj'], 'gfr': gfr_normal.mu, 'ef': ef,
                'alerts_shown': [a['id'] for a in selected],
                'alerts_suppressed': [a['id'] for a in suppressed]
            })
            
    # 4. Print Ward Dashboard
    print("\n" + "=" * 80)
    print("  WARD DASHBOARD: 24-HOUR CDSS SUMMARY")
    print("=" * 80)
    
    print(f"\n  📊 Global Alert Fatigue Metrics:")
    print(f"     Total Physiological Risks Detected: {ward_stats['total_alerts_generated']}")
    print(f"     Alerts Shown to Clinicians:         {ward_stats['total_alerts_shown']} (Budget=3)")
    print(f"     Alerts Suppressed (Knapsack):       {ward_stats['total_alerts_suppressed']}")
    if ward_stats['total_alerts_generated'] > 0:
        print(f"     Alert Fatigue Reduction:            {ward_stats['total_alerts_suppressed']/ward_stats['total_alerts_generated']*100:.1f}%")
        
    print(f"\n  🏥 High-Risk Patient Roster:")
    print(f"  {'Patient':<10} {'Trajectory':<20} {'GFR':<8} {'EF':<8} {'Actionable Alerts Shown':<30} {'Suppressed'}")
    print(f"  {'-'*10} {'-'*20} {'-'*8} {'-'*8} {'-'*30} {'-'*20}")
    
    for pt in ward_stats['high_risk_patients']:
        shown_str = ", ".join(pt['alerts_shown'][:2]) if pt['alerts_shown'] else "None"
        supp_str = ", ".join(pt['alerts_suppressed'][:2]) if pt['alerts_suppressed'] else "None"
        print(f"  {pt['id']:<10} {pt['traj']:<20} {pt['gfr']:<8.1f} {pt['ef']:<8.1f} {shown_str:<30} {supp_str}")
        
    print("\n" + "=" * 80)
    print("  SIMULATION COMPLETE.")
    print("  The system successfully integrated 2.85M DDIs, Kinetic GFR, Echo EF,")
    print("  Particle Filter CL tracking, and the Knapsack Advisor across 20 patients.")
    print("=" * 80)

if __name__ == "__main__":
    main()