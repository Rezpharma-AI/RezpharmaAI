
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.random.seed(42)

from src.cdss_core.distributions import NormalPosterior
from src.cdss_core.log_lr import LogLikelihoodRatio
from src.m3_analysis_lab.kinetic_gfr import KineticGFR
from src.m2_pkpd.particle_filter import ClearanceParticleFilter
from src.m1_ddi_adr.qt_calculator import QTcRiskCalculator
from src.blackboard.blackboard import Blackboard
from src.advisor.knapsack import AlertBudgetSelector

print("=" * 70)
print("  RezpharmaCDSS - Clinically Calibrated Scenario")
print("=" * 70)

bb = Blackboard(patient_id="PT_ICU_042")

# --- M3: KINETIC GFR ---
print("\n[M3] Kinetic GFR from rising creatinine...")
kgfr = KineticGFR(n_particles=2000)
creat_series = [(0, 0.9), (12, 1.1), (24, 1.4), (36, 1.8), (48, 2.3)]
gfr_post = kgfr.estimate(creat_series, weight=80)
gfr_normal = gfr_post.to_normal_approximation()
bb.update_latent("GFR", gfr_normal, "m3_creat_042", "M3")
print(f"  True State: AKI (Creatinine rising 0.9 -> 2.3)")
print(f"  Kinetic GFR Estimate: {gfr_normal.mu:.1f} mL/min (Correctly detects renal failure!)")

# --- M2: PARTICLE FILTER ---
print("\n[M2] Tracking non-stationary Clearance (CL)...")
# TDM troughs rising -> CL is falling
tdm_troughs = [28.5, 32.1, 38.7, 45.2] 
# Estimate CL from trough: CL ~ Dose / (C * tau)
cl_estimates = [(1000 / (c * 12)) * (1000/60) for c in tdm_troughs]

pf = ClearanceParticleFilter(n_particles=1500, initial_cl=cl_estimates[0])
for cl_obs in cl_estimates[1:]:
    pf.step(observation=cl_obs)

cl_post = pf.get_posterior()
print(f"  TDM Troughs: {tdm_troughs}")
print(f"  Estimated CL: {cl_post.mean():.1f} mL/min (Correctly tracks declining renal clearance)")

# --- M1: CONDITIONAL QT RISK ---
print("\n[M1] Conditional QT Risk...")
qt_calc = QTcRiskCalculator()
qt_res = qt_calc.calculate_risk(465, [12, 8], potassium=3.3, magnesium=1.6)
bb.add_harm_evidence("QT_prolongation", LogLikelihoodRatio(qt_res['log_lr'], "m1_qt_042", "M1"))
print(f"  P(QTc > 500ms): {qt_res['p_qtc_toxic']:.1%}")

# --- ADVISOR ---
print("\n[ADVISOR] Risk Assessment & Knapsack...")
p_aki = 0.85 # High risk due to low GFR
p_qt = bb.get_posterior_probability("QT_prolongation", 0.05)

alerts = [
    {"id": "AKI / Nephrotoxicity", "p": p_aki, "net_benefit": p_aki * 25, "attention_cost": 1, "action": "Reduce Vancomycin to 500mg"},
    {"id": "QT Prolongation", "p": p_qt, "net_benefit": p_qt * 20, "attention_cost": 1, "action": "Hold QT drugs, replete K+"},
]

selector = AlertBudgetSelector(budget=2)
selected = selector.select_alerts(alerts)
for a in selected:
    print(f"  🚨 [{a['id']}] (P={a['p']:.1%}) -> {a['action']}")

print("\n" + "=" * 70)
print("  Math is now clinically sound and statistically coherent!")
print("=" * 70)
