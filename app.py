import sys
sys.path.insert(0, '.')
import streamlit as st
import sqlite3
from src.m1_ddi_adr.rule_engine import DDIRuleEngine
from src.blackboard.blackboard import Blackboard
from src.advisor.utility import ExpectedUtilityCalculator
from src.advisor.knapsack import AlertBudgetSelector

st.set_page_config(page_title="RezpharmaCDSS", layout="wide", page_icon="💊")
st.title("RezpharmaCDSS Clinical Dashboard")
st.markdown("**Multi-Module Polypharmacy Decision Support** | M1: DDI/ADR Engine Active (2.85M Interactions)")

# Sidebar for input
st.sidebar.header("Patient Medication List")
drug_input = st.sidebar.text_area(
    "Enter drugs (comma-separated):", 
    value="Warfarin, Aspirin, Amiodarone, Simvastatin, Clarithromycin"
)
run_btn = st.sidebar.button("Analyze Regimen", type="primary", use_container_width=True)

if run_btn:
    drugs = [d.strip().title() for d in drug_input.split(',') if d.strip()]
    st.write(f"**Analyzing {len(drugs)} drugs:** {', '.join(drugs)}")
    
    # Initialize M1 and Blackboard
    engine = DDIRuleEngine()
    bb = Blackboard(patient_id="PT_DEMO_001")
    
    # Run M1
    with st.spinner("Querying 2.85 Million DDI Knowledge Base..."):
        interactions = engine.check_interactions(drugs)
        
    if not interactions:
        st.success("✅ No known drug-drug interactions found in the curated knowledge base.")
    else:
        st.subheader(f"M1 Detected {len(interactions)} Interactions")
        
        # Push to Blackboard
        for i, hit in enumerate(interactions):
            lr = engine.to_log_lr(hit)
            harm_id = hit['mechanism'].split(':')[0].strip().lower().replace(' ', '_')[:30]
            try:
                bb.add_harm_evidence(harm_id, lr)
            except Exception:
                pass # Ignore duplicate provenance
                
        # Display Interactions
        for hit in interactions[:10]: # Show top 10
            sev = hit['severity'].upper()
            color = "red" if sev in ("SEVERE", "CONTRAINDICATED") else "orange"
            st.markdown(f"- **<span style='color:{color}'>[{sev}]</span>** {hit['perpetrator']} + {hit['victim']}", unsafe_allow_html=True)
            st.caption(f"  *{hit['mechanism']}*")
            
        # Advisor Section
        st.subheader("Advisor: Risk Assessment & Action Selection")
        
        harm_props = list(bb.harm_log_lrs.keys())
        if harm_props:
            advisor = ExpectedUtilityCalculator()
            advisor.add_action("hold_or_switch_drug", u_harm=-2, u_noharm=-3)
            advisor.add_action("reduce_dose", u_harm=-5, u_noharm=-1)
            advisor.add_action("increase_monitoring", u_harm=-8, u_noharm=0)
            advisor.add_action("continue_current", u_harm=-30, u_noharm=0)
            
            alerts = []
            for harm in harm_props:
                p_harm = bb.get_posterior_probability(harm, prior_prob=0.05)
                best = advisor.select_best_action(p_harm)
                alerts.append({
                    "harm": harm,
                    "p_harm": p_harm,
                    "action": best["selected"],
                    "eu": best["expected_utility"],
                    "net_benefit": p_harm * 20, # simplified benefit score
                    "attention_cost": 1
                })
                
            # Knapsack
            selector = AlertBudgetSelector(budget=3)
            selected = selector.select_alerts(alerts)
            
            st.markdown("#### 🚨 Top 3 Actionable Alerts (Budget = 3)")
            for a in selected:
                st.warning(f"**{a['harm'].replace('_', ' ').title()}** (P={a['p_harm']:.1%})\n\n**Action:** {a['action'].replace('_', ' ').title()}")
                
            st.markdown("#### 🔇 Suppressed Alerts (Preventing Alert Fatigue)")
            suppressed = [a for a in alerts if a not in selected]
            if suppressed:
                for a in suppressed:
                    st.info(f"*{a['harm'].replace('_', ' ').title()}* (P={a['p_harm']:.1%}) -> Suppressed by Knapsack")
            else:
                st.info("No alerts suppressed (total alerts <= budget).")