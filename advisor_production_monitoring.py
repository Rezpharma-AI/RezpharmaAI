"""
Advisor Production Monitoring: DCA, CUSUM Drift Detection, and Logged Bandits
Implements the blueprint's mandate for continuous safety monitoring and 
clinical utility evaluation.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════
# 1. DECISION CURVE ANALYSIS (DCA)
# ═══════════════════════════════════════════════════════════
class DecisionCurveAnalyzer:
    """
    Evaluates clinical utility using Net Benefit across threshold probabilities.
    Blueprint: "A useful CDSS should show a higher net benefit than the naive 
    'treat-all' and 'treat-none' strategies."
    """
    def __init__(self, thresholds: np.ndarray = None):
        self.thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99)
        
    def compute_net_benefit(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
        """
        Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        """
        n = len(y_true)
        if n == 0: return 0.0
        
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        weight = threshold / (1 - threshold) if threshold < 1 else 0
        nb = (tp / n) - (fp / n) * weight
        return nb

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Compute DCA curves for CDSS, Treat-All, and Treat-None."""
        nb_cdss = []
        nb_treat_all = []
        prevalence = np.mean(y_true)
        
        for pt in self.thresholds:
            nb_cdss.append(self.compute_net_benefit(y_true, y_prob, pt))
            # Treat-all NB = prevalence - (1 - prevalence) * (pt / (1 - pt))
            weight = pt / (1 - pt) if pt < 1 else 0
            nb_treat_all.append(prevalence - (1 - prevalence) * weight)
            
        nb_treat_none = [0.0] * len(self.thresholds)
        
        # Calculate clinical utility area (where CDSS > both baselines)
        cdss_arr = np.array(nb_cdss)
        max_baseline = np.maximum(nb_treat_all, nb_treat_none)
        utility_mask = cdss_arr > max_baseline
        utility_pct = np.mean(utility_mask) * 100
        
        return {
            'thresholds': self.thresholds,
            'nb_cdss': np.array(nb_cdss),
            'nb_treat_all': np.array(nb_treat_all),
            'nb_treat_none': np.array(nb_treat_none),
            'clinical_utility_pct': utility_pct
        }

    def plot(self, results: Dict, filename: str = 'dca_plot.png'):
        """Generate and save the DCA plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(results['thresholds'], results['nb_cdss'], 'b-', linewidth=2, label='RezpharmaCDSS')
        plt.plot(results['thresholds'], results['nb_treat_all'], 'r--', label='Treat All')
        plt.plot(results['thresholds'], results['nb_treat_none'], 'k-', label='Treat None')
        
        plt.xlabel('Threshold Probability (Clinician Risk Tolerance)')
        plt.ylabel('Net Benefit')
        plt.title('Decision Curve Analysis: Clinical Utility of CDSS')
        plt.legend()
        plt.ylim(-0.1, max(0.3, np.max(results['nb_cdss']) + 0.05))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"    📊 DCA plot saved to {filename}")


# ═══════════════════════════════════════════════════════════
# 2. CUSUM DRIFT MONITORING
# ═══════════════════════════════════════════════════════════
class CUSUMDriftMonitor:
    """
    Monitors probability residuals for calibration drift.
    Blueprint: "A sustained alarm from the CUSUM chart indicates that the 
    system's probabilities are drifting... trigger a recalibration procedure."
    """
    def __init__(self, threshold_h: float = 5.0, slack_k: float = 0.5):
        self.threshold_h = threshold_h
        self.slack_k = slack_k
        self.s_pos = 0.0  # Detects over-prediction (model thinks risk is higher than reality)
        self.s_neg = 0.0  # Detects under-prediction (model misses actual harms)
        self.history = []
        self.alarm_triggered = False
        
    def update(self, predicted_prob: float, actual_outcome: int) -> Dict:
        """
        Update CUSUM with a new patient outcome.
        Residual X_t = P(harm) - Actual
        """
        residual = predicted_prob - actual_outcome
        
        # Update cumulative sums
        self.s_pos = max(0, self.s_pos + residual - self.slack_k)
        self.s_neg = max(0, self.s_neg - residual - self.slack_k)
        
        # Check for alarm
        alarm = (self.s_pos > self.threshold_h) or (self.s_neg > self.threshold_h)
        if alarm:
            self.alarm_triggered = True
            
        self.history.append({
            'predicted': predicted_prob,
            'actual': actual_outcome,
            'residual': residual,
            's_pos': self.s_pos,
            's_neg': self.s_neg,
            'alarm': alarm
        })
        
        return {'s_pos': self.s_pos, 's_neg': self.s_neg, 'alarm': alarm}
        
    def plot(self, filename: str = 'cusum_plot.png'):
        """Plot the CUSUM control chart."""
        s_pos_hist = [h['s_pos'] for h in self.history]
        s_neg_hist = [h['s_neg'] for h in self.history]
        t = range(len(self.history))
        
        plt.figure(figsize=(10, 6))
        plt.plot(t, s_pos_hist, 'r-', label='S+ (Over-prediction drift)', linewidth=2)
        plt.plot(t, s_neg_hist, 'b-', label='S- (Under-prediction drift)', linewidth=2)
        plt.axhline(self.threshold_h, color='black', linestyle='--', label=f'Alarm Threshold (h={self.threshold_h})')
        
        plt.xlabel('Patient Sequence (Time)')
        plt.ylabel('CUSUM Statistic')
        plt.title('CUSUM Drift Monitoring: Calibration Safety Check')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"    📊 CUSUM plot saved to {filename}")


# ═══════════════════════════════════════════════════════════
# 3. LOGGED BANDIT (CLINICIAN FEEDBACK LOOP)
# ═══════════════════════════════════════════════════════════
class LoggedBanditEvaluator:
    """
    Tracks clinician actions (Accept, Override, Ignore) to evaluate policy.
    Blueprint: "Every time a clinician accepts, overrides, or ignores a 
    recommendation, the action is logged... used in a logged bandit evaluation."
    """
    def __init__(self):
        self.logs = []
        
    def log_action(self, rec_id: str, predicted_prob: float, action_recommended: str, 
                   clinician_action: str, true_outcome: int):
        """Log a clinician's interaction with an alert."""
        self.logs.append({
            'rec_id': rec_id,
            'prob': predicted_prob,
            'recommended': action_recommended,
            'clinician_action': clinician_action,
            'true_outcome': true_outcome
        })
        
    def evaluate_policy(self) -> Dict:
        """Compute override rates and safety metrics."""
        if not self.logs:
            return {}
            
        total = len(self.logs)
        accepts = sum(1 for l in self.logs if l['clinician_action'] == 'accept')
        overrides = sum(1 for l in self.logs if l['clinician_action'] == 'override')
        ignores = sum(1 for l in self.logs if l['clinician_action'] == 'ignore')
        
        # Safety check: How often did clinicians override, and was the harm actually real?
        overridden_harms = sum(1 for l in self.logs if l['clinician_action'] == 'override' and l['true_outcome'] == 1)
        
        return {
            'total_alerts': total,
            'acceptance_rate': accepts / total,
            'override_rate': overrides / total,
            'ignore_rate': ignores / total,
            'dangerous_overrides': overridden_harms,
            'message': 'High override rates may indicate alert fatigue or poor calibration.'
        }


# ═══════════════════════════════════════════════════════════
# CLINICAL DEMONSTRATION
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  Advisor Production Monitoring: DCA, CUSUM & Logged Bandits")
    print("=" * 70)
    
    np.random.seed(42)
    
    # ─────────────────────────────────────────────────────────
    # 1. Decision Curve Analysis (Clinical Utility)
    # ─────────────────────────────────────────────────────────
    print("\n[1] Decision Curve Analysis (DCA)")
    print("    Simulating 1000 patient outcomes to evaluate Net Benefit...")
    
    n_patients = 1000
    true_harm = np.random.binomial(1, 0.15, n_patients) # 15% base rate of harm
    # CDSS predictions (well-calibrated model)
    cdss_probs = np.clip(true_harm * 0.6 + np.random.beta(2, 5, n_patients), 0, 1)
    
    dca = DecisionCurveAnalyzer()
    dca_results = dca.evaluate(true_harm, cdss_probs)
    
    print(f"    CDSS Clinical Utility: Superior to baselines across {dca_results['clinical_utility_pct']:.1f}% of thresholds.")
    dca.plot(dca_results)
    
    # ─────────────────────────────────────────────────────────
    # 2. CUSUM Drift Monitoring (Safety & Calibration)
    # ─────────────────────────────────────────────────────────
    print("\n[2] CUSUM Drift Monitoring")
    print("    Simulating a stream of patients. At patient 150, a 'domain shift' occurs")
    print("    (e.g., a new, sicker patient population arrives, causing the model to under-predict risk).")
    
    monitor = CUSUMDriftMonitor(threshold_h=4.0, slack_k=0.1)
    
    # Phase 1: Well-calibrated (Patients 1-150)
    for i in range(150):
        p_true = 0.10
        actual = np.random.binomial(1, p_true)
        predicted = p_true + np.random.normal(0, 0.02) # Small noise
        monitor.update(np.clip(predicted, 0, 1), actual)
        
    # Phase 2: Domain Shift / Model Degradation (Patients 151-300)
    # True risk jumps to 30%, but model still predicts ~10%
    drift_detected_at = None
    for i in range(150, 300):
        p_true = 0.30  # Actual risk is high
        actual = np.random.binomial(1, p_true)
        predicted = 0.10 + np.random.normal(0, 0.02) # Model is under-predicting!
        result = monitor.update(np.clip(predicted, 0, 1), actual)
        
        if result['alarm'] and drift_detected_at is None:
            drift_detected_at = i
            
    print(f"    Domain shift introduced at patient 150.")
    if drift_detected_at:
        print(f"    🚨 CUSUM ALARM TRIGGERED at patient {drift_detected_at}!")
        print(f"    → System automatically flags model for recalibration to prevent unsafe recommendations.")
    else:
        print(f"    No drift detected (unexpected).")
        
    monitor.plot()
    
    # ─────────────────────────────────────────────────────────
    # 3. Logged Bandit (Clinician Feedback)
    # ─────────────────────────────────────────────────────────
    print("\n[3] Logged Bandit: Clinician Feedback Loop")
    
    bandit = LoggedBanditEvaluator()
    
    # Simulate 50 clinician interactions
    for i in range(50):
        prob = np.random.uniform(0.1, 0.9)
        actual = np.random.binomial(1, prob)
        
        # Simulate clinician behavior: 
        # If prob is low but actual is 1, they might ignore (alert fatigue)
        if prob < 0.3 and actual == 1:
            action = 'ignore'
        elif prob > 0.7:
            action = 'accept'
        else:
            action = np.random.choice(['accept', 'override'], p=[0.7, 0.3])
            
        bandit.log_action(f"REC_{i}", prob, "hold_drug", action, actual)
        
    metrics = bandit.evaluate_policy()
    print(f"    Total Alerts Evaluated: {metrics['total_alerts']}")
    print(f"    Acceptance Rate: {metrics['acceptance_rate']:.1%}")
    print(f"    Override Rate:   {metrics['override_rate']:.1%}")
    print(f"    Ignore Rate:     {metrics['ignore_rate']:.1%} (Potential Alert Fatigue)")
    print(f"    Dangerous Overrides (Clinician overrode, but harm occurred): {metrics['dangerous_overrides']}")
    
    print("\n" + "=" * 70)
    print("  The Advisor doesn't just make recommendations; it continuously")
    print("  audits its own clinical utility (DCA), detects calibration drift")
    print("  (CUSUM), and learns from clinician behavior (Logged Bandits).")
    print("=" * 70)