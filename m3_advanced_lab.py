"""
Module M3: Advanced Clinical-Pharmacology Laboratory Intelligence
Implements BOCPD, Child-Pugh Posterior, and Dechallenge Analysis
as described in the architectural blueprint.
"""
import sys, os
import numpy as np
from scipy.stats import norm, gamma, poisson
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cdss_core.particles import ParticleSet

# ═══════════════════════════════════════════════════════════
# 1. BAYESIAN ONLINE CHANGE-POINT DETECTION (BOCPD)
# ═══════════════════════════════════════════════════════════
class BOCPDDetector:
    """
    Bayesian Online Change-Point Detection.
    
    Models the true physiological value as a piecewise-stationary process
    and calculates the posterior probability of a change-point having
    occurred at each moment in time.
    
    Reference: Adams & MacKay (2007) "Bayesian Online Change Point Detection"
    """
    
    def __init__(self, hazard_rate: float = 0.01, prior_mean: float = None,
                 prior_var: float = 1.0, obs_var: float = 0.1):
        self.hazard_rate = hazard_rate  # 1/expected run length
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.obs_var = obs_var
        
        # State tracking
        self.run_length_probs = np.array([1.0])
        self.data = []
        self.mu = [prior_mean] if prior_mean is not None else [0.0]
        self.var = [prior_var]
        
    def update(self, x: float) -> Dict:
        """
        Process new observation and return change-point probabilities.
        
        Returns:
            Dict with change_point_prob, run_length_distribution, current_mean
        """
        self.data.append(x)
        n = len(self.data)
        
        if self.prior_mean is None:
            self.prior_mean = x
            self.mu = [x]
        
        # Predictive probabilities for each run length
        pred_probs = np.zeros(len(self.run_length_probs) + 1)
        
        for r in range(len(self.run_length_probs)):
            # Sufficient statistics for segment with run length r
            segment = self.data[max(0, n - r - 1):n]
            if len(segment) > 0:
                seg_mean = np.mean(segment)
                seg_var = max(np.var(segment), 1e-6)
            else:
                seg_mean = self.prior_mean
                seg_var = self.prior_var
            
            # Predictive: P(x_t | r_t = r)
            pred_var = seg_var + self.obs_var
            pred_probs[r] = norm.pdf(x, loc=seg_mean, scale=np.sqrt(pred_var))
        
        # Growth probabilities: r_t = r_{t-1} + 1
        growth_probs = self.run_length_probs * pred_probs[:-1] * (1 - self.hazard_rate)
        
        # Change-point probability: r_t = 0
        change_prob = np.sum(self.run_length_probs * pred_probs[:-1]) * self.hazard_rate
        
        # Build new run-length distribution
        new_rl_probs = np.zeros(len(self.run_length_probs) + 1)
        new_rl_probs[0] = change_prob
        new_rl_probs[1:] = growth_probs
        
        # Normalize
        total = np.sum(new_rl_probs)
        if total > 0:
            self.run_length_probs = new_rl_probs / total
        else:
            self.run_length_probs = np.ones_like(new_rl_probs) / len(new_rl_probs)
        
        # Compute current posterior mean (weighted across run lengths)
        current_mean = 0.0
        for r in range(len(self.run_length_probs)):
            segment = self.data[max(0, n - r - 1):n]
            if len(segment) > 0:
                current_mean += self.run_length_probs[r] * np.mean(segment)
            else:
                current_mean += self.run_length_probs[r] * self.prior_mean
        
        return {
            "change_point_prob": float(self.run_length_probs[0]),
            "run_length_distribution": self.run_length_probs.copy(),
            "current_mean": current_mean,
            "most_likely_run_length": int(np.argmax(self.run_length_probs))
        }
    
    def detect_change_points(self, data: List[float], threshold: float = 0.3) -> List[int]:
        """Run BOCPD on a full series and return detected change-point indices."""
        self.__init__(self.hazard_rate, self.prior_mean, self.prior_var, self.obs_var)
        change_points = []
        
        for i, x in enumerate(data):
            result = self.update(x)
            if result["change_point_prob"] > threshold and i > 0:
                change_points.append(i)
        
        return change_points


# ═══════════════════════════════════════════════════════════
# 2. CHILD-PUGH POSTERIOR PROBABILITY
# ═══════════════════════════════════════════════════════════
class ChildPughPosterior:
    """
    Models hepatic function as a posterior probability distribution
    over Child-Pugh classes A, B, C rather than a single discrete label.
    
    Treats bilirubin, albumin, and INR as noisy observations of an
    underlying score, using Monte Carlo simulation to propagate
    measurement error.
    """
    
    def __init__(self, n_samples: int = 5000):
        self.n_samples = n_samples
        
        # Measurement error models (typical lab CVs)
        self.cv_bilirubin = 0.05   # 5% CV
        self.cv_albumin = 0.03     # 3% CV
        self.cv_inr = 0.05         # 5% CV
        
    def compute(self, bilirubin: float, albumin: float, inr: float,
                ascites: int = 0, encephalopathy: int = 0) -> Dict:
        """
        Compute posterior PMF over Child-Pugh classes.
        
        Args:
            bilirubin: Serum bilirubin (mg/dL)
            albumin: Serum albumin (g/dL)
            inr: International Normalized Ratio
            ascites: 0=none, 1=mild, 2=severe
            encephalopathy: 0=none, 1=grade 1-2, 2=grade 3-4
            
        Returns:
            Dict with class probabilities and weighted dose adjustment
        """
        # Monte Carlo: add measurement noise
        bili_samples = np.random.normal(bilirubin, bilirubin * self.cv_bilirubin, self.n_samples)
        alb_samples = np.random.normal(albumin, albumin * self.cv_albumin, self.n_samples)
        inr_samples = np.random.normal(inr, inr * self.cv_inr, self.n_samples)
        
        bili_samples = np.maximum(bili_samples, 0.1)
        alb_samples = np.clip(alb_samples, 1.0, 5.0)
        inr_samples = np.maximum(inr_samples, 0.8)
        
        # Score each sample
        scores = self._score_samples(bili_samples, alb_samples, inr_samples,
                                     ascites, encephalopathy)
        
        # Classify into Child-Pugh classes
        # Class A: 5-6 points, B: 7-9 points, C: 10-15 points
        class_a = np.mean(scores <= 6)
        class_b = np.mean((scores >= 7) & (scores <= 9))
        class_c = np.mean(scores >= 10)
        
        return {
            "P(class_A)": float(class_a),
            "P(class_B)": float(class_b),
            "P(class_C)": float(class_c),
            "mean_score": float(np.mean(scores)),
            "score_95_ci": (float(np.percentile(scores, 2.5)), 
                           float(np.percentile(scores, 97.5))),
            "weighted_dose_factor": self._weighted_dose_factor(class_a, class_b, class_c)
        }
    
    def _score_samples(self, bili, alb, inr, ascites, encephalopathy):
        """Compute Child-Pugh score for each Monte Carlo sample."""
        scores = np.zeros(self.n_samples)
        
        # Bilirubin scoring
        scores += np.where(bili < 2, 1, np.where(bili <= 3, 2, 3))
        
        # Albumin scoring
        scores += np.where(alb > 3.5, 1, np.where(alb >= 2.8, 2, 3))
        
        # INR scoring
        scores += np.where(inr < 1.7, 1, np.where(inr <= 2.3, 2, 3))
        
        # Ascites (fixed clinical assessment)
        scores += np.where(ascites == 0, 1, np.where(ascites == 1, 2, 3))
        
        # Encephalopathy (fixed clinical assessment)
        scores += np.where(encephalopathy == 0, 1, np.where(encephalopathy == 1, 2, 3))
        
        return scores
    
    def _weighted_dose_factor(self, p_a, p_b, p_c):
        """
        Compute weighted-average dose adjustment factor.
        Class A: 1.0 (full dose), B: 0.75, C: 0.5
        """
        return p_a * 1.0 + p_b * 0.75 + p_c * 0.5


# ═══════════════════════════════════════════════════════════
# 3. DECHALLENGE ANALYZER (N-of-1 CAUSAL INFERENCE)
# ═══════════════════════════════════════════════════════════
@dataclass
class DechallengeResult:
    """Result of N-of-1 dechallenge analysis."""
    drug_name: str
    analyte: str
    stop_time: float
    evidence_ratio: float  # Likelihood ratio for drug cessation effect
    is_confirmed: bool
    confidence: float
    description: str


class DechallengeAnalyzer:
    """
    N-of-1 Causal Inference for ADR Confirmation.
    
    Analyzes dechallenge data (lab values after drug cessation) to determine
    if observed improvement was due to drug cessation or background recovery.
    
    From the blueprint: "By running a change-point analysis anchored at the
    stop date, M3 can calculate a likelihood ratio for whether the observed
    improvement was due to the drug cessation or simply part of the background
    recovery process."
    """
    
    def __init__(self):
        self.bocpd = BOCPDDetector(hazard_rate=0.05)
        
    def analyze(self, drug_name: str, analyte: str, 
                pre_stop_values: List[float], post_stop_values: List[float],
                stop_time: float, background_recovery_rate: float = 0.02) -> DechallengeResult:
        """
        Perform dechallenge analysis.
        
        Args:
            drug_name: Suspected causative drug
            analyte: Lab analyte being monitored (e.g., 'ALT', 'creatinine')
            pre_stop_values: Lab values before drug cessation
            post_stop_values: Lab values after drug cessation
            stop_time: Time index of drug cessation
            background_recovery_rate: Expected natural recovery rate
            
        Returns:
            DechallengeResult with likelihood ratio and confidence
        """
        all_values = pre_stop_values + post_stop_values
        
        # Run BOCPD anchored at stop time
        change_probs = []
        self.bocpd = BOCPDDetector(hazard_rate=0.05)
        for i, x in enumerate(all_values):
            result = self.bocpd.update(x)
            change_probs.append(result["change_point_prob"])
        
        # Evidence under H1: Drug cessation caused change
        # Change-point probability near stop_time
        window = 3  # Look within 3 time points of stop
        stop_region = change_probs[max(0, stop_time - window):stop_time + window]
        p_change_at_stop = max(stop_region) if stop_region else 0.0
        
        # Evidence under H0: Background recovery only
        # Expected change probability without drug effect
        p_change_background = background_recovery_rate
        
        # Likelihood ratio
        if p_change_background > 0:
            lr = p_change_at_stop / p_change_background
        else:
            lr = 10.0 if p_change_at_stop > 0.3 else 1.0
        
        # Determine if dechallenge is confirmed
        is_confirmed = p_change_at_stop > 0.25 and lr > 2.0
        
        # Compute trend direction
        pre_mean = np.mean(pre_stop_values) if pre_stop_values else 0
        post_mean = np.mean(post_stop_values) if post_stop_values else 0
        
        # For adverse effects: we expect improvement (decrease for toxic markers,
        # increase for function markers like GFR)
        if analyte in ['ALT', 'AST', 'creatinine', 'bilirubin', 'INR']:
            improved = post_mean < pre_mean
        else:
            improved = post_mean > pre_mean
        
        confidence = min(lr / 10.0, 1.0)  # Normalize to 0-1
        
        if is_confirmed and improved:
            description = f"Dechallenge CONFIRMED: {analyte} improved after stopping {drug_name}"
        elif is_confirmed and not improved:
            description = f"Dechallenge INCONCLUSIVE: Change detected but {analyte} did not improve"
        else:
            description = f"Dechallenge NOT CONFIRMED: No significant change at stop time"
        
        return DechallengeResult(
            drug_name=drug_name,
            analyte=analyte,
            stop_time=stop_time,
            evidence_ratio=float(lr),
            is_confirmed=is_confirmed,
            confidence=float(confidence),
            description=description
        )


# ═══════════════════════════════════════════════════════════
# 4. RCV GATING (Reference Change Value)
# ═══════════════════════════════════════════════════════════
class RCVGate:
    """
    Reference Change Value gating from biological variation databases.
    
    RCV = z * sqrt(CV_a^2 + CV_i^2)
    
    Where CV_a = analytical imprecision, CV_i = within-subject biological variation
    Only changes exceeding RCV are considered clinically significant.
    """
    
    # Biological variation data (from EFLM database)
    BIO_VARIATION = {
        'creatinine': {'cv_a': 0.022, 'cv_i': 0.06},
        'potassium': {'cv_a': 0.015, 'cv_i': 0.048},
        'ALT': {'cv_a': 0.035, 'cv_i': 0.18},
        'AST': {'cv_a': 0.030, 'cv_i': 0.12},
        'bilirubin': {'cv_a': 0.030, 'cv_i': 0.26},
        'albumin': {'cv_a': 0.016, 'cv_i': 0.031},
        'INR': {'cv_a': 0.020, 'cv_i': 0.05},
        'hemoglobin': {'cv_a': 0.015, 'cv_i': 0.028},
        'platelets': {'cv_a': 0.040, 'cv_i': 0.091},
    }
    
    def __init__(self, z_score: float = 2.77):
        """z=2.77 for 95% confidence (two-sided)"""
        self.z_score = z_score
        
    def compute_rcv(self, analyte: str) -> float:
        """Compute RCV threshold for an analyte."""
        bv = self.BIO_VARIATION.get(analyte, {'cv_a': 0.03, 'cv_i': 0.10})
        return self.z_score * np.sqrt(bv['cv_a']**2 + bv['cv_i']**2)
    
    def is_significant_change(self, analyte: str, prev: float, curr: float) -> Dict:
        """Determine if a change exceeds RCV threshold."""
        rcv = self.compute_rcv(analyte)
        
        if prev == 0:
            return {'significant': True, 'rcv': rcv, 'relative_change': float('inf')}
        
        relative_change = abs(curr - prev) / prev
        significant = relative_change > rcv
        
        return {
            'significant': significant,
            'rcv': rcv,
            'relative_change': relative_change,
            'exceeds_by': relative_change - rcv if significant else 0
        }


# ═══════════════════════════════════════════════════════════
# CLINICAL DEMONSTRATION
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  M3 Advanced Lab Intelligence: BOCPD + Child-Pugh + Dechallenge")
    print("=" * 70)
    
    np.random.seed(42)
    
    # ─────────────────────────────────────────────────────────
    # 1. BOCPD: Detecting AKI Onset
    # ─────────────────────────────────────────────────────────
    print("\n[1] BOCPD: Detecting Acute Kidney Injury Onset")
    print("    Simulating creatinine series with sudden AKI at day 5...")
    
    # Simulate: stable creatinine, then sudden rise (AKI)
    stable = np.random.normal(1.0, 0.05, 5)  # Days 1-5: stable
    rising = np.linspace(1.1, 2.5, 7) + np.random.normal(0, 0.05, 7)  # Days 6-12: rising
    
    creatinine_series = list(stable) + list(rising)
    
    bocpd = BOCPDDetector(hazard_rate=0.1)
    change_points = bocpd.detect_change_points(creatinine_series, threshold=0.3)
    
    print(f"    Creatinine: {[f'{c:.2f}' for c in creatinine_series]}")
    print(f"    Detected change-points at indices: {change_points}")
    if change_points:
        print(f"    → AKI detected at approximately day {change_points[0] + 1}")
    else:
        print("    → No significant change detected")
    
    # ─────────────────────────────────────────────────────────
    # 2. Child-Pugh Posterior: Hepatic Function Assessment
    # ─────────────────────────────────────────────────────────
    print("\n[2] Child-Pugh Posterior: Hepatic Function Assessment")
    
    cp = ChildPughPosterior(n_samples=10000)
    result = cp.compute(bilirubin=2.8, albumin=3.0, inr=1.6, ascites=1, encephalopathy=0)
    
    print(f"    Labs: Bilirubin=2.8, Albumin=3.0, INR=1.6, Ascites=mild")
    print(f"    P(Class A): {result['P(class_A)']:.1%}")
    print(f"    P(Class B): {result['P(class_B)']:.1%}")
    print(f"    P(Class C): {result['P(class_C)']:.1%}")
    print(f"    Weighted Dose Factor: {result['weighted_dose_factor']:.2f}")
    print(f"    → Recommend {result['weighted_dose_factor']*100:.0f}% of standard dose")
    
    # ─────────────────────────────────────────────────────────
    # 3. Dechallenge Analysis: Confirming Drug-Induced Liver Injury
    # ─────────────────────────────────────────────────────────
    print("\n[3] Dechallenge Analysis: Drug-Induced Liver Injury")
    
    dechallenge = DechallengeAnalyzer()
    
    # Simulate: ALT elevated during drug use, drops after stopping
    pre_stop_alt = list(np.random.normal(180, 10, 5))  # Elevated ALT on drug
    post_stop_alt = list(np.linspace(175, 45, 7) + np.random.normal(0, 5, 7))  # Recovery
    
    result = dechallenge.analyze(
        drug_name="Methotrexate",
        analyte="ALT",
        pre_stop_values=pre_stop_alt,
        post_stop_values=post_stop_alt,
        stop_time=5,
        background_recovery_rate=0.02
    )
    
    print(f"    Drug: {result.drug_name}")
    print(f"    Analyte: {result.analyte}")
    print(f"    Evidence Ratio (LR): {result.evidence_ratio:.2f}")
    print(f"    Confirmed: {result.is_confirmed}")
    print(f"    Confidence: {result.confidence:.1%}")
    print(f"    → {result.description}")
    
    # ─────────────────────────────────────────────────────────
    # 4. RCV Gating: Filtering Noise
    # ─────────────────────────────────────────────────────────
    print("\n[4] RCV Gating: Filtering Lab Noise")
    
    rcv_gate = RCVGate()
    
    test_cases = [
        ('creatinine', 1.0, 1.05),   # Small change (noise)
        ('creatinine', 1.0, 1.15),   # Moderate change
        ('creatinine', 1.0, 1.30),   # Large change (significant)
        ('ALT', 40, 55),             # Moderate ALT change
        ('ALT', 40, 80),             # Large ALT change
    ]
    
    for analyte, prev, curr in test_cases:
        result = rcv_gate.is_significant_change(analyte, prev, curr)
        status = "🚨 SIGNIFICANT" if result['significant'] else "   (noise)"
        print(f"    {analyte:>10}: {prev:.1f} → {curr:.1f} | "
              f"RCV={result['rcv']:.1%} | Change={result['relative_change']:.1%} {status}")
    
    print("\n" + "=" * 70)
    print("  M3 transforms the clinical laboratory into a sophisticated")
    print("  sensor network, feeding probabilistic estimates of physiology")
    print("  into the blackboard to drive safer pharmacotherapy.")
    print("=" * 70)