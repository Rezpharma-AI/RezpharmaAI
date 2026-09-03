"""
Module M4: Pan-Modality Imaging as a Covariate Measurement Instrument
Implements errors-in-variables modeling, temperature scaling calibration,
conformal prediction, and modality-specific error models.
"""
import sys, os
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cdss_core.distributions import NormalPosterior

# ═══════════════════════════════════════════════════════════
# 1. MODALITY-SPECIFIC ERROR MODELS
# ═══════════════════════════════════════════════════════════
@dataclass
class ImagingMeasurement:
    """A quantitative covariate extracted from imaging with uncertainty."""
    modality: str           # 'echo', 'pocus', 'ct', 'cxr', 'ecg', 'histopath'
    covariate_name: str     # 'EF', 'lean_mass_kg', 'QTc_ms', 'PDFF', 'fibrosis_stage'
    value: float            # Point estimate
    variance: float         # Measurement variance (modality-specific)
    provenance_id: str      # Unique ID for blackboard tracking
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    def to_normal_posterior(self) -> NormalPosterior:
        """Convert to NormalPosterior for blackboard fusion."""
        return NormalPosterior(mu=self.value, sigma=self.std)


class ModalityErrorModel:
    """
    Defines measurement error characteristics for each imaging modality.
    
    From the blueprint: "The system categorizes different imaging modalities
    based on the physiological quantity they measure."
    """
    
    # Inter-reader and inter-machine variability (standard deviations)
    ERROR_MODELS = {
        # Cardiac function
        'echo_EF': {'std': 5.0, 'unit': '%', 'description': 'Transthoracic Echo EF'},
        'pocus_EF': {'std': 8.0, 'unit': '%', 'description': 'Bedside POCUS EF (less precise)'},
        'cardiac_mri_EF': {'std': 3.0, 'unit': '%', 'description': 'Cardiac MRI EF (gold standard)'},
        
        # Body composition
        'ct_lean_mass': {'std': 2.0, 'unit': 'kg', 'description': 'CT-derived lean muscle mass'},
        'ct_visceral_fat': {'std': 10.0, 'unit': 'cm2', 'description': 'CT visceral fat area'},
        'dxa_lean_mass': {'std': 1.5, 'unit': 'kg', 'description': 'DXA lean mass'},
        
        # Fluid status
        'cxr_fluid_score': {'std': 0.8, 'unit': 'score', 'description': 'CXR fluid overload score (0-4)'},
        'pocus_ivc': {'std': 0.3, 'unit': 'cm', 'description': 'IVC diameter (fluid status)'},
        
        # ECG / Waveform
        'ecg_qtc': {'std': 12.0, 'unit': 'ms', 'description': 'Automated QTc measurement'},
        'ecg_qtc_manual': {'std': 8.0, 'unit': 'ms', 'description': 'Manual QTc overread'},
        
        # Hepatic
        'mri_pdff': {'std': 2.0, 'unit': '%', 'description': 'MRI proton density fat fraction'},
        'histopath_fibrosis': {'std': 0.5, 'unit': 'stage', 'description': 'Biopsy fibrosis stage (0-4)'},
    }
    
    def get_variance(self, modality_covariate: str) -> float:
        """Get measurement variance for a specific modality-covariate pair."""
        model = self.ERROR_MODELS.get(modality_covariate, {'std': 5.0})
        return model['std'] ** 2
    
    def extract_measurement(self, modality: str, covariate: str, 
                            raw_value: float, provenance_id: str) -> ImagingMeasurement:
        """
        Extract a covariate with proper uncertainty quantification.
        
        This is where a real system would call EchoNet, TotalSegmentator, etc.
        Here we simulate the extraction with modality-specific noise.
        """
        key = f"{modality}_{covariate}"
        variance = self.get_variance(key)
        
        return ImagingMeasurement(
            modality=modality,
            covariate_name=covariate,
            value=raw_value,
            variance=variance,
            provenance_id=provenance_id
        )


# ═══════════════════════════════════════════════════════════
# 2. TEMPERATURE SCALING CALIBRATION
# ═══════════════════════════════════════════════════════════
class TemperatureScaler:
    """
    Local site calibration using temperature scaling.
    
    From the blueprint: "Since model calibration is notoriously poor across
    different institutions (domain shift), recalibrating the model's output
    probabilities on a small local held-out dataset is mandatory."
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Learn optimal temperature T on local validation data.
        
        Calibrated probability: p_cal = sigmoid(logit / T)
        """
        def nll(T):
            if T <= 0:
                return 1e10
            scaled_logits = logits / T
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature to new logits."""
        if not self.is_fitted:
            raise RuntimeError("TemperatureScaler not fitted. Call fit() first.")
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))
    
    def calibration_metrics(self, logits: np.ndarray, labels: np.ndarray, 
                           n_bins: int = 10) -> Dict:
        """Compute ECE before and after calibration."""
        probs_raw = 1.0 / (1.0 + np.exp(-logits))
        probs_cal = self.calibrate(logits)
        
        def expected_calibration_error(probs, labels, n_bins):
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
                if np.sum(mask) > 0:
                    avg_conf = np.mean(probs[mask])
                    avg_acc = np.mean(labels[mask])
                    ece += np.sum(mask) / len(probs) * abs(avg_conf - avg_acc)
            return ece
        
        return {
            'ece_before': expected_calibration_error(probs_raw, labels, n_bins),
            'ece_after': expected_calibration_error(probs_cal, labels, n_bins),
            'temperature': self.temperature,
            'improvement_pct': 100 * (1 - expected_calibration_error(probs_cal, labels, n_bins) / 
                                      max(expected_calibration_error(probs_raw, labels, n_bins), 1e-10))
        }


# ═══════════════════════════════════════════════════════════
# 3. CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════
class ConformalPredictor:
    """
    Conformal prediction for distribution-free prediction intervals.
    
    Provides calibrated prediction intervals without assuming
    a specific error distribution.
    """
    
    def __init__(self, alpha: float = 0.05):
        """alpha = 1 - coverage (e.g., 0.05 for 95% intervals)"""
        self.alpha = alpha
        self.calibration_residuals = None
        
    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray):
        """Store absolute residuals from calibration set."""
        self.calibration_residuals = np.abs(actuals - predictions)
        
    def predict_interval(self, point_prediction: float) -> Tuple[float, float]:
        """
        Generate prediction interval with guaranteed coverage.
        
        Returns (lower, upper) bounds.
        """
        if self.calibration_residuals is None:
            raise RuntimeError("Must call calibrate() first.")
        
        # Quantile of calibration residuals
        n = len(self.calibration_residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        q_hat = np.quantile(self.calibration_residuals, q_level)
        
        return (point_prediction - q_hat, point_prediction + q_hat)


# ═══════════════════════════════════════════════════════════
# 4. ERRORS-IN-VARIABLES INTEGRATION WITH M2
# ═══════════════════════════════════════════════════════════
class ErrorsInVariablesDosing:
    """
    Propagates imaging measurement uncertainty into dose calculations.
    
    From the blueprint: "Inside the NLME model, this variance is incorporated
    as part of the likelihood, preventing the model from becoming overconfident
    in its dose predictions."
    """
    
    def __init__(self, base_cl: float, base_v: float, 
                 allometric_exp_cl: float = 0.75, allometric_exp_v: float = 1.0):
        self.base_cl = base_cl
        self.base_v = base_v
        self.allometric_exp_cl = allometric_exp_cl
        self.allometric_exp_v = allometric_exp_v
        
    def compute_dose_with_uncertainty(self, lean_mass_mean: float, lean_mass_var: float,
                                      target_auc: float = 400.0,
                                      n_monte_carlo: int = 5000) -> Dict:
        """
        Compute dose recommendation accounting for lean mass measurement uncertainty.
        
        Uses Monte Carlo integration over the measurement error distribution.
        """
        # Sample lean mass from its measurement distribution
        lean_mass_samples = np.random.normal(lean_mass_mean, np.sqrt(lean_mass_var), n_monte_carlo)
        lean_mass_samples = np.maximum(lean_mass_samples, 30.0)  # Physiological floor
        
        # Allometric scaling: CL = CL_base * (LM/70)^0.75
        cl_samples = self.base_cl * (lean_mass_samples / 70.0) ** self.allometric_exp_cl
        v_samples = self.base_v * (lean_mass_samples / 70.0) ** self.allometric_exp_v
        
        # Dose = Target_AUC * CL
        dose_samples = target_auc * cl_samples
        
        return {
            'recommended_dose_mean': float(np.mean(dose_samples)),
            'recommended_dose_std': float(np.std(dose_samples)),
            'dose_95_ci': (float(np.percentile(dose_samples, 2.5)),
                          float(np.percentile(dose_samples, 97.5))),
            'cl_mean': float(np.mean(cl_samples)),
            'cl_95_ci': (float(np.percentile(cl_samples, 2.5)),
                        float(np.percentile(cl_samples, 97.5))),
            'n_samples': n_monte_carlo,
            'uncertainty_inflation_factor': float(np.std(dose_samples) / np.mean(dose_samples))
        }
    
    def compare_with_without_uncertainty(self, lean_mass_mean: float, lean_mass_var: float,
                                         target_auc: float = 400.0) -> Dict:
        """
        Compare dose recommendation with and without measurement uncertainty.
        Demonstrates why errors-in-variables matters.
        """
        # WITHOUT uncertainty: use point estimate
        cl_point = self.base_cl * (lean_mass_mean / 70.0) ** self.allometric_exp_cl
        dose_point = target_auc * cl_point
        
        # WITH uncertainty: Monte Carlo
        result_with = self.compute_dose_with_uncertainty(lean_mass_mean, lean_mass_var, target_auc)
        
        return {
            'dose_without_uncertainty': dose_point,
            'dose_with_uncertainty_mean': result_with['recommended_dose_mean'],
            'dose_with_uncertainty_ci': result_with['dose_95_ci'],
            'confidence_interval_width': result_with['dose_95_ci'][1] - result_with['dose_95_ci'][0],
            'message': 'Ignoring measurement uncertainty produces overconfident (narrow) intervals'
        }


# ═══════════════════════════════════════════════════════════
# CLINICAL DEMONSTRATION
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  M4 Imaging Module: Errors-in-Variables & Calibration")
    print("=" * 70)
    
    np.random.seed(42)
    
    # ─────────────────────────────────────────────────────────
    # 1. Modality-Specific Error Models
    # ─────────────────────────────────────────────────────────
    print("\n[1] Modality-Specific Error Models")
    
    error_model = ModalityErrorModel()
    
    # Simulate extracting EF from different modalities
    true_ef = 42.0
    
    echo_ef = error_model.extract_measurement('echo', 'EF', true_ef + np.random.normal(0, 5), 'echo_001')
    pocus_ef = error_model.extract_measurement('pocus', 'EF', true_ef + np.random.normal(0, 8), 'pocus_001')
    mri_ef = error_model.extract_measurement('cardiac_mri', 'EF', true_ef + np.random.normal(0, 3), 'mri_001')
    
    print(f"    True EF: {true_ef}%")
    print(f"    Echo EF:      {echo_ef.value:.1f}% ± {echo_ef.std:.1f}% (variance={echo_ef.variance:.1f})")
    print(f"    POCUS EF:     {pocus_ef.value:.1f}% ± {pocus_ef.std:.1f}% (variance={pocus_ef.variance:.1f})")
    print(f"    Cardiac MRI:  {mri_ef.value:.1f}% ± {mri_ef.std:.1f}% (variance={mri_ef.variance:.1f})")
    print(f"    → MRI provides tightest uncertainty (gold standard)")
    
    # ─────────────────────────────────────────────────────────
    # 2. Temperature Scaling Calibration
    # ─────────────────────────────────────────────────────────
    print("\n[2] Temperature Scaling: Local Site Calibration")
    
    # Simulate a miscalibrated model (overconfident)
    n_cal = 500
    true_labels = np.random.binomial(1, 0.3, n_cal)
    # Model outputs overconfident logits
    logits = true_labels * 3.0 + np.random.normal(0, 1, n_cal) + 0.5
    
    scaler = TemperatureScaler()
    T = scaler.fit(logits, true_labels)
    metrics = scaler.calibration_metrics(logits, true_labels)
    
    print(f"    Calibration samples: {n_cal}")
    print(f"    Learned temperature: T = {T:.3f}")
    print(f"    ECE before calibration: {metrics['ece_before']:.4f}")
    print(f"    ECE after calibration:  {metrics['ece_after']:.4f}")
    print(f"    Improvement: {metrics['improvement_pct']:.1f}%")
    print(f"    → Temperature scaling corrects domain shift between institutions")
    
    # ─────────────────────────────────────────────────────────
    # 3. Conformal Prediction
    # ─────────────────────────────────────────────────────────
    print("\n[3] Conformal Prediction: Distribution-Free Intervals")
    
    # Simulate EF prediction model
    n_test = 200
    true_efs = np.random.normal(45, 10, n_test)
    predicted_efs = true_efs + np.random.normal(0, 6, n_test)  # Model error
    
    conformal = ConformalPredictor(alpha=0.05)  # 95% coverage
    conformal.calibrate(predicted_efs, true_efs)
    
    # Predict interval for a new patient
    new_prediction = 40.0
    lower, upper = conformal.predict_interval(new_prediction)
    
    print(f"    New patient EF prediction: {new_prediction}%")
    print(f"    95% Conformal Interval: [{lower:.1f}%, {upper:.1f}%]")
    print(f"    Interval width: {upper - lower:.1f}%")
    print(f"    → Guaranteed 95% coverage without distributional assumptions")
    
    # ─────────────────────────────────────────────────────────
    # 4. Errors-in-Variables: Imaging Uncertainty → Dose
    # ─────────────────────────────────────────────────────────
    print("\n[4] Errors-in-Variables: Imaging Uncertainty Propagates to Dose")
    
    # CT-derived lean mass with measurement uncertainty
    lean_mass_measurement = error_model.extract_measurement(
        'ct', 'lean_mass', 55.0 + np.random.normal(0, 2), 'ct_body_001'
    )
    
    print(f"    CT Lean Mass: {lean_mass_measurement.value:.1f} kg ± {lean_mass_measurement.std:.1f} kg")
    
    eiv_dosing = ErrorsInVariablesDosing(base_cl=4.0, base_v=40.0)
    
    # Compare with and without uncertainty
    comparison = eiv_dosing.compare_with_without_uncertainty(
        lean_mass_mean=lean_mass_measurement.value,
        lean_mass_var=lean_mass_measurement.variance,
        target_auc=400.0
    )
    
    print(f"\n    WITHOUT uncertainty (point estimate):")
    print(f"      Dose = {comparison['dose_without_uncertainty']:.0f} mg (single number)")
    
    print(f"\n    WITH uncertainty (errors-in-variables):")
    print(f"      Dose = {comparison['dose_with_uncertainty_mean']:.0f} mg")
    print(f"      95% CI: [{comparison['dose_with_uncertainty_ci'][0]:.0f}, {comparison['dose_with_uncertainty_ci'][1]:.0f}] mg")
    print(f"      CI width: {comparison['confidence_interval_width']:.0f} mg")
    
    # Full Monte Carlo result
    full_result = eiv_dosing.compute_dose_with_uncertainty(
        lean_mass_mean=lean_mass_measurement.value,
        lean_mass_var=lean_mass_measurement.variance,
        target_auc=400.0
    )
    
    print(f"\n    Full Monte Carlo ({full_result['n_samples']} samples):")
    print(f"      CL: {full_result['cl_mean']:.2f} L/h, 95% CI [{full_result['cl_95_ci'][0]:.2f}, {full_result['cl_95_ci'][1]:.2f}]")
    print(f"      Dose uncertainty inflation: {full_result['uncertainty_inflation_factor']:.1%}")
    print(f"      → Ignoring imaging uncertainty produces DANGEROUSLY narrow intervals")
    
    # ─────────────────────────────────────────────────────────
    # 5. Integration: M4 → Blackboard → M1 (QT Risk)
    # ─────────────────────────────────────────────────────────
    print("\n[5] Cross-Module Integration: M4 ECG → M1 QT Risk")
    
    # ECG extracts QTc with measurement error
    qtc_measurement = error_model.extract_measurement(
        'ecg', 'qtc', 465.0 + np.random.normal(0, 12), 'ecg_001'
    )
    
    print(f"    ECG QTc: {qtc_measurement.value:.0f} ms ± {qtc_measurement.std:.0f} ms")
    
    # Propagate QTc uncertainty into QT prolongation risk
    # P(QTc > 500) accounting for measurement error
    qtc_samples = np.random.normal(qtc_measurement.value, qtc_measurement.std, 10000)
    drug_effect = 20.0  # Additive QT prolongation from drugs
    electrolyte_effect = 10.0  # From hypokalemia
    
    effective_qtc = qtc_samples + drug_effect + electrolyte_effect
    p_qt_toxic = np.mean(effective_qtc > 500)
    
    print(f"    Drug QT effect: +{drug_effect} ms")
    print(f"    Electrolyte effect: +{electrolyte_effect} ms")
    print(f"    P(QTc > 500ms): {p_qt_toxic:.1%}")
    print(f"    → M4's QTc measurement uncertainty properly propagates to M1's risk estimate")
    
    print("\n" + "=" * 70)
    print("  M4 transforms diverse imaging into quantitative, probabilistically")
    print("  sound covariates with proper uncertainty, preventing overconfident")
    print("  dose recommendations from measurement error.")
    print("=" * 70)