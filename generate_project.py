"""
RezpharmaCDSS - Clean Project Generator
Run: python generate_project.py
"""
import os, json
from pathlib import Path

BASE = Path(".")

def wf(rel_path, content):
    p = BASE / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  + {rel_path}")

def wj(rel_path, data):
    p = BASE / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"  + {rel_path}")

# ═══════════════════════════════════════
print("\n[1/8] Creating directories...")
# ═══════════════════════════════════════
for d in [
    "src/cdss_core", "src/m1_ddi_adr", "src/m2_pkpd",
    "src/m3_analysis_lab", "src/m4_graphics_imaging",
    "src/blackboard", "src/advisor",
    "notebooks", "data/raw", "data/processed",
    "database/seed_data", "tests", "config"
]:
    (BASE / d).mkdir(parents=True, exist_ok=True)
print("  Done.")

# ═══════════════════════════════════════
print("\n[2/8] Creating cdss_core...")
# ═══════════════════════════════════════

wf("src/cdss_core/__init__.py", '''
from .distributions import NormalPosterior
from .particles import ParticleSet
from .log_lr import LogLikelihoodRatio, fuse_log_lrs
from .fusion import precision_weighted_fusion
''')

wf("src/cdss_core/distributions.py", '''
import numpy as np
from dataclasses import dataclass

@dataclass
class NormalPosterior:
    """Represents N(mu, sigma^2) for precision-weighted fusion."""
    mu: float
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    @property
    def precision(self) -> float:
        return 1.0 / (self.sigma ** 2)

    @property
    def variance(self) -> float:
        return self.sigma ** 2

    def credible_interval(self, level=0.95):
        from scipy.stats import norm
        z = norm.ppf((1 + level) / 2)
        return (self.mu - z * self.sigma, self.mu + z * self.sigma)

    def __repr__(self):
        return f"N(mu={self.mu:.3f}, sigma={self.sigma:.3f})"
''')

wf("src/cdss_core/particles.py", '''
import numpy as np

class ParticleSet:
    """Weighted particle approximation of a posterior distribution."""

    def __init__(self, particles, weights=None):
        self.particles = np.asarray(particles, dtype=float)
        if weights is None:
            self.weights = np.ones(len(particles)) / len(particles)
        else:
            self.weights = np.asarray(weights, dtype=float)
            self.weights /= np.sum(self.weights)

    @property
    def n_particles(self):
        return len(self.particles)

    @property
    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights ** 2)

    def mean(self):
        return np.average(self.particles, weights=self.weights)

    def variance(self):
        mu = self.mean()
        return np.average((self.particles - mu) ** 2, weights=self.weights)

    def std(self):
        return np.sqrt(self.variance())

    def quantile(self, q):
        idx = np.argsort(self.particles)
        cumw = np.cumsum(self.weights[idx])
        return self.particles[idx][np.searchsorted(cumw, q)]

    def credible_interval(self, level=0.95):
        alpha = (1 - level) / 2
        return (self.quantile(alpha), self.quantile(1 - alpha))

    def resample_systematic(self):
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        cumsum = np.cumsum(self.weights)
        indices = np.clip(np.searchsorted(cumsum, positions), 0, self.n_particles - 1)
        return ParticleSet(self.particles[indices].copy())

    def __repr__(self):
        return f"ParticleSet(n={self.n_particles}, mean={self.mean():.3f})"
''')

wf("src/cdss_core/log_lr.py", '''
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class LogLikelihoodRatio:
    """Encapsulates a log-LR contribution from a single module."""
    value: float
    provenance_id: str
    module: str
    mechanism: str = ""
    evidence_level: str = "curated"

def fuse_log_lrs(log_lrs: List[LogLikelihoodRatio]) -> float:
    """Additive fusion of independent log-LRs."""
    return sum(lr.value for lr in log_lrs)

def log_lr_to_posterior_prob(log_lr: float, prior_prob: float = 0.05) -> float:
    """Convert log-LR to posterior probability via Bayes rule."""
    prior_odds = prior_prob / (1 - prior_prob)
    log_post_odds = np.log(prior_odds) + log_lr
    post_odds = np.exp(log_post_odds)
    return post_odds / (1 + post_odds)
''')

wf("src/cdss_core/fusion.py", '''
from typing import List
from .distributions import NormalPosterior

def precision_weighted_fusion(posteriors: List[NormalPosterior]) -> NormalPosterior:
    """Fuse multiple NormalPosterior objects using precision weighting."""
    if not posteriors:
        raise ValueError("Cannot fuse empty list")
    total_prec = sum(p.precision for p in posteriors)
    fused_mu = sum(p.mu * p.precision for p in posteriors) / total_prec
    fused_sigma = (1.0 / total_prec) ** 0.5
    return NormalPosterior(mu=fused_mu, sigma=fused_sigma)
''')

# ═══════════════════════════════════════
print("\n[3/8] Creating M1 (DDI/ADR)...")
# ═══════════════════════════════════════

wf("src/m1_ddi_adr/__init__.py", '''
from .rule_engine import DDIRuleEngine
from .signal_mining import EBGMSignalMiner
from .causal_inference import SCCSValidator
from .qt_calculator import QTcRiskCalculator
''')

wf("src/m1_ddi_adr/rule_engine.py", '''
import sqlite3
from typing import List, Dict, Optional
from ..cdss_core.log_lr import LogLikelihoodRatio

class DDIRuleEngine:
    """Rule-based DDI detection from curated knowledge base (FDA, DrugBank)."""

    def __init__(self, db_path="database/rezpharma.db"):
        self.db_path = db_path

    def check_interactions(self, drug_list: List[str]) -> List[Dict]:
        interactions = []
        for i, perp in enumerate(drug_list):
            for vict in drug_list[i+1:]:
                hit = self._query(perp, vict)
                if hit:
                    interactions.append(hit)
        return interactions

    def _query(self, perpetrator, victim) -> Optional[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT mechanism, severity, log_lr, evidence_level "
                "FROM ddi_interactions d "
                "JOIN drugs p ON d.perpetrator_drug_id = p.drug_id "
                "JOIN drugs v ON d.victim_drug_id = v.drug_id "
                "WHERE p.generic_name=? AND v.generic_name=?",
                (perpetrator, victim))
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "perpetrator": perpetrator, "victim": victim,
                    "mechanism": row[0], "severity": row[1],
                    "log_lr": row[2], "evidence_level": row[3],
                    "provenance_id": f"m1_rule_{perpetrator}_{victim}"
                }
        except Exception:
            pass
        return None

    def to_log_lr(self, interaction: Dict) -> LogLikelihoodRatio:
        return LogLikelihoodRatio(
            value=interaction["log_lr"],
            provenance_id=interaction["provenance_id"],
            module="M1",
            mechanism=interaction["mechanism"],
            evidence_level=interaction["evidence_level"])
''')

wf("src/m1_ddi_adr/signal_mining.py", '''
import numpy as np
from dataclasses import dataclass

@dataclass
class SignalResult:
    drug: str
    event: str
    n_observed: int
    e_expected: float
    ebgm: float
    ebgm_lower: float
    is_signal: bool

class EBGMSignalMiner:
    """DuMouchel gamma-Poisson shrinkage (MGPS) for FAERS signal detection."""

    def __init__(self, alpha=0.2, beta=0.1, signal_threshold=2.0):
        self.alpha = alpha
        self.beta = beta
        self.signal_threshold = signal_threshold

    def compute_ebgm(self, n_observed: int, e_expected: float) -> float:
        return (self.alpha + n_observed) / (self.beta + e_expected)

    def detect_signal(self, drug, event, n_observed, e_expected) -> SignalResult:
        from scipy.stats import gamma as gamma_dist
        ebgm = self.compute_ebgm(n_observed, e_expected)
        post_alpha = self.alpha + n_observed
        post_beta = self.beta + e_expected
        ebgm_lower = gamma_dist.ppf(0.05, post_alpha, scale=1.0 / post_beta)
        return SignalResult(
            drug=drug, event=event, n_observed=n_observed,
            e_expected=e_expected, ebgm=ebgm, ebgm_lower=ebgm_lower,
            is_signal=(ebgm_lower > self.signal_threshold))
''')

wf("src/m1_ddi_adr/causal_inference.py", '''
import numpy as np
from ..cdss_core.log_lr import LogLikelihoodRatio

class SCCSValidator:
    """Self-Controlled Case Series for causal validation."""

    def __init__(self, risk_window_days=28):
        self.risk_window_days = risk_window_days

    def validate_signal(self, events_in_risk, events_in_control,
                        risk_time, control_time):
        if control_time == 0 or risk_time == 0:
            return 0.0, 1.0
        rate_risk = events_in_risk / risk_time
        rate_control = events_in_control / control_time
        if rate_control == 0:
            return 0.0, 1.0
        ri = rate_risk / rate_control
        log_lr = np.log(ri) if ri > 0 else 0.0
        from scipy.stats import poisson
        expected = rate_control * risk_time
        p_val = 1 - poisson.cdf(events_in_risk, expected) if expected > 0 else 1.0
        return log_lr, p_val

    def to_log_lr(self, log_lr, p_value, provenance_id):
        level = "sccs_confirmed" if p_value < 0.05 else "signal"
        return LogLikelihoodRatio(
            value=log_lr, provenance_id=provenance_id,
            module="M1", mechanism="SCCS", evidence_level=level)
''')

wf("src/m1_ddi_adr/qt_calculator.py", '''
import numpy as np
from ..cdss_core.log_lr import LogLikelihoodRatio

class QTcRiskCalculator:
    """Conditional QT prolongation risk calculator."""
    TOXIC_THRESHOLD_MS = 500.0

    def calculate_risk(self, baseline_qtc, drug_liabilities,
                       potassium=None, magnesium=None):
        qt_eff = baseline_qtc + sum(drug_liabilities)
        if potassium is not None and potassium < 3.5:
            qt_eff += (3.5 - potassium) * 10
        if magnesium is not None and magnesium < 1.7:
            qt_eff += (1.7 - magnesium) * 15
        sigma_qt = 15.0
        from scipy.stats import norm
        p_toxic = 1 - norm.cdf(self.TOXIC_THRESHOLD_MS, loc=qt_eff, scale=sigma_qt)
        prior_p = 0.01
        if 0 < p_toxic < 1:
            log_lr = np.log(p_toxic / (1 - p_toxic)) - np.log(prior_p / (1 - prior_p))
        else:
            log_lr = 0.0
        return {"qt_effective_ms": qt_eff, "p_qtc_toxic": p_toxic, "log_lr": log_lr}
''')

# ═══════════════════════════════════════
print("\n[4/8] Creating M2 (PK/PD)...")
# ═══════════════════════════════════════

wf("src/m2_pkpd/__init__.py", '''
from .nlme_model import NLMEModel
from .bayesian_forecast import BayesianForecaster
from .particle_filter import ClearanceParticleFilter
from .dose_optimizer import ChanceConstrainedOptimizer
''')

wf("src/m2_pkpd/nlme_model.py", '''
import numpy as np
from dataclasses import dataclass

@dataclass
class PKParameters:
    cl: float
    v: float
    ka: float
    f_bio: float

class NLMEModel:
    """Population PK model with allometric scaling and covariates."""

    def __init__(self, drug_name, pop_params=None):
        self.drug_name = drug_name
        self.pop_params = pop_params or {
            "cl_pop": 10.0, "v_pop": 50.0, "ka_pop": 1.5,
            "f_bio": 0.8, "theta_gfr": 0.75,
            "omega_cl": 0.3, "omega_v": 0.2}

    def predict_parameters(self, weight, gfr, age=50):
        p = self.pop_params
        cl = p["cl_pop"] * (weight / 70.0) ** 0.75
        v = p["v_pop"] * (weight / 70.0) ** 1.0
        cl *= (gfr / 90.0) ** p["theta_gfr"]
        if age > 65:
            cl *= 0.85
        return PKParameters(cl=cl, v=v, ka=p["ka_pop"], f_bio=p["f_bio"])

    def predict_concentration(self, params, dose, time):
        ke = params.cl / params.v
        if abs(params.ka - ke) < 1e-10:
            c = (params.f_bio * dose * params.ka / params.v) * time * np.exp(-ke * time)
        else:
            c = (params.f_bio * dose * params.ka /
                 (params.v * (params.ka - ke))) * (np.exp(-ke * time) - np.exp(-params.ka * time))
        return max(c, 0.0)
''')

wf("src/m2_pkpd/bayesian_forecast.py", '''
import numpy as np
from scipy.optimize import minimize
from .nlme_model import NLMEModel, PKParameters

class BayesianForecaster:
    """Bayesian MAP estimation from TDM data."""

    def __init__(self, model: NLMEModel):
        self.model = model

    def forecast(self, tdm_levels, prior_params, obs_noise=0.1):
        def neg_log_post(params_vec):
            cl, v = params_vec
            if cl <= 0 or v <= 0:
                return 1e10
            nll = 0.0
            for t, c_obs in tdm_levels:
                pk = PKParameters(cl=cl, v=v, ka=prior_params.ka, f_bio=prior_params.f_bio)
                c_pred = self.model.predict_concentration(pk, dose=1000, time=t)
                sigma = obs_noise * c_pred + 1e-6
                nll += 0.5 * ((c_obs - c_pred) / sigma) ** 2 + np.log(sigma)
            nll += 0.5 * (np.log(cl / prior_params.cl) / 0.3) ** 2
            nll += 0.5 * (np.log(v / prior_params.v) / 0.2) ** 2
            return nll

        x0 = [prior_params.cl, prior_params.v]
        result = minimize(neg_log_post, x0, method="Nelder-Mead", options={"maxiter": 1000})
        return {"map_cl": max(result.x[0], 0.01), "map_v": max(result.x[1], 0.1),
                "converged": result.success, "n_tdm_samples": len(tdm_levels)}
''')

wf("src/m2_pkpd/particle_filter.py", '''
import numpy as np
from scipy.stats import norm
from ..cdss_core.particles import ParticleSet

class ClearanceParticleFilter:
    """Bootstrap particle filter for non-stationary clearance (CL_t)."""

    def __init__(self, n_particles=1000, initial_cl=100.0,
                 process_noise=5.0, obs_noise=10.0):
        self.n_particles = n_particles
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self.particles = np.ones(n_particles) * initial_cl
        self.weights = np.ones(n_particles) / n_particles
        self.history = [self.particles.copy()]

    def predict(self, process_noise_std=None):
        q = process_noise_std or self.process_noise
        self.particles += np.random.normal(0, q, self.n_particles)
        self.particles = np.maximum(self.particles, 0.1)

    def update(self, observation, obs_noise_std=None):
        r = obs_noise_std or self.obs_noise
        likelihoods = norm.pdf(observation, loc=self.particles, scale=r)
        self.weights *= likelihoods
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample_if_needed(self, threshold=0.5):
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess / self.n_particles < threshold:
            positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
            cumsum = np.cumsum(self.weights)
            indices = np.clip(np.searchsorted(cumsum, positions), 0, self.n_particles - 1)
            self.particles = self.particles[indices].copy()
            self.weights = np.ones(self.n_particles) / self.n_particles

    def step(self, observation):
        self.predict()
        self.update(observation)
        self.resample_if_needed()
        self.history.append(self.particles.copy())

    def get_posterior(self):
        return ParticleSet(self.particles.copy(), self.weights.copy())

    def get_credible_band(self, level=0.95):
        return self.get_posterior().credible_interval(level)
''')

wf("src/m2_pkpd/dose_optimizer.py", '''
import numpy as np

class ChanceConstrainedOptimizer:
    """Chance-constrained dosing: P(AUC>target)>0.90, P(Cmax>tox)<0.10."""

    def __init__(self, target_attainment=0.90, toxicity_ceiling=0.10):
        self.target_attainment = target_attainment
        self.toxicity_ceiling = toxicity_ceiling

    def optimize(self, posterior_samples, target_auc, tox_cmax, dose_grid=None):
        if dose_grid is None:
            dose_grid = np.arange(100, 5001, 100)
        best_dose, best_prob, p_tox = dose_grid[0], 0.0, 0.0
        for dose in dose_grid:
            aucs = dose / posterior_samples[:, 0]
            cmaxs = dose / posterior_samples[:, 1]
            p_att = np.mean(aucs > target_auc)
            p_tox = np.mean(cmaxs > tox_cmax)
            if p_att >= self.target_attainment and p_tox <= self.toxicity_ceiling:
                return {"recommended_dose": dose, "p_target_attainment": p_att,
                        "p_toxicity": p_tox, "n_posterior_samples": len(posterior_samples)}
            if p_att > best_prob:
                best_dose, best_prob = dose, p_att
        return {"recommended_dose": best_dose, "p_target_attainment": best_prob,
                "p_toxicity": p_tox, "n_posterior_samples": len(posterior_samples)}
''')

# ═══════════════════════════════════════
print("\n[5/8] Creating M3 (Analysis Lab)...")
# ═══════════════════════════════════════

wf("src/m3_analysis_lab/__init__.py", '''
from .kinetic_gfr import KineticGFR
from .child_pugh import ChildPughPosterior
from .bocpd import BOCPDDetector
from .rcv_gating import RCVGate
''')

wf("src/m3_analysis_lab/kinetic_gfr.py", '''
import numpy as np
from ..cdss_core.particles import ParticleSet

class KineticGFR:
    """Kinetic GFR from serial creatinine (mass-balance model)."""

    def __init__(self, n_particles=1000):
        self.n_particles = n_particles

    def estimate(self, creatinine_series, muscle_mass, dt=1.0):
        if len(creatinine_series) < 2:
            gfr_p = np.random.normal(90, 30, self.n_particles)
            return ParticleSet(np.maximum(gfr_p, 5))
        first_creat = creatinine_series[0][1]
        initial_gfr = min(140.0 / max(first_creat, 0.1), 150.0)
        particles = np.maximum(np.random.normal(initial_gfr, 15, self.n_particles), 5)
        weights = np.ones(self.n_particles) / self.n_particles
        for t, creat_obs in creatinine_series[1:]:
            particles += np.random.normal(0, 5.0, self.n_particles)
            particles = np.maximum(particles, 5)
            v_dist = 0.6 * 70
            c_expected = muscle_mass / (particles * v_dist) * 100
            from scipy.stats import norm
            likelihoods = norm.pdf(creat_obs, loc=c_expected, scale=0.2)
            weights *= likelihoods + 1e-300
            weights /= np.sum(weights)
        return ParticleSet(particles, weights)
''')

wf("src/m3_analysis_lab/child_pugh.py", '''
import numpy as np

class ChildPughPosterior:
    """Posterior PMF over Child-Pugh classes A, B, C via Monte Carlo."""

    def __init__(self, n_samples=5000):
        self.n_samples = n_samples

    def compute(self, bilirubin, albumin, inr, ascites=1, enceph=1):
        bili_s = np.maximum(np.random.normal(bilirubin, 0.1 * bilirubin, self.n_samples), 0.1)
        alb_s = np.clip(np.random.normal(albumin, 0.1, self.n_samples), 0.5, 5.0)
        inr_s = np.maximum(np.random.normal(inr, 0.05 * inr, self.n_samples), 0.8)

        bs = np.ones_like(bili_s)
        bs[(bili_s >= 2) & (bili_s < 3)] = 2
        bs[bili_s >= 3] = 3

        als = np.ones_like(alb_s)
        als[(alb_s >= 2.8) & (alb_s < 3.5)] = 2
        als[alb_s < 2.8] = 3

        ins = np.ones_like(inr_s)
        ins[(inr_s >= 1.7) & (inr_s < 2.3)] = 2
        ins[inr_s >= 2.3] = 3

        total = bs + als + ins + ascites + enceph
        return {"A": float(np.mean(total <= 6)),
                "B": float(np.mean((total >= 7) & (total <= 9))),
                "C": float(np.mean(total >= 10))}
''')

wf("src/m3_analysis_lab/bocpd.py", '''
import numpy as np
from scipy.stats import norm

class BOCPDDetector:
    """Bayesian Online Change-Point Detection (Adams and MacKay 2007)."""

    def __init__(self, hazard_prob=0.01, obs_noise=1.0):
        self.hazard_prob = hazard_prob
        self.obs_noise = obs_noise
        self.run_length_probs = np.array([1.0])
        self.data = []

    def update(self, new_observation):
        self.data.append(new_observation)
        t = len(self.data)
        predictive_probs = np.zeros(len(self.run_length_probs) + 1)
        for r in range(len(self.run_length_probs)):
            segment = self.data[max(0, t - r - 1):t]
            mu = np.mean(segment) if len(segment) > 0 else new_observation
            sigma = max(np.std(segment), self.obs_noise) if len(segment) > 1 else self.obs_noise
            predictive_probs[r] = norm.pdf(new_observation, mu, sigma)
        growth = predictive_probs[:-1] * self.run_length_probs * (1 - self.hazard_prob)
        change = np.sum(predictive_probs[:-1] * self.run_length_probs * self.hazard_prob)
        new_rl = np.zeros(len(self.run_length_probs) + 1)
        new_rl[0] = change
        new_rl[1:] = growth
        total = np.sum(new_rl)
        self.run_length_probs = new_rl / total if total > 0 else np.ones_like(new_rl) / len(new_rl)
        return float(self.run_length_probs[0])
''')

wf("src/m3_analysis_lab/rcv_gating.py", '''
import numpy as np

class RCVGate:
    """Reference Change Value gating based on biological variation (EFLM)."""

    BIOLOGICAL_VARIATION = {
        "creatinine": {"cv_a": 2.2, "cv_i": 6.0},
        "potassium": {"cv_a": 1.5, "cv_i": 4.8},
        "sodium": {"cv_a": 0.7, "cv_i": 0.7},
        "albumin": {"cv_a": 1.6, "cv_i": 3.1},
        "bilirubin": {"cv_a": 3.0, "cv_i": 26.0},
        "inr": {"cv_a": 2.0, "cv_i": 5.0},
        "alt": {"cv_a": 3.5, "cv_i": 18.0},
        "ast": {"cv_a": 3.0, "cv_i": 12.0},
    }

    def __init__(self, z_score=2.77):
        self.z_score = z_score

    def compute_rcv(self, analyte):
        bv = self.BIOLOGICAL_VARIATION.get(analyte, {"cv_a": 2.0, "cv_i": 5.0})
        return self.z_score * np.sqrt((bv["cv_a"] / 100) ** 2 + (bv["cv_i"] / 100) ** 2)

    def is_significant(self, prev, curr, analyte):
        if prev == 0:
            return True
        return abs(curr - prev) / prev > self.compute_rcv(analyte)
''')

# ═══════════════════════════════════════
print("\n[6/8] Creating M4, Blackboard, Advisor...")
# ═══════════════════════════════════════

wf("src/m4_graphics_imaging/__init__.py", '''
from .echo import EchoCovariateExtractor
from .ct_body_comp import CTBodyComposition
from .calibration import ModelCalibrator
''')

wf("src/m4_graphics_imaging/echo.py", '''
import numpy as np

class EchoCovariateExtractor:
    """Extracts EF and fluid status with modality-specific error models."""
    ECHO_EF_VARIANCE = 25.0
    POCUS_EF_VARIANCE = 49.0

    def extract_ef(self, ef_reported, modality="echo"):
        var = self.ECHO_EF_VARIANCE if modality == "echo" else self.POCUS_EF_VARIANCE
        return {"ef_mean": ef_reported, "ef_variance": var,
                "ef_std": np.sqrt(var), "modality": modality}

    def extract_fluid_status(self, ivc_diameter_cm, ivc_collapsibility):
        logit = -3.0 + 1.5 * ivc_diameter_cm - 2.0 * ivc_collapsibility
        return {"p_fluid_overload": 1.0 / (1.0 + np.exp(-logit))}
''')

wf("src/m4_graphics_imaging/ct_body_comp.py", '''
import numpy as np

class CTBodyComposition:
    """Body composition extraction for allometric PK scaling."""
    LEAN_MASS_VARIANCE = 4.0

    def segment(self, ct_scan_id, lean_mass_kg, visceral_fat_cm2):
        return {"lean_mass_kg": lean_mass_kg,
                "lean_mass_variance": self.LEAN_MASS_VARIANCE,
                "visceral_fat_cm2": visceral_fat_cm2,
                "allometric_factor_cl": (lean_mass_kg / 55.0) ** 0.75,
                "allometric_factor_v": (lean_mass_kg / 55.0) ** 1.0}
''')

wf("src/m4_graphics_imaging/calibration.py", '''
import numpy as np
from scipy.optimize import minimize_scalar

class ModelCalibrator:
    """Temperature scaling for local site calibration."""

    def __init__(self, t_min=0.1, t_max=10.0):
        self.t_min = t_min
        self.t_max = t_max
        self.temperature = 1.0

    def fit_temperature(self, logits, labels):
        def nll(T):
            probs = 1.0 / (1.0 + np.exp(-logits / T))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        result = minimize_scalar(nll, bounds=(self.t_min, self.t_max), method="bounded")
        self.temperature = result.x
        return self.temperature

    def calibrate(self, logits):
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))
''')

wf("src/blackboard/__init__.py", '''
from .blackboard import Blackboard, ProvenanceError
''')

wf("src/blackboard/blackboard.py", '''
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..cdss_core.distributions import NormalPosterior
from ..cdss_core.fusion import precision_weighted_fusion
from ..cdss_core.log_lr import LogLikelihoodRatio
import numpy as np

class ProvenanceError(Exception):
    """Raised when duplicate evidence is detected (echo prevention)."""
    pass

class Blackboard:
    """Joint posterior maintenance with strict provenance tracking."""

    def __init__(self, patient_id="unknown"):
        self.patient_id = patient_id
        self.latent_states: Dict[str, Any] = {}
        self.harm_log_lrs: Dict[str, List[LogLikelihoodRatio]] = {}
        self.provenance: set = set()
        self.audit_log: List[Dict] = []

    def update_latent(self, latent_id, evidence, provenance_id, module):
        if provenance_id in self.provenance:
            raise ProvenanceError(
                f"Duplicate evidence: {provenance_id} from {module}. Echo prevented.")
        self.provenance.add(provenance_id)
        if latent_id in self.latent_states:
            existing = self.latent_states[latent_id]
            if isinstance(existing, NormalPosterior) and isinstance(evidence, NormalPosterior):
                self.latent_states[latent_id] = precision_weighted_fusion([existing, evidence])
            else:
                self.latent_states[latent_id] = evidence
        else:
            self.latent_states[latent_id] = evidence
        self.audit_log.append({"ts": datetime.now().isoformat(),
                               "action": "update_latent", "latent": latent_id,
                               "module": module, "provenance": provenance_id})

    def add_harm_evidence(self, harm_id, log_lr: LogLikelihoodRatio):
        if log_lr.provenance_id in self.provenance:
            raise ProvenanceError(f"Duplicate harm evidence: {log_lr.provenance_id}")
        self.provenance.add(log_lr.provenance_id)
        self.harm_log_lrs.setdefault(harm_id, []).append(log_lr)
        self.audit_log.append({"ts": datetime.now().isoformat(),
                               "action": "add_harm", "harm": harm_id,
                               "module": log_lr.module, "log_lr": log_lr.value})

    def get_latent(self, latent_id):
        return self.latent_states.get(latent_id)

    def get_fused_log_lr(self, harm_id):
        return sum(lr.value for lr in self.harm_log_lrs.get(harm_id, []))

    def get_posterior_probability(self, harm_id, prior_prob=0.05):
        fused = self.get_fused_log_lr(harm_id)
        prior_odds = prior_prob / (1 - prior_prob)
        post_odds = prior_odds * np.exp(fused)
        return float(post_odds / (1 + post_odds))

    def get_full_state(self):
        return {"patient_id": self.patient_id,
                "latent_states": {k: str(v) for k, v in self.latent_states.items()},
                "harm_propositions": {h: {"fused_lr": self.get_fused_log_lr(h),
                                          "posterior_p": self.get_posterior_probability(h)}
                                      for h in self.harm_log_lrs},
                "total_evidence": len(self.provenance)}
''')

wf("src/advisor/__init__.py", '''
from .utility import ExpectedUtilityCalculator
from .knapsack import AlertBudgetSelector
from .cusum import CUSUMMonitor
from .dca import DecisionCurveAnalysis
''')

wf("src/advisor/utility.py", '''
from dataclasses import dataclass

@dataclass
class Action:
    name: str
    utility_if_harm: float
    utility_if_no_harm: float
    attention_cost: int = 1

class ExpectedUtilityCalculator:
    """Selects optimal actions based on expected utility theory."""

    def __init__(self):
        self.actions = []

    def add_action(self, name, u_harm, u_noharm, cost=1):
        self.actions.append(Action(name, u_harm, u_noharm, cost))

    def calculate_eu(self, action, p_harm):
        return p_harm * action.utility_if_harm + (1 - p_harm) * action.utility_if_no_harm

    def select_best_action(self, p_harm):
        if not self.actions:
            return {"selected": None}
        eu_values = {a.name: self.calculate_eu(a, p_harm) for a in self.actions}
        best = max(self.actions, key=lambda a: eu_values[a.name])
        return {"selected": best.name, "expected_utility": eu_values[best.name],
                "all_eu_values": eu_values}
''')

wf("src/advisor/knapsack.py", '''
class AlertBudgetSelector:
    """Greedy knapsack selection under alert budget constraint."""

    def __init__(self, budget=3):
        self.budget = budget

    def select_alerts(self, alerts):
        sorted_alerts = sorted(alerts,
            key=lambda a: a.get("net_benefit", 0) / max(a.get("attention_cost", 1), 1e-6),
            reverse=True)
        selected, remaining = [], self.budget
        for alert in sorted_alerts:
            cost = alert.get("attention_cost", 1)
            if cost <= remaining:
                selected.append(alert)
                remaining -= cost
        return selected
''')

wf("src/advisor/cusum.py", '''
class CUSUMMonitor:
    """Cumulative Sum chart for calibration drift detection."""

    def __init__(self, threshold=5.0, slack=0.5):
        self.threshold = threshold
        self.slack = slack
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.alarm_count = 0

    def update(self, residual):
        self.s_pos = max(0, self.s_pos + residual - self.slack)
        self.s_neg = max(0, self.s_neg - residual - self.slack)
        alarm = self.s_pos > self.threshold or self.s_neg > self.threshold
        if alarm:
            self.alarm_count += 1
        return alarm

    def reset(self):
        self.s_pos = self.s_neg = 0.0
        self.alarm_count = 0
''')

wf("src/advisor/dca.py", '''
import numpy as np

class DecisionCurveAnalysis:
    """Decision Curve Analysis for clinical utility evaluation."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99)

    def net_benefit(self, predictions, outcomes, threshold):
        n = len(predictions)
        if n == 0:
            return 0.0
        pred_pos = predictions >= threshold
        tp = np.sum(pred_pos & (outcomes == 1))
        fp = np.sum(pred_pos & (outcomes == 0))
        return (tp / n) - (fp / n) * (threshold / (1 - threshold))

    def compare_strategies(self, predictions, outcomes):
        nb_cdss = np.array([self.net_benefit(predictions, outcomes, t) for t in self.thresholds])
        prevalence = np.mean(outcomes)
        nb_all = prevalence - (1 - prevalence) * self.thresholds / (1 - self.thresholds)
        return {"thresholds": self.thresholds, "nb_cdss": nb_cdss,
                "nb_treat_all": nb_all, "nb_treat_none": np.zeros_like(self.thresholds)}
''')

# ═══════════════════════════════════════
print("\n[7/8] Creating config, tests, database...")
# ═══════════════════════════════════════

wf("config/settings.yaml", '''
system:
  name: RezpharmaCDSS
  version: 1.0.0
advisor:
  alert_budget: 3
  default_prior_harm: 0.05
m2_pkpd:
  target_attainment_prob: 0.90
  toxicity_ceiling_prob: 0.10
  particle_filter_particles: 1000
m3_lab:
  rcv_z_score: 2.77
  kinetic_gfr_particles: 1000
monitoring:
  cusum_threshold: 5.0
''')

wf("requirements.txt", '''
numpy>=1.24
scipy>=1.10
pandas>=2.0
scikit-learn>=1.3
pymc>=5.0
fastapi>=0.100
uvicorn>=0.23
redis>=4.6
jupyterlab>=4.0
ipykernel>=6.25
jupyterlab-code-formatter>=2.0
jupyterlab-lsp>=5.0
python-lsp-server[all]>=1.8
black>=23.0
matplotlib>=3.7
seaborn>=0.12
pytest>=7.4
pytest-cov>=4.1
pyyaml>=6.0
sqlalchemy>=2.0
''')

wf(".gitignore", '''
rezpharma_env/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
*.db
.env
htmlcov/
.coverage
''')

wf("tests/__init__.py", "")

wf("tests/test_blackboard.py", '''
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from blackboard.blackboard import Blackboard, ProvenanceError
from cdss_core.distributions import NormalPosterior
from cdss_core.fusion import precision_weighted_fusion
from cdss_core.log_lr import LogLikelihoodRatio

def test_duplicate_provenance_raises():
    bb = Blackboard()
    bb.update_latent("GFR", NormalPosterior(45, 5), "lab_001", "M3")
    with pytest.raises(ProvenanceError):
        bb.update_latent("GFR", NormalPosterior(46, 4), "lab_001", "M3")

def test_fusion_order_invariant():
    a = NormalPosterior(45, 5)
    b = NormalPosterior(50, 3)
    f1 = precision_weighted_fusion([a, b])
    f2 = precision_weighted_fusion([b, a])
    assert abs(f1.mu - f2.mu) < 1e-10

def test_fusion_reduces_uncertainty():
    a = NormalPosterior(45, 5)
    b = NormalPosterior(50, 3)
    fused = precision_weighted_fusion([a, b])
    assert fused.sigma < a.sigma and fused.sigma < b.sigma

def test_harm_lr_accumulation():
    bb = Blackboard()
    bb.add_harm_evidence("QT", LogLikelihoodRatio(1.5, "m1_qt", "M1"))
    bb.add_harm_evidence("QT", LogLikelihoodRatio(0.8, "m3_k", "M3"))
    assert abs(bb.get_fused_log_lr("QT") - 2.3) < 1e-10
''')

wf("tests/test_advisor.py", '''
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from advisor.utility import ExpectedUtilityCalculator
from advisor.knapsack import AlertBudgetSelector
from advisor.cusum import CUSUMMonitor

def test_expected_utility():
    calc = ExpectedUtilityCalculator()
    calc.add_action("hold", u_harm=-10, u_noharm=5)
    result = calc.select_best_action(p_harm=0.8)
    assert abs(result["expected_utility"] - (0.8 * -10 + 0.2 * 5)) < 1e-9

def test_knapsack_budget():
    sel = AlertBudgetSelector(budget=2)
    alerts = [{"net_benefit": i, "attention_cost": 1} for i in range(5)]
    assert len(sel.select_alerts(alerts)) <= 2

def test_cusum_alarm():
    mon = CUSUMMonitor(threshold=5.0, slack=0.5)
    for _ in range(20):
        mon.update(1.0)
    assert mon.alarm_count > 0
''')

wf("database/schema.sql", '''
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS drugs (
    drug_id TEXT PRIMARY KEY,
    generic_name TEXT NOT NULL UNIQUE,
    therapeutic_class TEXT,
    cyp_substrate TEXT,
    cyp_inhibitor TEXT,
    cyp_inducer TEXT,
    qt_liability_mv REAL DEFAULT 0,
    renal_clearance_fraction REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ddi_interactions (
    interaction_id TEXT PRIMARY KEY,
    perpetrator_drug_id TEXT REFERENCES drugs(drug_id),
    victim_drug_id TEXT REFERENCES drugs(drug_id),
    mechanism TEXT NOT NULL,
    severity TEXT CHECK(severity IN ('mild','moderate','severe','contraindicated')),
    log_lr REAL NOT NULL DEFAULT 0,
    evidence_level TEXT CHECK(evidence_level IN ('curated','signal','sccs_confirmed')),
    UNIQUE(perpetrator_drug_id, victim_drug_id, mechanism)
);

CREATE TABLE IF NOT EXISTS pkpd_parameters (
    param_id TEXT PRIMARY KEY,
    drug_id TEXT REFERENCES drugs(drug_id),
    param_name TEXT NOT NULL,
    population_mean REAL,
    population_cv REAL,
    iiv_omega REAL
);

CREATE TABLE IF NOT EXISTS biological_variation (
    analyte TEXT PRIMARY KEY,
    cv_within REAL NOT NULL,
    cv_between REAL NOT NULL,
    unit TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER, sex TEXT, weight_kg REAL
);

CREATE TABLE IF NOT EXISTS lab_observations (
    obs_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    analyte TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    provenance_id TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS imaging_findings (
    finding_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    modality TEXT NOT NULL,
    covariate_name TEXT NOT NULL,
    value_mean REAL NOT NULL,
    value_variance REAL NOT NULL,
    provenance_id TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    module TEXT NOT NULL,
    harm_proposition TEXT NOT NULL,
    posterior_probability REAL,
    log_lr REAL,
    provenance_id TEXT UNIQUE NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS advisor_recommendations (
    rec_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    harm_proposition TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    expected_utility REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clinician_feedback (
    feedback_id TEXT PRIMARY KEY,
    rec_id TEXT REFERENCES advisor_recommendations(rec_id),
    action_taken TEXT CHECK(action_taken IN ('accepted','overridden','ignored')),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labs_patient ON lab_observations(patient_id);
CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id);
''')

# Keep placeholders
wf("data/raw/.gitkeep", "")
wf("data/processed/.gitkeep", "")
wf("database/seed_data/.gitkeep", "")

# ═══════════════════════════════════════
print("\n[8/8] Creating notebooks and automation...")
# ═══════════════════════════════════════

NB_META = {
    "kernelspec": {"display_name": "Python (Rezpharma CDSS)", "language": "python", "name": "rezpharma_kernel"},
    "language_info": {"name": "python", "version": "3.11.0"}
}

def make_nb(title, code_cells):
    cells = [{"cell_type": "markdown", "metadata": {},
              "source": [f"# {title}\n", "\n", "> RezpharmaCDSS | Kernel: `rezpharma_kernel`\n"]}]
    for code in code_cells:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split("\n")})
    return {"cells": cells, "metadata": NB_META, "nbformat": 4, "nbformat_minor": 5}

wj("notebooks/01_m1_ddi_signal_mining.ipynb", make_nb("M1: DDI Signal Mining", [
    "import sys; sys.path.insert(0, '../src')\nfrom m1_ddi_adr.signal_mining import EBGMSignalMiner\nfrom m1_ddi_adr.qt_calculator import QTcRiskCalculator\nprint('M1 modules loaded')",
    "miner = EBGMSignalMiner(alpha=0.2, beta=0.1)\nresult = miner.detect_signal('drug_x', 'rhabdomyolysis', n_observed=5, e_expected=1.2)\nprint(f'EBGM: {result.ebgm:.3f}, Signal: {result.is_signal}')"
]))

wj("notebooks/02_m2_particle_filter.ipynb", make_nb("M2: Particle Filter for CL Tracking", [
    "import sys; sys.path.insert(0, '../src')\nimport numpy as np\nfrom m2_pkpd.particle_filter import ClearanceParticleFilter\nprint('M2 loaded')",
    "np.random.seed(42)\npf = ClearanceParticleFilter(n_particles=1000, initial_cl=100.0)\ntrue_cl = [100, 95, 85, 70, 55, 45]\nfor cl in true_cl:\n    pf.step(observation=cl + np.random.normal(0, 5))\npost = pf.get_posterior()\nprint(f'Estimated CL: {post.mean():.1f}, 95% CI: {post.credible_interval()}')"
]))

wj("notebooks/03_m3_kinetic_gfr.ipynb", make_nb("M3: Kinetic GFR Estimation", [
    "import sys; sys.path.insert(0, '../src')\nfrom m3_analysis_lab.kinetic_gfr import KineticGFR\nfrom m3_analysis_lab.rcv_gating import RCVGate\nprint('M3 loaded')",
    "kgfr = KineticGFR(n_particles=1000)\ncreat = [(0, 1.0), (6, 1.2), (12, 1.5), (18, 2.0), (24, 2.8)]\npost = kgfr.estimate(creat, muscle_mass=1.2)\nprint(f'Kinetic GFR: {post.mean():.1f} mL/min, CI: {post.credible_interval()}')"
]))

wj("notebooks/05_blackboard_integration.ipynb", make_nb("Blackboard: Evidence Fusion Demo", [
    "import sys; sys.path.insert(0, '../src')\nfrom blackboard.blackboard import Blackboard, ProvenanceError\nfrom cdss_core.distributions import NormalPosterior\nfrom cdss_core.log_lr import LogLikelihoodRatio\nprint('Blackboard loaded')",
    "bb = Blackboard(patient_id='PT_001')\nbb.update_latent('GFR', NormalPosterior(42, 8), 'm3_creat', 'M3')\nbb.update_latent('GFR', NormalPosterior(45, 5), 'm2_tdm', 'M2')\nprint(f'Fused GFR: {bb.get_latent(\"GFR\")}')",
    "bb.add_harm_evidence('QT', LogLikelihoodRatio(1.5, 'm1_qt', 'M1'))\nbb.add_harm_evidence('QT', LogLikelihoodRatio(0.8, 'm3_k', 'M3'))\nprint(f'P(QT): {bb.get_posterior_probability(\"QT\", 0.05):.4f}')"
]))

wj("notebooks/06_advisor_demo.ipynb", make_nb("Advisor: Decision Theory Demo", [
    "import sys; sys.path.insert(0, '../src')\nfrom advisor.utility import ExpectedUtilityCalculator\nfrom advisor.knapsack import AlertBudgetSelector\nfrom advisor.cusum import CUSUMMonitor\nprint('Advisor loaded')",
    "calc = ExpectedUtilityCalculator()\ncalc.add_action('hold_drug', u_harm=-5, u_noharm=-2)\ncalc.add_action('continue', u_harm=-30, u_noharm=0)\nresult = calc.select_best_action(p_harm=0.65)\nprint(f'Best action: {result[\"selected\"]}, EU: {result[\"expected_utility\"]:.2f}')"
]))

wf("start_jupyter.bat", '''@echo off
title RezpharmaCDSS - Jupyter Lab
call rezpharma_env\\Scripts\\activate.bat
jupyter lab
pause
''')

wf("run_tests.bat", '''@echo off
title RezpharmaCDSS - Tests
call rezpharma_env\\Scripts\\activate.bat
pytest tests/ -v --tb=short
pause
''')

print("\n" + "=" * 50)
print("DONE! All files generated successfully.")
print("=" * 50)
print("\nNext steps:")
print("  1. pip install -r requirements.txt")
print("  2. python -m ipykernel install --user --name rezpharma_kernel --display-name \"Python (Rezpharma CDSS)\"")
print("  3. start_jupyter.bat")
print("  4. run_tests.bat")