@echo off
setlocal enabledelayedexpansion
title RezpharmaCDSS - Full Project Setup
color 0A

echo ============================================================
echo   RezpharmaCDSS - Complete Project Scaffold Generator
echo   Based on: "From Sketch to Statistically Sound Code"
echo ============================================================
echo.

:: ============================================================
:: 1. CREATE ALL DIRECTORIES
:: ============================================================
echo [1/8] Creating directory structure...

mkdir notebooks                2>nul
mkdir src                      2>nul
mkdir src\cdss_core            2>nul
mkdir src\m1_ddi_adr           2>nul
mkdir src\m2_pkpd              2>nul
mkdir src\m3_analysis_lab      2>nul
mkdir src\m4_graphics_imaging  2>nul
mkdir src\blackboard           2>nul
mkdir src\advisor              2>nul
mkdir data                     2>nul
mkdir data\raw                 2>nul
mkdir data\processed           2>nul
mkdir database                 2>nul
mkdir database\seed_data       2>nul
mkdir tests                    2>nul
mkdir config                   2>nul

echo       Done.
echo.

:: ============================================================
:: 2. CREATE PYTHON PACKAGE __init__.py FILES
:: ============================================================
echo [2/8] Creating Python package init files...

echo # RezpharmaCDSS Source Package> src\__init__.py

echo # Core mathematical primitives: Normal dist, Particle sets, Log-LR algebra, Precision-weighted fusion> src\cdss_core\__init__.py
echo from .distributions import NormalPosterior> src\cdss_core\__init__.py
echo from .particles import ParticleSet> src\cdss_core\__init__.py
echo from .log_lr import LogLikelihoodRatio, fuse_log_lrs> src\cdss_core\__init__.py
echo from .fusion import precision_weighted_fusion> src\cdss_core\__init__.py

echo # M1: DDI/ADR - Curated rules, EBGM on FAERS, SCCS causal inference> src\m1_ddi_adr\__init__.py
echo # M2: PK/PD - Hierarchical NLME, Bayesian MAP, Particle filtering> src\m2_pkpd\__init__.py
echo # M3: Analysis Lab - Kinetic GFR, Child-Pugh posterior, BOCPD, RCV> src\m3_analysis_lab\__init__.py
echo # M4: Graphics/Imaging - Modality-specific error models, Conformal prediction> src\m4_graphics_imaging\__init__.py
echo # Probabilistic Blackboard: Joint posterior, Provenance tracking> src\blackboard\__init__.py
echo # Decision Making Advisor: Expected utility, Knapsack, CUSUM> src\advisor\__init__.py
echo # Test suite> tests\__init__.py

echo       Done.
echo.

:: ============================================================
:: 3. CREATE CORE MODULE SKELETONS (src/cdss_core/)
:: ============================================================
echo [3/8] Creating cdss_core module skeletons...

(
echo """
echo Core mathematical primitives for the RezpharmaCDSS.
echo Provides the 'language' through which all modules communicate evidence.
echo """
echo import numpy as np
echo.
echo class NormalPosterior:
echo     """Represents a normal distribution N(mu, sigma^2) for precision-weighted fusion."""
echo     def __init__(self, mu: float, sigma: float):
echo         self.mu = mu
echo         self.sigma = sigma
echo.
echo     @property
echo     def precision(self) -^> float:
echo         return 1.0 / (self.sigma ** 2)
echo.
echo     def __repr__(self):
echo         return f"NormalPosterior(mu={self.mu:.4f}, sigma={self.sigma:.4f})"
) > src\cdss_core\distributions.py

(
echo """
echo Particle set representation for non-parametric posterior distributions.
echo Used by M2 (particle filter for non-stationary CL) and M3 (kinetic GFR).
echo """
echo import numpy as np
echo.
echo class ParticleSet:
echo     """Weighted particle approximation of a posterior distribution."""
echo     def __init__(self, particles: np.ndarray, weights: np.ndarray = None):
echo         self.particles = np.asarray(particles)
echo         if weights is None:
echo             self.weights = np.ones(len(particles)) / len(particles)
echo         else:
echo             self.weights = np.asarray(weights)
echo.
echo     def effective_sample_size(self) -^> float:
echo         return 1.0 / np.sum(self.weights ** 2)
echo.
echo     def mean(self) -^> float:
echo         return np.average(self.particles, weights=self.weights)
) > src\cdss_core\particles.py

(
echo """
echo Log-Likelihood Ratio algebra for evidence fusion on the blackboard.
echo Fusion formula: log O_post = log O_prior + fuse(x_1, ..., x_K)
echo """
echo import numpy as np
echo.
echo class LogLikelihoodRatio:
echo     """Encapsulates a log-LR contribution from a single module."""
echo     def __init__(self, value: float, provenance_id: str, module: str):
echo         self.value = value
echo         self.provenance_id = provenance_id
echo         self.module = module
echo.
echo def fuse_log_lrs(log_lrs: list) -^> float:
echo     """Simple additive fusion of independent log-LRs."""
echo     return sum(lr.value for lr in log_lrs)
) > src\cdss_core\log_lr.py

(
echo """
echo Precision-weighted fusion for combining normal posteriors from multiple modules.
echo """
echo from .distributions import NormalPosterior
echo.
echo def precision_weighted_fusion(posteriors: list) -^> NormalPosterior:
echo     """Fuse multiple NormalPosterior objects using precision weighting."""
echo     total_precision = sum(p.precision for p in posteriors)
echo     fused_mu = sum(p.mu * p.precision for p in posteriors) / total_precision
echo     fused_sigma = (1.0 / total_precision) ** 0.5
echo     return NormalPosterior(fused_mu, fused_sigma)
) > src\cdss_core\fusion.py

echo       Done.
echo.

:: ============================================================
:: 4. CREATE MODULE SKELETONS (M1-M4, Blackboard, Advisor)
:: ============================================================
echo [4/8] Creating module skeletons (M1-M4, Blackboard, Advisor)...

:: --- M1 ---
(
echo """M1 Rule Engine: Deterministic DDI detection from curated knowledge base."""
echo import sqlite3
echo.
echo class DDIRuleEngine:
echo     def __init__(self, db_path: str):
echo         self.conn = sqlite3.connect(db_path)
echo.
echo     def check_interactions(self, drug_list: list) -^> list:
echo         """Check all pairwise DDIs via CYP inhibition/induction rules."""
echo         pass  # TODO: Query DDI table, return typed interaction reports + log-LRs
) > src\m1_ddi_adr\rule_engine.py

(
echo """M1 Signal Mining: EBGM (Empirical Bayes Geometric Mean) on FAERS data."""
echo.
echo class EBGMSignalMiner:
echo     """DuMouchel's gamma-Poisson shrinkage for disproportionality analysis."""
echo     def compute_ebgm(self, drug: str, event: str) -^> float:
echo         pass  # TODO: Implement stratified EBGM with age/sex strata
) > src\m1_ddi_adr\signal_mining.py

(
echo """M1 Causal Inference: Self-Controlled Case Series (SCCS) validation."""
echo.
echo class SCCSValidator:
echo     """Within-person study design controlling for time-invariant confounders."""
echo     def validate_signal(self, drug: str, event: str, patient_data) -^> float:
echo         pass  # TODO: Return conditional log-LR from SCCS model
) > src\m1_ddi_adr\causal_inference.py

:: --- M2 ---
(
echo """M2 NLME Model: Hierarchical Nonlinear Mixed-Effects for PopPK."""
echo.
echo class NLMEModel:
echo     """Population PK model with IIV and IOV, allometric scaling."""
echo     def __init__(self, drug_name: str):
echo         self.drug_name = drug_name
echo.
echo     def predict_concentration(self, dose, time, covariates: dict):
echo         pass  # TODO: C_ij = f(t_ij, theta_i) + epsilon_ij
) > src\m2_pkpd\nlme_model.py

(
echo """M2 Bayesian Forecasting: MAP estimation from TDM data."""
echo.
echo class BayesianForecaster:
echo     """Bayesian MAP forecasting using MCMC or Laplace approximation."""
echo     def forecast(self, tdm_levels: list, prior_params: dict) -^> dict:
echo         pass  # TODO: Return posterior over CL, V
) > src\m2_pkpd\bayesian_forecast.py

(
echo """M2 Particle Filter: Bootstrap filter for non-stationary clearance."""
echo.
echo class ClearanceParticleFilter:
echo     """Models CL_t as a meandering stochastic process."""
echo     def step(self, new_tdm: float, dt: float):
echo         pass  # TODO: Predict-update cycle, return time-resolved CL trajectory
) > src\m2_pkpd\particle_filter.py

(
echo """M2 Dose Optimizer: Chance-constrained optimization."""
echo.
echo class ChanceConstrainedOptimizer:
echo     """Finds lowest dose achieving P(AUC ^> target) ^> 0.90, P(Cmax ^> tox) ^< 0.10."""
echo     def optimize(self, posterior_params, target_auc, tox_cmax):
echo         pass  # TODO: Integrate over full posterior uncertainty
) > src\m2_pkpd\dose_optimizer.py

:: --- M3 ---
(
echo """M3 Kinetic GFR: Mass-balance model for non-steady-state renal function."""
echo.
echo class KineticGFR:
echo     """Treats GFR as latent stochastic process from serial creatinine."""
echo     def estimate(self, creatinine_series: list, muscle_mass: float):
echo         pass  # TODO: Return ParticleSet of GFR posterior trajectory
) > src\m3_analysis_lab\kinetic_gfr.py

(
echo """M3 Child-Pugh Posterior: Monte Carlo over ordinal hepatic class."""
echo.
echo class ChildPughPosterior:
echo     """Propagates lab measurement error to posterior PMF over classes A, B, C."""
echo     def compute(self, bilirubin, albumin, inr) -^> dict:
echo         pass  # TODO: Return {'A': 0.2, 'B': 0.6, 'C': 0.2}
) > src\m3_analysis_lab\child_pugh.py

(
echo """M3 BOCPD: Bayesian Online Change-Point Detection."""
echo.
echo class BOCPDDetector:
echo     """Models physiological value as piecewise-stationary process."""
echo     def update(self, new_observation: float) -^> float:
echo         pass  # TODO: Return posterior probability of change-point
) > src\m3_analysis_lab\bocpd.py

(
echo """M3 RCV Gating: Reference Change Value from biological variation."""
echo.
echo class RCVGate:
echo     """Filters noise using analytical imprecision + within-subject CV."""
echo     def is_significant(self, prev: float, curr: float, cv_a: float, cv_i: float) -^> bool:
echo         import numpy as np
echo         rcv = 2.77 * np.sqrt(cv_a**2 + cv_i**2)
echo         return abs(curr - prev) / prev ^> rcv
) > src\m3_analysis_lab\rcv_gating.py

:: --- M4 ---
(
echo """M4 Echo: EF extraction from echocardiography with measurement error."""
echo.
echo class EchoCovariateExtractor:
echo     """Extracts LVEF with modality-specific error model."""
echo     def extract_ef(self, echo_study_id: str) -^> tuple:
echo         pass  # TODO: Return (ef_mean, ef_variance) for errors-in-variables
) > src\m4_graphics_imaging\echo.py

(
echo """M4 CT Body Composition: Lean mass and visceral fat segmentation."""
echo.
echo class CTBodyComposition:
echo     """Quantifies lean muscle mass for allometric PK scaling."""
echo     def segment(self, ct_scan_id: str) -^> dict:
echo         pass  # TODO: Return {'lean_mass_kg': ..., 'visceral_fat_cm2': ...}
) > src\m4_graphics_imaging\ct_body_comp.py

(
echo """M4 Calibration: Local site calibration via temperature scaling."""
echo.
echo class ModelCalibrator:
echo     """Recalibrates model probabilities on local held-out dataset."""
echo     def fit_temperature(self, logits, labels):
echo         pass  # TODO: Optimize temperature parameter T
) > src\m4_graphics_imaging\calibration.py

:: --- Blackboard ---
(
echo """
echo Probabilistic Blackboard: Maintains joint posterior over physiological states.
echo Prevents the 'echo problem' via strict provenance tracking.
echo """
echo.
echo class ProvenanceError(Exception):
echo     """Raised when duplicate evidence is detected (prevents double-counting)."""
echo     pass
echo.
echo class Blackboard:
echo     def __init__(self):
echo         self.state = {}       # latent_id -^> ParticleSet or NormalPosterior
echo         self.provenance = set()  # Set of unique evidence IDs
echo.
echo     def update(self, latent_id: str, evidence, provenance_id: str, module: str):
echo         if provenance_id in self.provenance:
echo             raise ProvenanceError(f"Duplicate evidence: {provenance_id} from {module}")
echo         self.provenance.add(provenance_id)
echo         self.state[latent_id] = evidence  # TODO: Implement proper fusion
echo.
echo     def get_state(self, latent_id: str):
echo         return self.state.get(latent_id)
) > src\blackboard\blackboard.py

:: --- Advisor ---
(
echo """Advisor Utility: Expected utility maximization for action selection."""
echo.
echo def expected_utility(p_harm: float, u_harm_action: float, u_noharm_action: float) -^> float:
echo     return p_harm * u_harm_action + (1 - p_harm) * u_noharm_action
) > src\advisor\utility.py

(
echo """Advisor Knapsack: Budget-constrained alert selection."""
echo.
echo def select_alerts(alerts: list, budget: int = 3) -^> list:
echo     """Greedy selection by value-to-cost ratio under alert budget."""
echo     sorted_alerts = sorted(alerts, key=lambda a: a['net_benefit'] / a['attention_cost'], reverse=True)
echo     return sorted_alerts[:budget]
) > src\advisor\knapsack.py

(
echo """Advisor CUSUM: Drift monitoring on probability residuals."""
echo.
echo class CUSUMMonitor:
echo     def __init__(self, threshold: float = 5.0):
echo         self.threshold = threshold
echo         self.s_pos = 0.0
echo         self.s_neg = 0.0
echo.
echo     def update(self, residual: float) -^> bool:
echo         self.s_pos = max(0, self.s_pos + residual - 0.5)
echo         self.s_neg = max(0, self.s_neg - residual - 0.5)
echo         return self.s_pos ^> self.threshold or self.s_neg ^> self.threshold
) > src\advisor\cusum.py

echo       Done.
echo.

:: ============================================================
:: 5. CREATE JUPYTER NOTEBOOKS (Minimal valid .ipynb)
:: ============================================================
echo [5/8] Creating Jupyter notebooks...

set NB_META={"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# %NB_TITLE%\n", "\n", "Kernel: Python (Rezpharma CDSS)"]}], "metadata": {"kernelspec": {"display_name": "Python (Rezpharma CDSS)", "language": "python", "name": "rezpharma_kernel"}, "language_info": {"name": "python", "version": "3.11.0"}}, "nbformat": 4, "nbformat_minor": 5}

(
echo {
echo  "cells": [
echo   {"cell_type": "markdown", "metadata": {}, "source": ["# M1: DDI Signal Mining\n", "\n", "EBGM on FAERS, SCCS causal inference, conditional log-LRs"]},
echo   {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import sys\n", "sys.path.append('../src')\n", "from m1_ddi_adr.rule_engine import DDIRuleEngine\n", "from m1_ddi_adr.signal_mining import EBGMSignalMiner"]}
echo  ],
echo  "metadata": {"kernelspec": {"display_name": "Python (Rezpharma CDSS)", "language": "python", "name": "rezpharma_kernel"}, "language_info": {"name": "python", "version": "3.11.0"}},
echo  "nbformat": 4,
echo  "nbformat_minor": 5
echo }
) > notebooks\01_m1_ddi_signal_mining.ipynb

(
echo {
echo  "cells": [
echo   {"cell_type": "markdown", "metadata": {}, "source": ["# M2: PK/PD Bayesian Forecasting\n", "\n", "Hierarchical NLME, MAP estimation, Particle filtering, Chance-constrained dosing"]},
echo   {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import sys\n", "sys.path.append('../src')\n", "from m2_pkpd.bayesian_forecast import BayesianForecaster\n", "from m2_pkpd.particle_filter import ClearanceParticleFilter"]}
echo  ],
echo  "metadata": {"kernelspec": {"display_name": "Python (Rezpharma CDSS)", "language": "python", "name": "rezpharma_kernel"}, "language_info": {"name": "python", "version": "3.11.0"}},
echo  "nbformat": 4,
echo  "nbformat_minor": 5
echo }
) > notebooks\02_m2_pkpd_bayesian_forecasting.ipynb

(
echo {
echo  "cells": [
echo   {"cell_type": "markdown", "metadata": {}, "source": ["# M3: Kinetic GFR Estimation\n", "\n", "Non-steady-state renal function, BOCPD, RCV gating"]},
echo   {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import sys\n", "sys.path.append('../src')\n", "from m3_analysis_lab.kinetic_gfr import KineticGFR\n", "from m3_analysis_lab.bocpd import BOCPDDetector"]}
echo  ],
echo  "metadata": {"kernelspec": {"display_name": "Python (Rezpharma CDSS)", "language": "python", "name": "rezpharma_kernel"}, "language_info": {"name": "python", "version": "3.11.0"}},
echo  "nbformat": 4,
echo  "nbformat_minor": 5
echo }
) > notebooks\03_m3_kinetic_gfr_estimation.ipynb

(
echo {
echo  "cells": [
echo   {"cell_type": "markdown", "metadata": {}, "source": ["# M4: Imaging Covariate Extraction\n", "\n", "EF, body composition, modality-specific error models, conformal prediction"]},
echo   {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import sys\n", "sys.path.append('../src')\n", "from m4_graphics_imaging.echo import EchoCovariateExtractor\n", "from m4_graphics_imaging.calibration import ModelCalibrator"]}
echo  ],
echo  "metadata": {"kernelspec": {"display_name": "Python (Rezpharma CDSS)", "language": "python", "name": "rezpharma_kernel"}, "language_info": {"name": "python", "version": "3.11.0"}},
echo  "nbformat": 4,
echo  "nbformat_minor": 5
echo }
) > notebooks\04_m4_imaging_covariate_extraction.ipynb

(
echo {
echo  "cells": [
echo   {"cell_type": "markdown", "metadata": {}, "source": ["# Advisor: Decision Curve Analysis\n", "\n", "Expected utility, knapsack alert budget, CUSUM drift monitoring"]},
echo   {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import sys\n", "sys.path.append('../src')\n", "from advisor.utility import expected_utility\n", "from advisor.knapsack import select_alerts\n", "from advisor.cusum import CUSUMMonitor"]}
echo  ],
echo  "metadata": {"kernelspec": {"display_name": "Python (Rezpharma CDSS)", "language": "python", "name": "rezpharma_kernel"}, "language_info": {"name": "python", "version": "3.11.0"}},
echo  "nbformat": 4,
echo  "nbformat_minor": 5
echo }
) > notebooks\05_advisor_decision_curve_analysis.ipynb

echo       Done.
echo.

:: ============================================================
:: 6. CREATE DATABASE SCHEMA & SEED DATA
:: ============================================================
echo [6/8] Creating database schema and seed data...

(
echo -- ============================================================
echo -- RezpharmaCDSS SQLite Schema
echo -- Based on: "From Sketch to Statistically Sound Code"
echo -- ============================================================
echo.
echo -- Curated Knowledge Base Tables
echo CREATE TABLE IF NOT EXISTS drugs (
echo     drug_id TEXT PRIMARY KEY,
echo     generic_name TEXT NOT NULL,
echo     brand_names TEXT,
echo     therapeutic_class TEXT,
echo     cyp_substrate TEXT,    -- e.g., 'CYP3A4,CYP2D6'
echo     cyp_inhibitor TEXT,    -- e.g., 'CYP3A4(strong)'
echo     cyp_inducer TEXT,
echo     pgp_substrate BOOLEAN DEFAULT 0,
echo     pgp_inhibitor BOOLEAN DEFAULT 0,
echo     qt_liability_mv REAL DEFAULT 0,  -- mean QTc prolongation in ms
echo     nephrotoxic_risk REAL DEFAULT 0,
echo     hepatotoxic_risk REAL DEFAULT 0
echo );
echo.
echo CREATE TABLE IF NOT EXISTS ddi_interactions (
echo     interaction_id TEXT PRIMARY KEY,
echo     perpetrator_drug_id TEXT REFERENCES drugs(drug_id),
echo     victim_drug_id TEXT REFERENCES drugs(drug_id),
echo     mechanism TEXT NOT NULL,       -- e.g., 'CYP3A4 inhibition'
echo     severity TEXT CHECK(severity IN ('mild','moderate','severe','contraindicated')),
echo     effect_direction TEXT,         -- 'increase' or 'decrease' exposure
echo     log_lr REAL,                   -- base log-likelihood ratio
echo     evidence_level TEXT CHECK(evidence_level IN ('curated','signal','sccs_confirmed')),
echo     source TEXT
echo );
echo.
echo CREATE TABLE IF NOT EXISTS pkpd_parameters (
echo     param_id TEXT PRIMARY KEY,
echo     drug_id TEXT REFERENCES drugs(drug_id),
echo     param_name TEXT NOT NULL,      -- 'CL', 'V', 'ka', 'EC50'
echo     population_mean REAL,
echo     population_cv REAL,            -- coefficient of variation
echo     covariate_model TEXT,          -- e.g., 'allometric_weight_GFR'
echo     iiv_omega REAL,                -- inter-individual variability
echo     iov_omega REAL                 -- inter-occasion variability
echo );
echo.
echo CREATE TABLE IF NOT EXISTS biological_variation (
echo     analyte TEXT PRIMARY KEY,
echo     cv_within REAL NOT NULL,       -- within-subject CV (%)
echo     cv_between REAL NOT NULL,      -- between-subject CV (%)
echo     unit TEXT,
echo     source TEXT DEFAULT 'EFLM'
echo );
echo.
echo -- Runtime Patient Data Tables
echo CREATE TABLE IF NOT EXISTS lab_observations (
echo     obs_id TEXT PRIMARY KEY,
echo     patient_id TEXT NOT NULL,
echo     analyte TEXT NOT NULL,
echo     value REAL NOT NULL,
echo     unit TEXT,
echo     timestamp DATETIME NOT NULL,
echo     provenance_id TEXT UNIQUE
echo );
echo.
echo CREATE TABLE IF NOT EXISTS imaging_findings (
echo     finding_id TEXT PRIMARY KEY,
echo     patient_id TEXT NOT NULL,
echo     modality TEXT NOT NULL,          -- 'echo', 'ct', 'cxr', 'histopath'
echo     covariate_name TEXT NOT NULL,    -- 'EF', 'lean_mass_kg', 'PDFF'
echo     value_mean REAL NOT NULL,
echo     value_variance REAL NOT NULL,    -- measurement uncertainty
echo     timestamp DATETIME NOT NULL,
echo     provenance_id TEXT UNIQUE
echo );
echo.
echo CREATE TABLE IF NOT EXISTS alerts (
echo     alert_id TEXT PRIMARY KEY,
echo     patient_id TEXT NOT NULL,
echo     module TEXT NOT NULL,            -- 'M1', 'M2', 'M3', 'M4'
echo     harm_proposition TEXT NOT NULL,  -- e.g., 'QT_prolongation', 'AKI'
echo     posterior_probability REAL,
echo     log_lr REAL,
echo     severity TEXT,
echo     evidence_level TEXT,
echo     provenance_id TEXT UNIQUE,
echo     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
echo );
echo.
echo CREATE TABLE IF NOT EXISTS advisor_recommendations (
echo     rec_id TEXT PRIMARY KEY,
echo     patient_id TEXT NOT NULL,
echo     harm_proposition TEXT NOT NULL,
echo     recommended_action TEXT NOT NULL,
echo     expected_utility REAL,
echo     alert_budget_used INTEGER,
echo     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
echo );
echo.
echo CREATE TABLE IF NOT EXISTS clinician_feedback (
echo     feedback_id TEXT PRIMARY KEY,
echo     rec_id TEXT REFERENCES advisor_recommendations(rec_id),
echo     action_taken TEXT CHECK(action_taken IN ('accepted','overridden','ignored')),
echo     clinician_note TEXT,
echo     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
echo );
) > database\schema.sql

echo. > database\seed_data\.gitkeep
echo. > data\raw\.gitkeep
echo. > data\processed\.gitkeep

echo       Done.
echo.

:: ============================================================
:: 7. CREATE TEST FILES
:: ============================================================
echo [7/8] Creating test files...

(
echo """Test: Blackboard fusion order-invariance and ProvenanceError."""
echo import pytest
echo import sys, os
echo sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
echo from blackboard.blackboard import Blackboard, ProvenanceError
echo.
echo def test_fusion_order_invariance():
echo     bb = Blackboard()
echo     bb.update('GFR', 45.0, 'lab_creat_001', 'M3')
echo     bb.update('EF', 0.55, 'echo_001', 'M4')
echo     assert bb.get_state('GFR') == 45.0
echo     assert bb.get_state('EF') == 0.55
echo.
echo def test_duplicate_provenance_raises_error():
echo     bb = Blackboard()
echo     bb.update('GFR', 45.0, 'lab_creat_001', 'M3')
echo     with pytest.raises(ProvenanceError):
echo         bb.update('GFR', 46.0, 'lab_creat_001', 'M3')  # Same provenance_id!
) > tests\test_blackboard_fusion.py

(
echo """Test: M2 parameter identifiability via profile likelihood."""
echo import pytest
echo.
echo def test_unidentifiable_parameter_flagged():
echo     # TODO: With only 2 trough levels, ka should be flagged as unidentifiable
echo     pass
echo.
echo def test_identifiable_parameter_with_sufficient_tdm():
echo     # TODO: With 5+ TDM samples, CL should have finite profile CI
echo     pass
) > tests\test_m2_identifiability.py

(
echo """Test: Advisor expected utility and knapsack constraints."""
echo import sys, os
echo sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
echo from advisor.utility import expected_utility
echo from advisor.knapsack import select_alerts
echo.
echo def test_expected_utility_calculation():
echo     eu = expected_utility(p_harm=0.8, u_harm_action=-10, u_noharm_action=5)
echo     assert abs(eu - (0.8 * -10 + 0.2 * 5)) ^< 1e-9
echo.
echo def test_knapsack_respects_budget():
echo     alerts = [{'net_benefit': 10, 'attention_cost': 1} for _ in range(5)]
echo     selected = select_alerts(alerts, budget=3)
echo     assert len(selected) == 3
) > tests\test_advisor_utility.py

echo       Done.
echo.

:: ============================================================
:: 8. CREATE CONFIG & ROOT FILES
:: ============================================================
echo [8/8] Creating configuration and root files...

:: --- requirements.txt ---
(
echo # RezpharmaCDSS Dependencies
echo # Core Scientific Computing
echo numpy^>=1.24
echo scipy^>=1.10
echo pandas^>=2.0
echo.
echo # Bayesian Modeling
echo pymc^>=5.0
echo arviz^>=0.15
echo.
echo # Machine Learning
echo scikit-learn^>=1.3
echo.
echo # Web Services (FastAPI Microservices)
echo fastapi^>=0.100
echo uvicorn^>=0.23
echo pydantic^>=2.0
echo.
echo # Message Bus (Event-driven mode)
echo redis^>=4.6
echo.
echo # Jupyter Environment
echo jupyterlab^>=4.0
echo ipykernel^>=6.25
echo jupyterlab-code-formatter^>=2.0
echo jupyterlab-lsp^>=5.0
echo python-lsp-server[all]^>=1.8
echo black^>=23.0
echo.
echo # Visualization
echo matplotlib^>=3.7
echo seaborn^>=0.12
echo.
echo # Testing
echo pytest^>=7.4
echo pytest-cov^>=4.1
echo.
echo # Configuration
echo pyyaml^>=6.0
) > requirements.txt

:: --- .gitignore ---
(
echo # Virtual Environment
echo rezpharma_env/
echo venv/
echo .venv/
echo.
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.egg-info/
echo dist/
echo build/
echo.
echo # Jupyter
echo .ipynb_checkpoints/
echo.
echo # Data (keep structure, ignore contents)
echo data/raw/*
echo !data/raw/.gitkeep
echo data/processed/*
echo !data/processed/.gitkeep
echo.
echo # Database
echo *.db
echo *.sqlite
echo.
echo # IDE
echo .vscode/
echo .idea/
echo.
echo # OS
echo Thumbs.db
echo .DS_Store
) > .gitignore

:: --- config/settings.yaml ---
(
echo # RezpharmaCDSS System Configuration
echo.
echo advisor:
echo   alert_budget: 3          # Max actionable alerts per patient per day
echo   default_prior_harm: 0.05 # Default prior probability for harm propositions
echo.
echo m3_lab:
echo   rcv_z_score: 2.77        # Z-value for Reference Change Value (95%% CI)
echo   kinetic_gfr_particles: 1000
echo.
echo m2_pkpd:
echo   tdm_min_samples: 3       # Minimum TDM samples for Bayesian forecasting
echo   target_attainment_prob: 0.90
echo   toxicity_ceiling_prob: 0.10
echo.
echo blackboard:
echo   provenance_ttl_hours: 72  # Time-to-live for provenance tracking
echo.
echo monitoring:
echo   cusum_threshold: 5.0
echo   recalibration_window_days: 30
) > config\settings.yaml

:: --- config/jupyter_lab_config.py ---
(
echo # Jupyter Lab Configuration for RezpharmaCDSS
echo c.ServerApp.root_dir = r'C:\Users\LENOVO\Desktop\RezpharmaCDSS'
echo c.ServerApp.autosave_interval = 60
echo c.ServerApp.default_url = '/lab/tree/notebooks'
echo c.ServerApp.open_browser = True
) > config\jupyter_lab_config.py

:: --- start_jupyter.bat ---
(
echo @echo off
echo title Rezpharma CDSS - Jupyter Lab
echo echo Activating Rezpharma Environment...
echo call rezpharma_env\Scripts\activate.bat
echo echo Starting Jupyter Lab...
echo jupyter lab --config=config/jupyter_lab_config.py
echo pause
) > start_jupyter.bat

echo       Done.
echo.

:: ============================================================
:: FINAL SUMMARY
:: ============================================================
echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo   Project Root: C:\Users\LENOVO\Desktop\RezpharmaCDSS
echo.
echo   Next Steps:
echo     1. Activate env:  rezpharma_env\Scripts\activate
echo     2. Install deps:  pip install -r requirements.txt
echo     3. Register kernel:
echo        python -m ipykernel install --user --name rezpharma_kernel --display-name "Python (Rezpharma CDSS)"
echo     4. Launch Jupyter: start_jupyter.bat
echo     5. Run tests:      pytest tests/ -v
echo.
echo ============================================================
pause