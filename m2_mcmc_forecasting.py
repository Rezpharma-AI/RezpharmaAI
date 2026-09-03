"""
Module M2: Full Bayesian MCMC Forecasting & Chance-Constrained Dosing
Implements Metropolis-Hastings MCMC for posterior generation over PK parameters.
No external dependencies beyond numpy/scipy.
"""
import sys, os
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.m2_pkpd.nlme_model import NLMEModel, PKParameters

# ═══════════════════════════════════════════════════════════
# METROPOLIS-HASTINGS MCMC SAMPLER
# ═══════════════════════════════════════════════════════════
class MCMCSampler:
    """
    Metropolis-Hastings sampler for Bayesian PK parameter estimation.
    Generates full posterior distributions over CL and V.
    """
    
    def __init__(self, model: NLMEModel, tdm_data: List[Tuple[float, float]], 
                 prior: PKParameters, obs_noise: float = 0.15,
                 proposal_std: float = 0.1, n_iterations: int = 10000,
                 burn_in: int = 2000, thinning: int = 5):
        self.model = model
        self.tdm_data = tdm_data
        self.prior = prior
        self.obs_noise = obs_noise
        self.proposal_std = proposal_std
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thinning = thinning
        
    def log_likelihood(self, cl: float, v: float) -> float:
        """Compute log-likelihood of TDM data given parameters."""
        if cl <= 0 or v <= 0:
            return -np.inf
        
        ll = 0.0
        for t, c_obs in self.tdm_data:
            pk = PKParameters(cl=cl, v=v, ka=self.prior.ka, f_bio=self.prior.f_bio)
            c_pred = self.model.predict_concentration(pk, dose=1000, time=t)
            sigma = self.obs_noise * max(c_pred, 1.0) + 1e-6
            ll += norm.logpdf(c_obs, loc=c_pred, scale=sigma)
        return ll
    
    def log_prior(self, cl: float, v: float) -> float:
        """Compute log-prior from population NLME parameters."""
        if cl <= 0 or v <= 0:
            return -np.inf
        
        # Get omega values from model's population parameters
        omega_cl = self.model.pop_params.get("omega_cl", 0.3)
        omega_v = self.model.pop_params.get("omega_v", 0.2)
        
        # Log-normal priors on CL and V
        lp = norm.logpdf(np.log(cl), loc=np.log(self.prior.cl), scale=omega_cl)
        lp += norm.logpdf(np.log(v), loc=np.log(self.prior.v), scale=omega_v)
        return lp
    
    def log_posterior(self, cl: float, v: float) -> float:
        """Unnormalized log-posterior = log-likelihood + log-prior."""
        return self.log_likelihood(cl, v) + self.log_prior(cl, v)
    
    def sample(self) -> Dict:
        """Run MCMC and return posterior samples."""
        # Initialize at prior
        current_cl = self.prior.cl
        current_v = self.prior.v
        current_log_post = self.log_posterior(current_cl, current_v)
        
        samples_cl = []
        samples_v = []
        accepted = 0
        
        for i in range(self.n_iterations):
            # Propose new values (random walk in log-space for positivity)
            prop_cl = current_cl * np.exp(np.random.normal(0, self.proposal_std))
            prop_v = current_v * np.exp(np.random.normal(0, self.proposal_std))
            
            prop_log_post = self.log_posterior(prop_cl, prop_v)
            
            # Metropolis acceptance ratio
            log_alpha = prop_log_post - current_log_post
            
            if np.log(np.random.uniform()) < log_alpha:
                current_cl = prop_cl
                current_v = prop_v
                current_log_post = prop_log_post
                accepted += 1
            
            # Store samples after burn-in with thinning
            if i >= self.burn_in and (i - self.burn_in) % self.thinning == 0:
                samples_cl.append(current_cl)
                samples_v.append(current_v)
        
        acceptance_rate = accepted / self.n_iterations
        
        return {
            "cl_samples": np.array(samples_cl),
            "v_samples": np.array(samples_v),
            "acceptance_rate": acceptance_rate,
            "n_effective_samples": len(samples_cl)
        }

# ═══════════════════════════════════════════════════════════
# BAYESIAN DOSE OPTIMIZER (Chance-Constrained)
# ═══════════════════════════════════════════════════════════
class BayesianDoseOptimizer:
    """
    Chance-constrained dosing using full posterior uncertainty.
    P(AUC > target) > target_prob AND P(Cmax > toxic) < tox_prob
    """
    
    def __init__(self, model: NLMEModel, posterior_cl: np.ndarray, 
                 posterior_v: np.ndarray):
        self.model = model
        self.posterior_cl = posterior_cl
        self.posterior_v = posterior_v
        
    def simulate_auc_cmax(self, dose: float, tau: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate AUC and Cmax for each posterior sample."""
        n_samples = len(self.posterior_cl)
        aucs = np.zeros(n_samples)
        cmaxs = np.zeros(n_samples)
        
        for i in range(n_samples):
            cl = self.posterior_cl[i]
            v = self.posterior_v[i]
            
            # AUC = Dose / CL (for IV or complete absorption)
            aucs[i] = dose / cl
            
            # Cmax approximation (peak after infusion)
            ke = cl / v
            cmaxs[i] = (dose / v) * (1 - np.exp(-ke * 1.0)) / (1 - np.exp(-ke * tau))
        
        return aucs, cmaxs
    
    def optimize(self, target_auc: float = 400.0, tox_cmax: float = 25.0,
                 target_prob: float = 0.90, tox_prob: float = 0.10,
                 dose_grid: np.ndarray = None) -> Dict:
        """Find optimal dose satisfying chance constraints."""
        if dose_grid is None:
            dose_grid = np.arange(250, 3001, 250)
        
        best_dose = dose_grid[0]
        best_attainment = 0.0
        
        for dose in dose_grid:
            aucs, cmaxs = self.simulate_auc_cmax(dose)
            
            p_attainment = np.mean(aucs > target_auc)
            p_toxicity = np.mean(cmaxs > tox_cmax)
            
            if p_attainment >= target_prob and p_toxicity <= tox_prob:
                return {
                    "recommended_dose": dose,
                    "p_attainment": p_attainment,
                    "p_toxicity": p_toxicity,
                    "status": "OPTIMAL"
                }
            elif p_attainment > best_attainment:
                best_dose = dose
                best_attainment = p_attainment
        
        return {
            "recommended_dose": best_dose,
            "p_attainment": best_attainment,
            "p_toxicity": p_toxicity,
            "status": "SUBOPTIMAL (constraints not fully met)"
        }

# ═══════════════════════════════════════════════════════════
# CLINICAL DEMONSTRATION
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  M2 Full Bayesian MCMC Forecasting & Chance-Constrained Dosing")
    print("=" * 70)
    
    # Setup
    model = NLMEModel("vancomycin", pop_params={
        "cl_pop": 4.0, "v_pop": 40.0, "ka_pop": 1.5, "f_bio": 1.0,
        "theta_gfr": 0.75, "omega_cl": 0.3, "omega_v": 0.2
    })
    
    prior = model.predict_parameters(weight=80, gfr=45, age=70)
    print(f"\n[1] Population Prior: CL = {prior.cl:.2f} L/h, V = {prior.v:.2f} L")
    
    # Rich TDM data (Peak + 2 Troughs)
    tdm_data = [(1.0, 28.0), (12.0, 15.0), (24.0, 12.0)]
    print(f"[2] TDM Data: {tdm_data}")
    
    # Run MCMC
    print(f"\n[3] Running Metropolis-Hastings MCMC (10,000 iterations)...")
    sampler = MCMCSampler(model, tdm_data, prior, 
                          n_iterations=10000, burn_in=2000, thinning=5)
    results = sampler.sample()
    
    cl_samples = results["cl_samples"]
    v_samples = results["v_samples"]
    
    print(f"    Acceptance rate: {results['acceptance_rate']:.1%}")
    print(f"    Effective samples: {results['n_effective_samples']}")
    print(f"\n[4] Posterior Distributions:")
    print(f"    CL: mean={np.mean(cl_samples):.2f}, "
          f"95% CI=[{np.percentile(cl_samples, 2.5):.2f}, {np.percentile(cl_samples, 97.5):.2f}]")
    print(f"    V:  mean={np.mean(v_samples):.2f}, "
          f"95% CI=[{np.percentile(v_samples, 2.5):.2f}, {np.percentile(v_samples, 97.5):.2f}]")
    
    # Chance-Constrained Dosing
    print(f"\n[5] Chance-Constrained Dose Optimization:")
    print(f"    Target: P(AUC > 400) > 0.90")
    print(f"    Constraint: P(Cmax > 25) < 0.10")
    
    optimizer = BayesianDoseOptimizer(model, cl_samples, v_samples)
    dose_result = optimizer.optimize(target_auc=400, tox_cmax=25)
    
    print(f"\n    Status: {dose_result['status']}")
    print(f"    Recommended Dose: {dose_result['recommended_dose']} mg")
    print(f"    P(AUC > 400): {dose_result['p_attainment']:.1%}")
    print(f"    P(Cmax > 25): {dose_result['p_toxicity']:.1%}")
    
    # Comparison with MAP
    print(f"\n[6] Comparison: MCMC vs MAP (Point Estimate):")
    map_cl = np.mean(cl_samples)
    map_auc = 1000 / map_cl  # Dose=1000
    print(f"    MAP approach: AUC = {map_auc:.0f} (single point)")
    print(f"    MCMC approach: AUC distribution with {results['n_effective_samples']} samples")
    print(f"    → MCMC captures full uncertainty, preventing overconfident dosing")
    
    print("\n" + "=" * 70)
    print("  This implements the blueprint's mandate:")
    print("  'Instead of targeting a single point estimate for AUC, the system")
    print("   recommends the lowest dose that achieves the target attainment")
    print("   probability while staying below the toxicity ceiling.'")
    print("=" * 70)