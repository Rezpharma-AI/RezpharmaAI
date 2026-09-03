import sys, os
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.m2_pkpd.nlme_model import NLMEModel, PKParameters

class ProfileLikelihoodAnalyzer:
    def __init__(self, model: NLMEModel, tdm_data: list, prior_params: PKParameters, obs_noise=0.15):
        self.model = model
        self.tdm_data = tdm_data
        self.prior = prior_params
        self.obs_noise = obs_noise
        
    def _neg_log_posterior(self, cl, v):
        if cl <= 0 or v <= 0:
            return 1e10
        nll = 0.0
        for t, c_obs in self.tdm_data:
            pk = PKParameters(cl=cl, v=v, ka=self.prior.ka, f_bio=self.prior.f_bio)
            c_pred = self.model.predict_concentration(pk, dose=1000, time=t)
            sigma = self.obs_noise * max(c_pred, 1.0) + 1e-6
            nll += 0.5 * ((c_obs - c_pred) / sigma) ** 2 + np.log(sigma)
        nll += 0.5 * (np.log(cl / self.prior.cl) / self.prior.omega_cl) ** 2
        nll += 0.5 * (np.log(v / self.prior.v) / self.prior.omega_v) ** 2
        return nll

    def find_map(self):
        def objective(x):
            return self._neg_log_posterior(x[0], x[1])
        res = minimize(objective, [self.prior.cl, self.prior.v], method='Nelder-Mead', options={'maxiter': 2000})
        return res.x[0], res.x[1], res.fun

    def compute_profile_likelihood_ci(self, param_idx, map_cl, map_v, alpha=0.05):
        map_val = map_cl if param_idx == 0 else map_v
        other_val = map_v if param_idx == 0 else map_cl
        map_nll = self._neg_log_posterior(map_cl, map_v)
        threshold = map_nll + 0.5 * chi2.ppf(1 - alpha, df=1)
        
        def profile_nll(target_val):
            def objective(x):
                if param_idx == 0:
                    return self._neg_log_posterior(target_val, x[0])
                else:
                    return self._neg_log_posterior(x[0], target_val)
            res = minimize(objective, [other_val], method='Nelder-Mead', options={'maxiter': 500})
            return res.fun

        try:
            lower = brentq(lambda x: profile_nll(x) - threshold, map_val * 0.1, map_val)
        except ValueError:
            lower = 0.0
            
        try:
            upper = brentq(lambda x: profile_nll(x) - threshold, map_val, map_val * 10.0)
        except ValueError:
            upper = np.inf
            
        return lower, upper

    def check_identifiability(self):
        map_cl, map_v, map_nll = self.find_map()
        cl_lower, cl_upper = self.compute_profile_likelihood_ci(0, map_cl, map_v)
        v_lower, v_upper = self.compute_profile_likelihood_ci(1, map_cl, map_v)
        
        cl_identifiable = (cl_lower > 0) and (cl_upper < np.inf)
        v_identifiable = (v_lower > 0) and (v_upper < np.inf)
        
        return {
            "map_cl": map_cl, "map_v": map_v,
            "cl_95_ci": (cl_lower, cl_upper), "v_95_ci": (v_lower, v_upper),
            "cl_identifiable": cl_identifiable, "v_identifiable": v_identifiable,
            "safe_to_recommend": cl_identifiable and v_identifiable
        }

if __name__ == "__main__":
    print("=" * 70)
    print("  M2 Advanced Inference: Profile Likelihood Identifiability Check")
    print("=" * 70)
    
    model = NLMEModel("vancomycin", pop_params={
        "cl_pop": 4.0, "v_pop": 40.0, "ka_pop": 1.5, "f_bio": 1.0,
        "theta_gfr": 0.75, "omega_cl": 0.3, "omega_v": 0.2
    })
    prior = model.predict_parameters(weight=80, gfr=45, age=70)
    print(f"\n[1] Population Prior (NLME): CL = {prior.cl:.2f} L/h, V = {prior.v:.2f} L")
    
    print("\n[2] Scenario A: Sparse TDM Data (2 samples)")
    tdm_sparse = [(12.0, 15.0), (24.0, 12.0)]
    analyzer_a = ProfileLikelihoodAnalyzer(model, tdm_sparse, prior)
    result_a = analyzer_a.check_identifiability()
    print(f"  MAP Estimates: CL = {result_a['map_cl']:.2f} L/h, V = {result_a['map_v']:.2f} L")
    print(f"  CL 95% CI: [{result_a['cl_95_ci'][0]:.2f}, {result_a['cl_95_ci'][1]:.2f}]")
    print(f"  Identifiable? CL: {result_a['cl_identifiable']}, V: {result_a['v_identifiable']}")
    if not result_a['safe_to_recommend']:
        print("  🚨 SAFETY INTERVENTION: Parameters unidentifiable. System will NOT recommend a dose. More TDM samples required.")
    
    print("\n[3] Scenario B: Rich TDM Data (Peak + 2 Troughs)")
    tdm_rich = [(1.0, 28.0), (12.0, 15.0), (24.0, 12.0)]
    analyzer_b = ProfileLikelihoodAnalyzer(model, tdm_rich, prior)
    result_b = analyzer_b.check_identifiability()
    print(f"  MAP Estimates: CL = {result_b['map_cl']:.2f} L/h, V = {result_b['map_v']:.2f} L")
    print(f"  CL 95% CI: [{result_b['cl_95_ci'][0]:.2f}, {result_b['cl_95_ci'][1]:.2f}]")
    print(f"  Identifiable? CL: {result_b['cl_identifiable']}, V: {result_b['v_identifiable']}")
    if result_b['safe_to_recommend']:
        print("  ✅ SAFE TO RECOMMEND: Parameters are well-identified. Proceeding to chance-constrained dose optimization.")
    print("\n" + "=" * 70)