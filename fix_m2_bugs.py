import re

# Fix 1: m2_advanced_inference.py - typo in __init__
with open("m2_advanced_inference.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("self.tdm_data = tdm\n", "self.tdm_data = tdm_data\n")
with open("m2_advanced_inference.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed m2_advanced_inference.py (tdm -> tdm_data)")

# Fix 2: m2_mcmc_forecasting.py - omega_cl/omega_v not in PKParameters
with open("m2_mcmc_forecasting.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the log_prior method to use model.pop_params instead of prior.omega_cl
old_prior = """    def log_prior(self, cl: float, v: float) -> float:
        \"\"\"Compute log-prior from population NLME parameters.\"\"\"
        if cl <= 0 or v <= 0:
            return -np.inf
        
        # Log-normal priors on CL and V
        lp = norm.logpdf(np.log(cl), loc=np.log(self.prior.cl), 
                         scale=self.prior.omega_cl)
        lp += norm.logpdf(np.log(v), loc=np.log(self.prior.v), 
                          scale=self.prior.omega_v)
        return lp"""

new_prior = """    def log_prior(self, cl: float, v: float) -> float:
        \"\"\"Compute log-prior from population NLME parameters.\"\"\"
        if cl <= 0 or v <= 0:
            return -np.inf
        
        # Get omega values from model's population parameters
        omega_cl = self.model.pop_params.get("omega_cl", 0.3)
        omega_v = self.model.pop_params.get("omega_v", 0.2)
        
        # Log-normal priors on CL and V
        lp = norm.logpdf(np.log(cl), loc=np.log(self.prior.cl), scale=omega_cl)
        lp += norm.logpdf(np.log(v), loc=np.log(self.prior.v), scale=omega_v)
        return lp"""

content = content.replace(old_prior, new_prior)

with open("m2_mcmc_forecasting.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed m2_mcmc_forecasting.py (omega_cl/omega_v from model.pop_params)")

print("\nDone! Now run:")
print("  python m2_advanced_inference.py")
print("  python m2_mcmc_forecasting.py")