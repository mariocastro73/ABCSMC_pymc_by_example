import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.integrate import odeint

# Step 1: Read data with group information
data = np.loadtxt("epidemic_data_groups.csv", delimiter=",", skiprows=1)

groups = data[:, 0].astype(int)  # First column: group IDs
unique_groups = np.unique(groups)
n_groups = len(unique_groups)

print(f"\nFound {n_groups} groups: {unique_groups}")

# Step 2: Organize data by group
# Expected columns: Group, Time, S, I, R
# But we need to check if we have S, I, R or just I, R
group_data = {}
for group_id in unique_groups:
    mask = groups == group_id
    group_indices = np.where(mask)[0]
    
    # Extract time and observed data for this group
    t_group = data[mask, 1]  # Time is second column
    
    # Check number of columns to determine data structure
    if data.shape[1] == 4:  # Group, Time, I, R (no S column)
        print(f"Detected format: Group, Time, I, R")
        observed_group = data[mask, 2:4]  # I, R columns only
        # We need to infer S from total population
        # Assume initial total population N = S0 + I0 + R0
        I0 = observed_group[0, 0]
        R0 = observed_group[0, 1]
        # Assume a reasonable population size (e.g., 1000 or use max of I+R)
        N = 1000  # You may need to adjust this
        S0 = N - I0 - R0
        X0_group = [S0, 0, I0, R0]  # S, E, I, R
    elif data.shape[1] == 5:  # Group, Time, S, I, R
        print(f"Detected format: Group, Time, S, I, R")
        observed_group = data[mask, 2:5]  # S, I, R columns
        X0_group = [observed_group[0, 0], 0, observed_group[0, 1], observed_group[0, 2]]
    else:
        raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
    
    group_data[group_id] = {
        't': t_group,
        'X0': X0_group,
        'observed': observed_group,
        'observed_flat': observed_group.reshape(-1)
    }
    print(f"Group {group_id}: {len(t_group)} time points, X0={X0_group}")

# Step 3: Define the ODE system (SEIR model)
def dX_dt(X, t, beta, sigma, gamma):
    S, E, I, R = X
    N = S + E + I + R
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])

# Step 4: Define SEIR model factory for each group
# Determine which columns to return based on data format
has_S_column = data.shape[1] == 5

def create_seir_simulator(group_id):
    t_group = group_data[group_id]['t']
    X0_group = group_data[group_id]['X0']
    
    def seir_model(rng, beta, sigma, gamma, size=None):
        beta_scalar = beta.item() if hasattr(beta, "item") else float(beta)
        sigma_scalar = sigma.item() if hasattr(sigma, "item") else float(sigma)
        gamma_scalar = gamma.item() if hasattr(gamma, "item") else float(gamma)
        
        result = odeint(dX_dt, y0=X0_group, t=t_group, rtol=0.01, 
                       args=(beta_scalar, sigma_scalar, gamma_scalar))
        
        # Return columns matching observed data
        if has_S_column:
            # Return S, I, R (columns 0, 2, 3)
            return result[:, [0, 2, 3]].reshape(-1)
        else:
            # Return only I, R (columns 2, 3)
            return result[:, [2, 3]].reshape(-1)
    
    return seir_model

# Step 5: Hierarchical Bayesian inference with PyMC
with pm.Model() as model_hierarchical:
    # Hyperpriors (population-level distributions)
    # These represent the distribution of parameters across all groups
    mu_beta = pm.HalfNormal("mu_beta", 2.0)
    sigma_beta = pm.HalfNormal("sigma_beta", 1.0)
    
    mu_sigma = pm.HalfNormal("mu_sigma", 1.0)
    sigma_sigma = pm.HalfNormal("sigma_sigma", 0.5)
    
    mu_gamma = pm.HalfNormal("mu_gamma", 0.5)
    sigma_gamma = pm.HalfNormal("sigma_gamma", 0.25)
    
    # Group-specific parameters drawn from hyperpriors
    beta_group = pm.TruncatedNormal("beta", mu=mu_beta, sigma=sigma_beta, 
                                     lower=0, shape=n_groups)
    sigma_group = pm.TruncatedNormal("sigma", mu=mu_sigma, sigma=sigma_sigma, 
                                      lower=0, shape=n_groups)
    gamma_group = pm.TruncatedNormal("gamma", mu=mu_gamma, sigma=sigma_gamma, 
                                      lower=0, shape=n_groups)
    
    # Likelihood for each group
    for idx, group_id in enumerate(unique_groups):
        seir_model = create_seir_simulator(group_id)
        observed_flat = group_data[group_id]['observed_flat']
        
        # ABC likelihood with epsilon tolerance
        sim = pm.Simulator(f"sim_group_{group_id}", 
                          seir_model, 
                          params=(beta_group[idx], sigma_group[idx], gamma_group[idx]),
                          epsilon=0.5, 
                          observed=observed_flat)
    
    # Inference
    print("Starting SMC sampling...")
    samples = pm.sample_smc(draws=1000, chains=4)  # Faster for testing
    
    # Convert to ArviZ InferenceData
    posterior = samples.posterior.stack(samples=("draw", "chain"))

# Print summary statistics
az.summary(samples, var_names=["mu_beta", "sigma_beta", "mu_sigma", 
                                "sigma_sigma", "mu_gamma", "sigma_gamma"], 
           hdi_prob=0.95)

az.summary(samples, var_names=["beta", "sigma", "gamma"], hdi_prob=0.95)

# Step 6: Plotting
# Plot posterior predictive for each group
fig, axes = plt.subplots(n_groups, 1, figsize=(14, 6 * n_groups), squeeze=False)

for idx, group_id in enumerate(unique_groups):
    ax = axes[idx, 0]
    
    t_group = group_data[group_id]['t']
    X0_group = group_data[group_id]['X0']
    observed_group = group_data[group_id]['observed']
    
    # Plot observed data based on format
    if has_S_column:
        ax.plot(t_group, observed_group[:, 0], "o", label="susceptible (observed)", 
                c="C0", mec="k")
        ax.plot(t_group, observed_group[:, 1], "o", label="infected (observed)", 
                c="C1", mec="k")
        ax.plot(t_group, observed_group[:, 2], "o", label="recovered (observed)", 
                c="C2", mec="k")
    else:
        # Only I and R are observed
        ax.plot(t_group, observed_group[:, 0], "o", label="infected (observed)", 
                c="C1", mec="k")
        ax.plot(t_group, observed_group[:, 1], "o", label="recovered (observed)", 
                c="C2", mec="k")
    
    # Mean trajectory
    mean_beta = posterior["beta"].values[:, idx].mean()
    mean_sigma = posterior["sigma"].values[:, idx].mean()
    mean_gamma = posterior["gamma"].values[:, idx].mean()
    mean_sim = odeint(dX_dt, y0=X0_group, t=t_group, rtol=0.01, 
                     args=(mean_beta, mean_sigma, mean_gamma))
    
    if has_S_column:
        ax.plot(t_group, mean_sim[:, 0], linewidth=3, label="mean susceptible", c="C0")
    else:
        ax.plot(t_group, mean_sim[:, 0], linewidth=3, label="mean susceptible (inferred)", 
                c="C0", linestyle="--")
    ax.plot(t_group, mean_sim[:, 1], linewidth=3, label="mean exposed (unobserved)", c="C3")
    ax.plot(t_group, mean_sim[:, 2], linewidth=3, label="mean infected", c="C1")
    ax.plot(t_group, mean_sim[:, 3], linewidth=3, label="mean recovered", c="C2")
    
    # Posterior samples
    for i in np.random.randint(0, posterior.samples.size/2, 15):
        # beta_i = posterior["beta"].values[i, idx]
        # sigma_i = posterior["sigma"].values[i, idx]
        # gamma_i = posterior["gamma"].values[i, idx]
        beta_i = posterior["beta"].values[idx, i]
        sigma_i = posterior["sigma"].values[idx, i]
        gamma_i = posterior["gamma"].values[idx, i]
        sim_i = odeint(dX_dt, y0=X0_group, t=t_group, rtol=0.01, 
                      args=(beta_i, sigma_i, gamma_i))
        ax.plot(t_group, sim_i[:, 0], alpha=0.1, c="C0")
        ax.plot(t_group, sim_i[:, 1], alpha=0.1, c="C3")
        ax.plot(t_group, sim_i[:, 2], alpha=0.1, c="C1")
        ax.plot(t_group, sim_i[:, 3], alpha=0.1, c="C2")
    
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title(f"Group {group_id}")
    ax.legend()

plt.tight_layout()
plt.show()

# Plot hyperparameter posteriors
az.plot_posterior(samples, var_names=["mu_beta", "sigma_beta", "mu_sigma", 
                                       "sigma_sigma", "mu_gamma", "sigma_gamma"],
                  kind="hist", bins=30)
plt.suptitle("Hyperparameter Posteriors", y=1.02)
plt.tight_layout()
plt.show()

# Plot group-specific parameter posteriors
az.plot_posterior(samples, var_names=["beta", "sigma", "gamma"], 
                  kind="hist", bins=30)
plt.suptitle("Group-Specific Parameter Posteriors", y=1.02)
plt.tight_layout()
plt.show()

# Plot trace diagnostics
az.plot_trace(samples, var_names=["mu_beta", "mu_sigma", "mu_gamma",
                                   "beta", "sigma", "gamma"], 
              kind="rank_bars", compact=True)
plt.tight_layout()
plt.show()

# Forest plot comparing parameters across groups
az.plot_forest(samples, var_names=["beta", "sigma", "gamma"], 
               combined=False, hdi_prob=0.95)
plt.tight_layout()
plt.show()
