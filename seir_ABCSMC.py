import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.integrate import odeint

# Step 1: Model parameters and initial conditions
# Don't know anything about the parameters

# Step 2: read from file and store into t and observed_vector
data = np.loadtxt("epidemic_data.csv", delimiter=",", skiprows=1)
t = data[:, 0]
X0 = [data[0,1], 0, data[0,2], data[0,3]]
observed_matrix = data[:, 1:4]  # Columns: S, I, R (no E)
observed = observed_matrix.reshape(-1)  # A vector 1D for PyMC

# Step 3: Define the ODE system (SEIR model)
def dX_dt(X, t, beta, sigma, gamma):
    S, E, I, R = X
    N = S + E + I + R
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])

# Step 4: Define the SEIR model for PyMC. Return a 1D array (S, I, R only)
def seir_model(rng, beta, sigma, gamma, size=None):
    beta_scalar = beta.item() if hasattr(beta, "item") else float(beta)
    sigma_scalar = sigma.item() if hasattr(sigma, "item") else float(sigma)
    gamma_scalar = gamma.item() if hasattr(gamma, "item") else float(gamma)
    result = odeint(dX_dt, y0=X0, t=t, rtol=0.05, atol=0.05, args=(beta_scalar, sigma_scalar, gamma_scalar))
    # Return only S, I, R (columns 0, 2, 3) and reshape to 1D
    return result[:, [0, 2, 3]].reshape(-1)


# Step 5: Bayesian inference with PyMC
with pm.Model() as model_seir:
    # Priors
    beta = pm.HalfNormal("beta", 2.0)
    sigma = pm.HalfNormal("sigma", 1.0)
    gamma = pm.HalfNormal("gamma", 0.5)
    # Likelihood (ABC). Epsilon is the initial tolerance (if the posterior is too narrow, increase it, and the other way around).
    sim = pm.Simulator("sim", seir_model, params=(beta, sigma, gamma), epsilon=0.5, observed=observed)
    # Inference
    samples = pm.sample_smc(draws=500, chains=4, threshold=0.3, correlation_threshold=0.1)
    # Convert to ArviZ InferenceData
    posterior = samples.posterior.stack(samples=("draw", "chain"))
    # post = posterior.to_pandas()

az.summary(samples, hdi_prob=0.95)

# Plotting
## Plot posterior predictive
_, ax = plt.subplots(figsize=(14, 6))
ax.plot(t, observed_matrix[:, 0], "o", label="susceptible (observed)", c="C0", mec="k")
ax.plot(t, observed_matrix[:, 1], "o", label="infected (observed)", c="C1", mec="k")
ax.plot(t, observed_matrix[:, 2], "o", label="recovered (observed)", c="C2", mec="k")

mean_beta = posterior["beta"].mean().item()
mean_sigma = posterior["sigma"].mean().item()
mean_gamma = posterior["gamma"].mean().item()
mean_sim = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(mean_beta, mean_sigma, mean_gamma))
ax.plot(t, mean_sim[:, 0], linewidth=3, label="mean susceptible", c="C0")
ax.plot(t, mean_sim[:, 1], linewidth=3, label="mean exposed (unobserved)", c="C3")
ax.plot(t, mean_sim[:, 2], linewidth=3, label="mean infected", c="C1")
ax.plot(t, mean_sim[:, 3], linewidth=3, label="mean recovered", c="C2")

for i in np.random.default_rng(42).integers(0, posterior.sizes["samples"], 75):
    beta_i = posterior["beta"].values[i]
    sigma_i = posterior["sigma"].values[i]
    gamma_i = posterior["gamma"].values[i]
    sim_i = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(beta_i, sigma_i, gamma_i))
    ax.plot(t, sim_i[:, 0], alpha=0.1, c="C0")
    ax.plot(t, sim_i[:, 1], alpha=0.1, c="C3")
    ax.plot(t, sim_i[:, 2], alpha=0.1, c="C1")
    ax.plot(t, sim_i[:, 3], alpha=0.1, c="C2")

ax.set_xlabel("time")
ax.set_ylabel("population")
ax.legend()
plt.tight_layout()
plt.savefig("results/seir_predictive.pdf")
plt.close()

## Plot posterior distributions
# az.plot_posterior(samples)
az.plot_posterior(samples, kind="hist", bins=30)
plt.tight_layout()
plt.savefig("results/seir_marginals.pdf")
plt.close()

## Plot diagnostics
# az.plot_trace(samples, kind="rank_vlines")
az.plot_trace(samples, kind="rank_bars")
plt.tight_layout()
plt.savefig("results/seir_trace.pdf")
plt.close()

