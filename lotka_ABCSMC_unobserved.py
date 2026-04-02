import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.integrate import odeint

# Step 1: Model parameters and initial conditions
c = 1.5
d = 0.75

# Step 2: read from file and store into t and observed_vector
data = np.loadtxt("observed_data.csv", delimiter=",", skiprows=1)
t = data[:, 0]
X0 = data[0, 1:]  # Initial condition from the first row of data
observed_vector = data[:, 1]  # Only the first species is observed
observed = observed_vector  # A vector 1D for PyMC

# Step 3: Define the ODE system
def dX_dt(X, t, a, b, c, d):
    return np.array([a * X[0] - b * X[0] * X[1],
                     -c * X[1] + d * b * X[0] * X[1]])

# Step 4: Define the competition model for PyMC. Return a 1D array (only first species)
def competition_model(rng, a, b, size=None):
    a_scalar = a.item() if hasattr(a, "item") else float(a)
    b_scalar = b.item() if hasattr(b, "item") else float(b)
    result = odeint(dX_dt, y0=X0, t=t, rtol=0.05, atol=0.05, args=(a_scalar, b_scalar, c, d))
    return result[:, 0]  # Return only the first species (prey)


# Step 5: Bayesian inference with PyMC
with pm.Model() as model_lv:
    # Priors
    a = pm.HalfNormal("a", 1.0)
    b = pm.HalfNormal("b", 1.0)
    # Likelihood (ABC). Epsilon is the initial tolerance
    sim = pm.Simulator("sim", competition_model, params=(a, b), epsilon=15, observed=observed)
    # Inference
    # samples = pm.sample_smc()
    samples = pm.sample_smc(draws=500, chains=4, threshold=0.3, correlation_threshold=0.1)
    # Convert to ArviZ InferenceData
    posterior = samples.posterior.stack(samples=("draw", "chain"))
    # post = posterior.to_pandas()

# Plotting
## Plot posterior predictive
_, ax = plt.subplots(figsize=(14, 6))
ax.plot(t, observed_vector, "o", label="prey (observed)", c="C0", mec="k")

mean_a = posterior["a"].mean().item()
mean_b = posterior["b"].mean().item()
mean_sim = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(mean_a, mean_b, c, d))
ax.plot(t, mean_sim[:, 0], linewidth=3, label="mean prey", c="C0")
ax.plot(t, mean_sim[:, 1], linewidth=3, label="mean predator (unobserved)", c="C1")

for i in np.random.default_rng(42).integers(0, posterior.sizes["samples"], 75):
    ai = posterior["a"].values[i]
    bi = posterior["b"].values[i]
    sim_i = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(ai, bi, c, d))
    ax.plot(t, sim_i[:, 0], alpha=0.1, c="C0")
    ax.plot(t, sim_i[:, 1], alpha=0.1, c="C1")

ax.set_xlabel("time")
ax.set_ylabel("population")
ax.legend()
plt.tight_layout()
plt.savefig("results/lotka_unobserved_predictive.pdf")
plt.close()

## Plot posterior distributions
az.plot_posterior(samples)
plt.tight_layout()
plt.savefig("results/lotka_unobserved_marginals.pdf")
plt.close()

## Plot diagnostics
az.plot_trace(samples)
plt.tight_layout()
plt.savefig("results/lotka_unobserved_trace.pdf")
plt.close()
