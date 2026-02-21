import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.integrate import odeint

# Step 1: Model parameters and initial conditions
c = 1.5
d = 0.75

# Step 2: read from file and store into t and observed_matrix
data = np.loadtxt("observed_data.csv", delimiter=",", skiprows=1)
t = data[:, 0]
X0 = data[1, 1:3]
observed_matrix = data[:, 1:3]
observed = observed_matrix.reshape(-1) # A vector 1D for PyMC

# Step 3: Define the ODE system
def dX_dt(X, t, a, b, c, d):
    return np.array([a * X[0] - b * X[0] * X[1],
                     -c * X[1] + d * b * X[0] * X[1]])

# Step 4: Define the competition model for PyMC. Return a 1D array
def competition_model(rng, a, b, size=None):
    a_scalar = a.item() if hasattr(a, "item") else float(a)
    b_scalar = b.item() if hasattr(b, "item") else float(b)
    result = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(a_scalar, b_scalar, c, d))
    return result.reshape(-1)


def norm1_normalized_distance(epsilon, obs_data, sim_data):
    """L1 distance with each species normalized by its observed maximum.

    Normalizing by the observed max puts both prey and predator series
    in a comparable [0, 1] scale, giving equal weight to both populations.
    Returns log-likelihood via a Laplace-like kernel: -distance / epsilon.
    """
    n_times = len(obs_data) // 2
    obs_2d = obs_data.reshape(n_times, 2)   # (T, 2): col0=prey, col1=predator
    sim_2d = sim_data.reshape(n_times, 2)

    max_prey = np.max(np.abs(obs_2d[:, 0]))
    max_pred = np.max(np.abs(obs_2d[:, 1]))

    d_prey = np.sum(np.abs(obs_2d[:, 0] / max_prey - sim_2d[:, 0] / max_prey))
    d_pred = np.sum(np.abs(obs_2d[:, 1] / max_pred - sim_2d[:, 1] / max_pred))

    return -(d_prey + d_pred) / epsilon


# Step 5: Bayesian inference with PyMC
if __name__ == "__main__":
    with pm.Model() as model_lv:
        # Priors
        a = pm.HalfNormal("a", 1.0)
        b = pm.HalfNormal("b", 1.0)
        # Likelihood (ABC). With normalization each species contributes values in ~[0,1],
        # so the combined L1 distance is at most 2*T. epsilon=1 is a reasonable starting
        # point; pm.sample_smc will adaptively tighten it during sampling.
        sim = pm.Simulator(
            "sim",
            competition_model,
            params=(a, b),
            distance=norm1_normalized_distance,
            epsilon=1,
            observed=observed,
        )
        # Inference
        samples = pm.sample_smc(draws=500, chains=4) # Faster for testing
        # Convert to ArviZ InferenceData
        posterior = samples.posterior.stack(samples=("draw", "chain"))
        # post = posterior.to_pandas()

    az.summary(samples, hdi_prob=0.95)

    # Plotting
    ## Plot posterior predictive
    _, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, observed_matrix[:, 0], "o", label="prey", c="C0", mec="k")
    ax.plot(t, observed_matrix[:, 1], "o", label="predator", c="C1", mec="k")

    mean_a = posterior["a"].mean().item()
    mean_b = posterior["b"].mean().item()
    mean_sim = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(mean_a, mean_b, c, d))
    ax.plot(t, mean_sim[:, 0], linewidth=3, label="mean prey", c="C0")
    ax.plot(t, mean_sim[:, 1], linewidth=3, label="mean predator", c="C1")

    for i in np.random.randint(0, posterior.samples.size, 75):
        ai = posterior["a"].values[i]
        bi = posterior["b"].values[i]
        sim_i = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(ai, bi, c, d))
        ax.plot(t, sim_i[:, 0], alpha=0.1, c="C0")
        ax.plot(t, sim_i[:, 1], alpha=0.1, c="C1")

    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    plt.show()

    ## Plot posterior distributions
    az.plot_posterior(samples, kind="hist", bins=30)
    plt.show()

    ## Plot diagnostics
    az.plot_trace(samples)
    # az.plot_trace(samples, kind="rank_bars") # Alternative
    plt.suptitle(f"Trace Plot");
    plt.show()

    # with variables a and b in variable posterior, plot a heatmap
    plt.hist2d(posterior["a"].values, posterior["b"].values, bins=100, cmap="Blues")
    plt.colorbar(label="Counts")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title("2D Histogram of Posterior Samples for a and b")
    plt.show()

    # same but with kde
    # plt.figure(figsize=(8, 6))
    az.plot_kde(posterior["a"].values, posterior["b"].values, fill_last=True)
    # az.plot_kde(posterior["a"].values, posterior["b"].values, hdi_probs=[0.05, 0.5, 0.95]) # Alternative
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title("KDE of Posterior Samples for a and b")
    plt.show()

    # plt.figure(figsize=(8, 6))
    az.plot_autocorr(samples)
    plt.show()
