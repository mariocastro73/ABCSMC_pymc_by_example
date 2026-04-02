import arviz as az
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless plot saving
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from pathlib import Path
from scipy.integrate import odeint

# Fixed model parameters
c = 1.5
d = 0.75

# Load observed data
data = np.loadtxt("observed_data.csv", delimiter=",", skiprows=1)
t = data[:, 0]
X0 = data[1, 1:3]
observed_matrix = data[:, 1:3]
observed = observed_matrix.reshape(-1)  # 1D vector for PyMC

# Precompute normalized observation arrays (avoid per-call computation in distance)
_max_prey = float(np.max(np.abs(observed_matrix[:, 0])))
_max_pred = float(np.max(np.abs(observed_matrix[:, 1])))
_obs_prey_norm = observed_matrix[:, 0] / _max_prey  # shape (T,)
_obs_pred_norm = observed_matrix[:, 1] / _max_pred  # shape (T,)
_n_times = len(t)


def dX_dt(X, t, a, b, c, d):
    return np.array([a * X[0] - b * X[0] * X[1],
                     -c * X[1] + d * b * X[0] * X[1]])


def competition_model(rng, a, b, size=None):
    a_scalar = a.item() if hasattr(a, "item") else float(a)
    b_scalar = b.item() if hasattr(b, "item") else float(b)
    result = odeint(dX_dt, y0=X0, t=t, rtol=0.05, atol=0.05, args=(a_scalar, b_scalar, c, d))
    return result.reshape(-1)


def norm1_normalized_distance(epsilon, obs_data, sim_data):
    """L1 distance with each species normalized by its observed maximum.

    Precomputed normalized obs arrays avoid per-call reshape and max operations.
    Returns log-likelihood via a Laplace-like kernel: -distance / epsilon.
    """
    sim_2d = sim_data.reshape(_n_times, 2)
    d_prey = np.sum(np.abs(_obs_prey_norm - sim_2d[:, 0] / _max_prey))
    d_pred = np.sum(np.abs(_obs_pred_norm - sim_2d[:, 1] / _max_pred))
    return -(d_prey + d_pred) / epsilon


def save_plots(idata: az.InferenceData, output_prefix: str) -> None:
    """Save posterior predictive and marginal plots as PDFs."""
    Path(output_prefix).parent.mkdir(exist_ok=True)
    posterior = idata.posterior.stack(samples=("draw", "chain"))

    # Plot 1: Posterior predictive trajectories
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, observed_matrix[:, 0], "o", label="prey (obs)", c="C0", mec="k")
    ax.plot(t, observed_matrix[:, 1], "o", label="predator (obs)", c="C1", mec="k")
    mean_a = posterior["a"].mean().item()
    mean_b = posterior["b"].mean().item()
    mean_sim = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(mean_a, mean_b, c, d))
    ax.plot(t, mean_sim[:, 0], linewidth=3, label="mean prey", c="C0")
    ax.plot(t, mean_sim[:, 1], linewidth=3, label="mean predator", c="C1")
    rng = np.random.default_rng(42)
    n_samples = posterior.dims["samples"]
    for i in rng.integers(0, n_samples, size=75):
        ai = float(posterior["a"].values[i])
        bi = float(posterior["b"].values[i])
        sim_i = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(ai, bi, c, d))
        ax.plot(t, sim_i[:, 0], alpha=0.1, c="C0")
        ax.plot(t, sim_i[:, 1], alpha=0.1, c="C1")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_predictive.pdf")
    plt.close(fig)

    # Plot 2: Posterior marginals
    axes = az.plot_posterior(idata, kind="hist", bins=30)
    fig2 = axes.flatten()[0].get_figure()
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_marginals.pdf")
    plt.close(fig2)



def run_sampling() -> az.InferenceData:
    """Run ABC-SMC inference and return InferenceData. This is the function the
    agent optimizes. Priors and the ODE model are fixed and must not change."""
    with pm.Model() as model_lv:
        # Priors — FROZEN, do not modify
        a = pm.HalfNormal("a", 1.0)
        b = pm.HalfNormal("b", 1.0)
        # Likelihood (ABC)
        sim = pm.Simulator(
            "sim",
            competition_model,
            params=(a, b),
            distance=norm1_normalized_distance,
            epsilon=1,
            observed=observed,
        )
        samples = pm.sample_smc(draws=500, chains=4, threshold=0.3, correlation_threshold=0.1)
    return samples


if __name__ == "__main__":
    samples = run_sampling()
    az.summary(samples, hdi_prob=0.95)
    save_plots(samples, "results/standalone")
    print("Plots saved to results/standalone_*.pdf")
