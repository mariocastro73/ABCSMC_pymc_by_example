# Setup Instructions

This guide explains how to create a Python virtual environment and install the required dependencies.

```bash
# 1. Create a virtual environment (named .venv)
python3 -m venv .venv

# 2. Activate the environment
# On Linux/MacOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install scikit-learn numpy matplotlib pandas scipy pymc

# 5. Verify installation
python -c "import sklearn, numpy, matplotlib, pandas, scipy, pymc; print('All packages installed successfully!')"
```

This script `lotka_ABCSMC.py` implements a predator-prey model (Lotka-Volterra) using Bayesian inference with PyMC. Here's a step-by-step explanation of each block:

---

### **Imports**
```python
import arviz as az 
import matplotlib.pyplot as plt 
import numpy as np 
import pymc as pm 
from scipy.integrate import odeint
```
These libraries are used for:
- `arviz`: Bayesian analysis and diagnostics.
- `matplotlib.pyplot`: plotting.
- `numpy`: numerical operations.
- `pymc`: probabilistic modeling.
- `odeint`: solving systems of ODEs.

---

### **Step 1: Model parameters and initial conditions**
```python
a_true = 1.0
b_true = 0.1
c = 1.5
d = 0.75
X0 = [10.0, 5.0]
size = 100
time = 15
t = np.linspace(0, time, size)
```
Defines:
- True model parameters (`a_true`, `b_true`) for reference.
- Initial populations (`X0`).
- Time range for simulation (`t`).

---

### **Step 2: Load observed data**
```python
data = np.loadtxt("observed_data.csv", delimiter=",", skiprows=1)
t = data[:, 0]
observed_matrix = data[:, 1:3]
observed = observed_matrix.reshape(-1)
```
Loads data from a CSV file:
- `t`: time points.
- `observed_matrix`: prey and predator populations.
- `observed`: reshaped to a 1D vector for PyMC.

---

### **Step 3: Define the ODE system (Lotka-Volterra)**
```python
def dX_dt(X, t, a, b, c, d):
    return np.array([
        a * X[0] - b * X[0] * X[1],
        -c * X[1] + d * b * X[0] * X[1]
    ])
```
Defines the system of differential equations:
- First equation: prey population change.
- Second equation: predator population change.

---

### **Step 4: Simulation model for PyMC**
```python
def competition_model(rng, a, b, size=None):
    a_scalar = a.item() if hasattr(a, "item") else float(a)
    b_scalar = b.item() if hasattr(b, "item") else float(b)
    result = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(a_scalar, b_scalar, c, d))
    return result.reshape(-1)
```
Simulates the Lotka-Volterra system for given `a` and `b`, returning a 1D vector.

---

### **Step 5: Bayesian inference with PyMC**
```python
with pm.Model() as model_lv:
    a = pm.HalfNormal("a", 1.0)
    b = pm.HalfNormal("b", 1.0)
    sim = pm.Simulator("sim", competition_model, params=(a, b), epsilon=10, observed=observed)
    samples = pm.sample_smc()
    posterior = samples.posterior.stack(samples=("draw", "chain"))
```
- Defines the probabilistic model:
  - Priors: `HalfNormal` distributions for `a` and `b`.
  - `Simulator`: uses Approximate Bayesian Computation (ABC) to match simulated data to observations.
  - `sample_smc()`: performs Sequential Monte Carlo sampling.
  - `posterior`: stacks samples for analysis.

---

### **Visualization: Posterior predictive**
```python
_, ax = plt.subplots(figsize=(14, 6))
ax.plot(t, observed_matrix[:, 0], "o", label="prey", c="C0", mec="k")
ax.plot(t, observed_matrix[:, 1], "o", label="predator", c="C1", mec="k")
mean_a = posterior["a"].mean().item()
mean_b = posterior["b"].mean().item()
mean_sim = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(mean_a, mean_b, c, d))
ax.plot(t, mean_sim[:, 0], linewidth=3, label="mean prey", c="C0")
ax.plot(t, mean_sim[:, 1], linewidth=3, label="mean predator", c="C1")
```
Plots:
- Observed data.
- Simulation using mean values of `a` and `b`.

---

### **Visualization: Sample trajectories**
```python
for i in np.random.randint(0, posterior.samples.size, 75):
    ai = posterior["a"].values[i]
    bi = posterior["b"].values[i]
    sim_i = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(ai, bi, c, d))
    ax.plot(t, sim_i[:, 0], alpha=0.1, c="C0")
    ax.plot(t, sim_i[:, 1], alpha=0.1, c="C1")
```
Plots 75 random simulations to visualize uncertainty in predictions.

---

### **Finalize plot**
```python
ax.set_xlabel("time")
ax.set_ylabel("population")
ax.legend()
plt.show()
```

---

### **Visualization: Posterior distributions**
```python
az.plot_posterior(samples)
plt.show()
```
Plots the posterior distributions of `a` and `b`.

---

### **Visualization: Trace diagnostics**
```python
az.plot_trace(samples)
plt.show()
```
Plots trace diagnostics to assess convergence of the sampling.

---

