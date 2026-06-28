# Collapsing Decision Boundaries: SBI Project Guide
### Simulation-Based Inference with BayesFlow — Evidence Accumulation Models

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Domain Knowledge](#2-domain-knowledge)
3. [Major Workflows](#3-major-workflows)
4. [Experiment Setup & Data Description](#4-experiment-setup--data-description)
5. [Implementation Guide](#5-implementation-guide)
6. [Sample Code](#6-sample-code)
7. [Evaluation & Discussion](#7-evaluation--discussion)
8. [References](#8-references)

---

## 1. Project Overview

### Goal
Extend prior EAM modeling work (Katsimpokis et al., 2020) by implementing **non-linear collapsing boundaries** and/or **urgency gating mechanisms** within a simulation-based inference (SBI) framework using BayesFlow. Fit these models to empirical RT + choice data and compare their descriptive adequacy against fixed and linearly-collapsing boundary variants.

### Research Question
> Do non-linear boundary functions (exponential, sigmoid) or urgency gating improve model fit over linear collapsing boundaries — and does this depend on the type of speed-accuracy trade-off manipulation used?

### Deliverables
- Implemented EAM simulator with ≥2 non-linear boundary variants
- Trained BayesFlow amortized posterior
- Posterior predictive checks vs. empirical data
- Model comparison (qualitative or via posterior predictive metrics)
- Written discussion of results

---

## 2. Domain Knowledge

### 2.1 Speeded Decision Tasks
Participants make forced-choice decisions (e.g., is this stimulus left or right?) under time pressure. The data are:
- **RT**: continuous, typically right-skewed (range ~200ms–2000ms)
- **Choice**: binary (correct / error)

Two SAT (speed-accuracy trade-off) manipulations in Katsimpokis et al.:
| Condition | Manipulation | Best-fitting boundary |
|---|---|---|
| Speed instruction | Verbal emphasis on speed | Fixed boundary |
| Response deadline | Hard cutoff time enforced | Linear collapsing boundary |

### 2.2 Evidence Accumulation Models (EAMs)

EAMs treat decision-making as a **noisy accumulation process** in continuous time. The most common EAM is the **Diffusion Decision Model (DDM)**:

```
dX(t) = v·dt + σ·dW(t)
```

where:
- `X(t)` = evidence at time t
- `v` = drift rate (signal quality / task difficulty)
- `σ` = noise (usually fixed to 1)
- `dW(t)` = Wiener noise increment
- Decision occurs when `X(t)` hits upper boundary `+a` (correct) or lower boundary `-a` (error)

**Core DDM Parameters:**

| Parameter | Symbol | Interpretation |
|---|---|---|
| Drift rate | v | Speed of evidence accumulation; reflects task difficulty |
| Boundary separation | a | Response caution; wider = slower but more accurate |
| Non-decision time | t0 | Encoding + motor execution time |
| Starting point | z | A priori bias toward one response |

### 2.3 Boundary Functions

This project requires implementing alternatives to fixed boundaries:

#### Fixed Boundary
```
B(t) = a   (constant)
```

#### Linear Collapsing Boundary
```
B(t) = max(a - k·t, ε)
```
- `k` = collapse rate
- `ε` = minimum boundary (prevents B→0)

#### Exponential Collapsing Boundary *(to implement)*
```
B(t) = a · exp(-λ·t)
```
- `λ` = decay rate
- Collapses fast early, asymptotes toward 0

#### Sigmoid / Logistic Collapsing Boundary *(to implement)*
```
B(t) = a · (1 - 1/(1 + exp(-k·(t - t_mid))))
```
- `k` = steepness, `t_mid` = inflection point
- Stays near `a` initially, then collapses sharply around `t_mid`

#### Urgency Gating *(alternative to implement)*
The urgency signal `u(t)` is added to the accumulated evidence rather than modifying the threshold:
```
Decision rule: |X(t) + u(t)| ≥ a
u(t) = g·t   (linear urgency)
u(t) = g·t²  (quadratic urgency)
```
This is mechanistically distinct from collapsing boundaries — the boundary stays fixed but accumulated evidence is boosted by urgency.

### 2.4 Simulation-Based Inference (SBI) with BayesFlow

Classical fitting (e.g., MLE via grid approximation) is expensive for EAMs because the likelihood is analytically intractable for collapsing boundaries.

**SBI bypasses likelihood computation** by learning the posterior `p(θ | data)` from simulations:

```
1. Sample θ ~ p(θ)           # prior
2. Simulate data ~ p(x | θ)  # forward model / simulator
3. Train neural network to learn p(θ | x) amortized
4. At inference: pass real data → instant posterior
```

BayesFlow implements this via:
- **Summary Network**: compresses variable-length RT/choice data to fixed summary statistics
- **Inference Network** (Normalizing Flow / Coupling Layers): maps summaries to posterior over θ

---

## 3. Major Workflows

### Workflow Overview

```
[Prior] → [Simulator] → [Summary Network] → [Inference Network] → [Posterior]
              ↑                                                          ↓
         [Boundary                                              [Posterior Predictive
          Function]                                              Check vs. Real Data]
```

---

### Step 1: Define Prior Distributions

Specify plausible ranges for each model parameter. These must be **broad enough** to cover all realistic values but **not pathological**.

**Tasks:**
- Define prior `p(θ)` for each parameter
- Use truncated normal or uniform distributions
- Ensure prior predictive data looks realistic (prior predictive check)

**Key considerations:**
- Drift `v`: can be negative (error-prone), typically `v ~ Normal(0, 2)`
- Boundary `a`: must be positive; `a ~ HalfNormal(1.0)`
- Collapse rate `λ` or `k`: must be positive; consider log-normal prior
- Non-decision time `t0`: strictly bounded below 0 and above minimum RT in data

---

### Step 2: Implement the Simulator

The simulator generates synthetic RT + choice data given a parameter vector θ. This is the **core scientific component**.

**Tasks:**
- Implement Euler-Maruyama numerical integration of the diffusion process
- Implement all boundary functions (fixed, linear, exponential, sigmoid, urgency)
- Output: `(RT, choice)` pairs per trial
- Handle non-responses (trials that don't cross boundary before deadline)

**Key considerations:**
- Time step `dt` should be small (0.001s) for accuracy
- Cap maximum integration time (e.g., 3s or experimental deadline)
- Vectorize simulation over multiple trials and parameter samples

---

### Step 3: Configure BayesFlow

**Tasks:**
- Define `bf.simulation.Prior` wrapping your prior sampler
- Define `bf.simulation.Simulator` wrapping your RT simulator
- Create `bf.simulation.GenerativeModel` combining both
- Define `bf.networks.SetTransformer` or `bf.networks.DeepSet` as summary network
- Define `bf.networks.InvertibleNetwork` (coupling flow) as inference network
- Assemble `bf.amortizers.SingleModelAmortizer`

**Key considerations:**
- Summary network must handle **variable-length** trial sequences (different N per participant)
- Input to summary net: matrix `[N_trials × 2]` (RT, choice per row)
- Output of summary net: fixed-length vector fed to inference net

---

### Step 4: Train the Amortized Posterior

**Tasks:**
- Generate offline dataset of `(θ, x)` pairs (e.g., 50,000–200,000 simulations)
- Train summary + inference networks jointly via `amortizer.train()`
- Monitor loss convergence
- Run simulation-based calibration (SBC) to check posterior validity

**Key considerations:**
- Training is **offline** — do it once, then query posterior for any dataset instantly
- SBC: check that rank statistics of θ under posterior are uniform (calibration diagnostic)
- Use GPU if available; training typically takes 30–120 min depending on simulation cost

---

### Step 5: Fit to Empirical Data

**Tasks:**
- Load Katsimpokis et al. (2020) dataset
- Filter by condition (speed instruction vs. response deadline)
- Pass empirical data through trained amortizer → obtain posterior samples
- Compute posterior means/credible intervals per participant

**Key considerations:**
- Condition-level and individual-level fits may differ
- Check that empirical data lies within prior predictive range (otherwise posterior is unreliable)

---

### Step 6: Posterior Predictive Checks (PPC)

**Tasks:**
- Sample θ from posterior → re-simulate RT/choice
- Compare simulated vs. empirical:
  - RT quantiles (10th, 25th, 50th, 75th, 90th percentile)
  - Accuracy rates
  - Conditional accuracy functions (CAF): accuracy as function of RT quantile
  - Defective CDFs: CDF of RTs separately for correct/error responses

**Key considerations:**
- PPCs are the primary model evaluation tool (no analytic likelihood = no BIC/AIC directly)
- Discrepancies in error RT quantiles are especially diagnostic

---

### Step 7: Model Comparison

**Tasks:**
- Compare fixed / linear / exponential / sigmoid / urgency models via PPCs
- Optionally compute approximate Bayes Factors via marginal likelihood estimation
- Replicate Katsimpokis et al.'s condition-specific pattern

---

## 4. Experiment Setup & Data Description

### Katsimpokis et al. (2020) Paradigm

**Task**: Random dot motion task (RDK)
- Participants see a cloud of dots, some fraction moving coherently left or right
- Task: identify the dominant motion direction (left / right)
- Difficulty manipulated by coherence level

**SAT Manipulations** (between- or within-subjects):
1. **Speed instruction**: "Please respond as fast as possible while maintaining some accuracy"
2. **Response deadline**: A visual countdown or enforced cutoff — response after deadline triggers penalty or feedback

**Data structure** (per trial):
```
participant_id | condition | coherence | RT (s) | choice (0=error, 1=correct)
```

**Typical empirical patterns:**
- Speed instruction → shift in RT distribution; boundary estimated to be fixed
- Deadline → fast right-skewed RT distribution; linear collapse estimated to fit better
- Non-responses under deadline condition must be handled (censored observations)

### Handling Non-Responses

When a participant fails to respond before the deadline, the trial is **right-censored**:
- Include as `RT = deadline, choice = NaN` or as a separate censoring indicator
- Simulator must also produce non-response trials at the same rate as empirical data
- PPCs must match non-response rate

---

## 5. Implementation Guide

### Software Requirements

```
python >= 3.9
bayesflow >= 1.1
numpy
scipy
matplotlib
seaborn
pandas
torch (PyTorch, for BayesFlow backend)
```

### File Structure

```
project/
├── data/
│   └── katsimpokis2020.csv         # empirical data
├── models/
│   ├── boundaries.py               # boundary function implementations
│   ├── ddm_simulator.py            # Euler-Maruyama DDM simulator
│   └── priors.py                   # prior distributions
├── training/
│   ├── configure_bayesflow.py      # BayesFlow network setup
│   └── train.py                    # training loop
├── analysis/
│   ├── fit_empirical.py            # posterior inference on real data
│   ├── ppc.py                      # posterior predictive checks
│   └── model_comparison.py
├── notebooks/
│   └── exploration.ipynb
└── results/
    ├── figures/
    └── posteriors/
```

---

## 6. Sample Code

### 6.1 Boundary Functions

```python
# models/boundaries.py
import numpy as np

def fixed_boundary(t: np.ndarray, a: float, **kwargs) -> np.ndarray:
    """Constant boundary at value a."""
    return np.full_like(t, a)


def linear_collapse(t: np.ndarray, a: float, k: float,
                    epsilon: float = 0.05, **kwargs) -> np.ndarray:
    """
    Linearly collapsing boundary.
    B(t) = max(a - k*t, epsilon)
    """
    return np.maximum(a - k * t, epsilon)


def exponential_collapse(t: np.ndarray, a: float, lam: float,
                          epsilon: float = 1e-4, **kwargs) -> np.ndarray:
    """
    Exponentially collapsing boundary.
    B(t) = a * exp(-lambda * t)
    
    Parameters
    ----------
    lam : float
        Decay rate (lambda). Larger = faster collapse.
    """
    return np.maximum(a * np.exp(-lam * t), epsilon)


def sigmoid_collapse(t: np.ndarray, a: float, k_sig: float,
                      t_mid: float, **kwargs) -> np.ndarray:
    """
    Sigmoid (logistic) collapsing boundary.
    B(t) = a * (1 - sigmoid(k * (t - t_mid)))
    
    Parameters
    ----------
    k_sig  : float  Steepness of collapse (larger = sharper)
    t_mid  : float  Time of inflection point
    """
    sig = 1.0 / (1.0 + np.exp(-k_sig * (t - t_mid)))
    return np.maximum(a * (1.0 - sig), 1e-4)


def urgency_gate(t: np.ndarray, g: float, power: float = 1.0, **kwargs) -> np.ndarray:
    """
    Urgency signal added to accumulated evidence.
    u(t) = g * t^power
    
    Returns the urgency value (not boundary — add to evidence X).
    power=1 → linear urgency
    power=2 → quadratic urgency
    """
    return g * (t ** power)


# Registry for easy access
BOUNDARY_FUNCTIONS = {
    'fixed': fixed_boundary,
    'linear': linear_collapse,
    'exponential': exponential_collapse,
    'sigmoid': sigmoid_collapse,
}
```

---

### 6.2 DDM Simulator

```python
# models/ddm_simulator.py
import numpy as np
from typing import Callable, Optional, Tuple

def simulate_ddm_trials(
    v: float,
    a: float,
    t0: float,
    z: float,                         # starting point bias, in [0,1]; 0.5 = unbiased
    boundary_fn: Callable,
    boundary_params: dict,
    n_trials: int = 200,
    dt: float = 0.001,
    max_time: float = 3.0,
    deadline: Optional[float] = None, # if set, truncate at deadline
    noise_sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate DDM via Euler-Maruyama integration.
    
    Returns
    -------
    rts     : shape (n_trials,)  RT in seconds; np.nan = non-response
    choices : shape (n_trials,)  1=correct (upper), 0=error (lower), -1=non-response
    """
    if rng is None:
        rng = np.random.default_rng()
    
    rts = np.full(n_trials, np.nan)
    choices = np.full(n_trials, -1, dtype=int)
    
    max_steps = int(max_time / dt)
    time_axis = np.arange(max_steps) * dt
    
    # Pre-compute boundary values at each time step
    boundary = boundary_fn(time_axis, a=a, **boundary_params)
    
    for i in range(n_trials):
        # Starting point: z is relative [0,1]; convert to absolute position
        x = (z - 0.5) * 2 * a   # maps z=0.5 → 0, z=0 → -a, z=1 → +a
        crossed = False
        
        for step in range(max_steps):
            t_current = step * dt
            
            # Check deadline
            if deadline is not None and t_current >= deadline:
                # Non-response
                break
            
            b = boundary[step]
            
            # Check boundaries
            if x >= b:
                rts[i] = t0 + t_current
                choices[i] = 1  # correct
                crossed = True
                break
            elif x <= -b:
                rts[i] = t0 + t_current
                choices[i] = 0  # error
                crossed = True
                break
            
            # Euler-Maruyama update
            x += v * dt + noise_sd * np.sqrt(dt) * rng.standard_normal()
    
    return rts, choices


def simulate_ddm_urgency(
    v: float,
    a: float,
    t0: float,
    z: float,
    g: float,                          # urgency gain
    urgency_power: float = 1.0,
    n_trials: int = 200,
    dt: float = 0.001,
    max_time: float = 3.0,
    deadline: Optional[float] = None,
    noise_sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DDM with urgency gating: fixed boundary but decision rule includes urgency signal.
    
    X(t) + u(t) ≥ a  →  upper response
    X(t) - u(t) ≤ -a →  lower response
    """
    if rng is None:
        rng = np.random.default_rng()
    
    rts = np.full(n_trials, np.nan)
    choices = np.full(n_trials, -1, dtype=int)
    
    max_steps = int(max_time / dt)
    
    for i in range(n_trials):
        x = (z - 0.5) * 2 * a
        
        for step in range(max_steps):
            t_current = step * dt
            
            if deadline is not None and t_current >= deadline:
                break
            
            # Urgency signal at current time
            u = g * (t_current ** urgency_power)
            
            if (x + u) >= a:
                rts[i] = t0 + t_current
                choices[i] = 1
                break
            elif (x - u) <= -a:
                rts[i] = t0 + t_current
                choices[i] = 0
                break
            
            x += v * dt + noise_sd * np.sqrt(dt) * rng.standard_normal()
    
    return rts, choices
```

---

### 6.3 Prior Distributions

```python
# models/priors.py
import numpy as np
from scipy import stats

def sample_prior_exponential_boundary(n_samples: int, rng=None) -> np.ndarray:
    """
    Prior for DDM with exponential collapsing boundary.
    
    Parameters (5D):
    ----------------
    v    ~ Normal(0, 2)       drift rate
    a    ~ HalfNormal(1.0)    boundary separation (must be > 0)
    t0   ~ Uniform(0.05, 0.5) non-decision time
    z    ~ Uniform(0.3, 0.7)  starting point (near 0.5 = unbiased)
    lam  ~ HalfNormal(0.5)    exponential decay rate (must be > 0)
    
    Returns
    -------
    samples : shape (n_samples, 5)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    v   = rng.normal(0, 2, n_samples)
    a   = np.abs(rng.normal(0, 1.0, n_samples)) + 0.1       # HalfNormal + offset
    t0  = rng.uniform(0.05, 0.5, n_samples)
    z   = rng.uniform(0.3, 0.7, n_samples)
    lam = np.abs(rng.normal(0, 0.5, n_samples)) + 0.01      # decay rate > 0
    
    return np.stack([v, a, t0, z, lam], axis=-1)


def sample_prior_sigmoid_boundary(n_samples: int, rng=None) -> np.ndarray:
    """
    Prior for DDM with sigmoid collapsing boundary.
    
    Parameters (6D):
    ----------------
    v      ~ Normal(0, 2)
    a      ~ HalfNormal(1.0)
    t0     ~ Uniform(0.05, 0.5)
    z      ~ Uniform(0.3, 0.7)
    k_sig  ~ HalfNormal(5.0)    collapse steepness
    t_mid  ~ Uniform(0.3, 1.5)  inflection point
    """
    if rng is None:
        rng = np.random.default_rng()
    
    v     = rng.normal(0, 2, n_samples)
    a     = np.abs(rng.normal(0, 1.0, n_samples)) + 0.1
    t0    = rng.uniform(0.05, 0.5, n_samples)
    z     = rng.uniform(0.3, 0.7, n_samples)
    k_sig = np.abs(rng.normal(0, 5.0, n_samples)) + 0.5
    t_mid = rng.uniform(0.3, 1.5, n_samples)
    
    return np.stack([v, a, t0, z, k_sig, t_mid], axis=-1)


def sample_prior_urgency(n_samples: int, rng=None) -> np.ndarray:
    """
    Prior for DDM with urgency gating.
    
    Parameters (5D):
    ----------------
    v     ~ Normal(0, 2)
    a     ~ HalfNormal(1.0)
    t0    ~ Uniform(0.05, 0.5)
    z     ~ Uniform(0.3, 0.7)
    g     ~ HalfNormal(1.0)    urgency gain
    """
    if rng is None:
        rng = np.random.default_rng()
    
    v  = rng.normal(0, 2, n_samples)
    a  = np.abs(rng.normal(0, 1.0, n_samples)) + 0.1
    t0 = rng.uniform(0.05, 0.5, n_samples)
    z  = rng.uniform(0.3, 0.7, n_samples)
    g  = np.abs(rng.normal(0, 1.0, n_samples)) + 0.01
    
    return np.stack([v, a, t0, z, g], axis=-1)
```

---

### 6.4 BayesFlow Configuration & Training

```python
# training/configure_bayesflow.py
import bayesflow as bf
import numpy as np
from models.boundaries import exponential_collapse
from models.ddm_simulator import simulate_ddm_trials
from models.priors import sample_prior_exponential_boundary


# ─── 1. Wrap Prior ───────────────────────────────────────────────────────────

def prior_fn(batch_size):
    """Returns dict with 'prior_draws' key: shape (batch_size, n_params)."""
    samples = sample_prior_exponential_boundary(batch_size)
    return {"prior_draws": samples}


# ─── 2. Wrap Simulator ───────────────────────────────────────────────────────

def simulator_fn(prior_draws: np.ndarray,
                 n_trials: int = 200,
                 deadline: float = 2.0) -> dict:
    """
    Simulate DDM for a batch of parameter vectors.
    
    Returns
    -------
    dict with 'sim_data' : shape (batch_size, n_trials, 2)
        Last dim: [RT (or 0 if non-response), choice (1/0/-1)]
    """
    batch_size = prior_draws.shape[0]
    rng = np.random.default_rng()
    
    all_data = []
    for i in range(batch_size):
        v, a, t0, z, lam = prior_draws[i]
        
        rts, choices = simulate_ddm_trials(
            v=v, a=a, t0=t0, z=z,
            boundary_fn=exponential_collapse,
            boundary_params={'lam': lam},
            n_trials=n_trials,
            dt=0.001,
            max_time=deadline + 0.5,
            deadline=deadline,
            rng=rng,
        )
        
        # Replace NaN RTs with 0; encode non-response choice as -1
        rts_clean = np.where(np.isnan(rts), 0.0, rts)
        trial_data = np.stack([rts_clean, choices.astype(float)], axis=-1)
        all_data.append(trial_data)
    
    return {"sim_data": np.array(all_data)}  # (batch_size, n_trials, 2)


# ─── 3. Build Generative Model ───────────────────────────────────────────────

prior = bf.simulation.Prior(batch_prior_fun=prior_fn)
simulator = bf.simulation.Simulator(batch_simulator_fun=simulator_fn)
generative_model = bf.simulation.GenerativeModel(prior=prior, simulator=simulator)


# ─── 4. Summary Network (handles variable-length sequences) ──────────────────

summary_net = bf.networks.SetTransformer(
    input_dim=2,       # (RT, choice) per trial
    output_dim=64,     # summary vector size
    num_seeds=4,       # attention seeds
    summary_dim=64,
)


# ─── 5. Inference Network (normalizing flow) ─────────────────────────────────

inference_net = bf.networks.InvertibleNetwork(
    num_params=5,      # v, a, t0, z, lam
    num_coupling_layers=8,
    coupling_design='affine',
    permutation='fixed',
    use_act_norm=True,
)


# ─── 6. Assemble Amortizer ───────────────────────────────────────────────────

amortizer = bf.amortizers.SingleModelAmortizer(
    inference_net=inference_net,
    summary_net=summary_net,
)


# ─── 7. Trainer ──────────────────────────────────────────────────────────────

trainer = bf.trainers.Trainer(
    amortizer=amortizer,
    generative_model=generative_model,
    configurator=bf.configuration.DefaultModelComparison(),  # handles input prep
)
```

```python
# training/train.py
from configure_bayesflow import trainer

history = trainer.train_online(
    epochs=50,
    iterations_per_epoch=1000,
    batch_size=64,
    validation_sims=500,
)

# Save
trainer.amortizer.save_weights("results/amortizer_exp_boundary.h5")
```

---

### 6.5 Posterior Inference on Empirical Data

```python
# analysis/fit_empirical.py
import numpy as np
import pandas as pd
import bayesflow as bf
from configure_bayesflow import amortizer, summary_net

# Load and preprocess empirical data
df = pd.read_csv("data/katsimpokis2020.csv")

def prepare_participant_data(participant_df: pd.DataFrame,
                              max_trials: int = 200) -> np.ndarray:
    """
    Format participant RT/choice data for the amortizer.
    
    Returns
    -------
    data : shape (1, n_trials, 2)
    """
    rts = participant_df['rt'].values
    choices = participant_df['correct'].values.astype(float)
    
    # Handle non-responses: set RT=0, choice=-1
    non_response_mask = participant_df['rt'].isna()
    rts[non_response_mask] = 0.0
    choices[non_response_mask] = -1.0
    
    trial_data = np.stack([rts, choices], axis=-1)
    return trial_data[np.newaxis, :max_trials, :]  # (1, n_trials, 2)


# Condition-level fitting
conditions = df['condition'].unique()
posteriors = {}

for cond in conditions:
    cond_df = df[df['condition'] == cond]
    participant_ids = cond_df['participant_id'].unique()
    
    posteriors[cond] = {}
    for pid in participant_ids:
        p_df = cond_df[cond_df['participant_id'] == pid]
        obs_data = prepare_participant_data(p_df)
        
        # Load weights (if not already loaded)
        amortizer.load_weights("results/amortizer_exp_boundary.h5")
        
        # Infer posterior (1000 samples)
        post_samples = amortizer.sample({"sim_data": obs_data}, n_samples=1000)
        posteriors[cond][pid] = post_samples  # shape (1000, 5)

# Save posteriors
import pickle
with open("results/posteriors/exp_boundary_posteriors.pkl", "wb") as f:
    pickle.dump(posteriors, f)
```

---

### 6.6 Posterior Predictive Checks

```python
# analysis/ppc.py
import numpy as np
import matplotlib.pyplot as plt
from models.ddm_simulator import simulate_ddm_trials
from models.boundaries import exponential_collapse


def posterior_predictive_rts(post_samples: np.ndarray,
                               n_trials: int = 500,
                               n_ppc_samples: int = 200,
                               deadline: float = 2.0) -> dict:
    """
    Generate posterior predictive RT distributions.
    
    Returns dict with keys 'correct_rts', 'error_rts', 'non_response_rate'.
    """
    rng = np.random.default_rng()
    
    # Randomly draw parameter sets from posterior
    idx = rng.choice(post_samples.shape[0], n_ppc_samples, replace=False)
    
    all_correct_rts = []
    all_error_rts = []
    all_nr_rates = []
    
    for i in idx:
        v, a, t0, z, lam = post_samples[i]
        
        rts, choices = simulate_ddm_trials(
            v=v, a=a, t0=t0, z=z,
            boundary_fn=exponential_collapse,
            boundary_params={'lam': lam},
            n_trials=n_trials,
            deadline=deadline,
            rng=rng,
        )
        
        correct_mask = choices == 1
        error_mask = choices == 0
        nr_mask = choices == -1
        
        all_correct_rts.append(rts[correct_mask])
        all_error_rts.append(rts[error_mask])
        all_nr_rates.append(nr_mask.mean())
    
    return {
        'correct_rts': all_correct_rts,
        'error_rts': all_error_rts,
        'non_response_rate': np.array(all_nr_rates),
    }


def plot_rt_quantiles(ppc_dict: dict,
                       empirical_correct: np.ndarray,
                       empirical_error: np.ndarray,
                       quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)) -> plt.Figure:
    """
    Quantile-probability (QP) plot comparing PPC vs. empirical data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    q = np.array(quantiles)
    
    for ax, label, ppc_rts, emp_rts in zip(
        axes,
        ['Correct Responses', 'Error Responses'],
        [ppc_dict['correct_rts'], ppc_dict['error_rts']],
        [empirical_correct, empirical_error]
    ):
        # PPC quantiles (distribution over posterior samples)
        ppc_q = np.array([np.quantile(rt_set, q) if len(rt_set) > 0
                           else np.full_like(q, np.nan)
                           for rt_set in ppc_rts])
        
        ppc_median = np.nanmedian(ppc_q, axis=0)
        ppc_lo = np.nanpercentile(ppc_q, 5, axis=0)
        ppc_hi = np.nanpercentile(ppc_q, 95, axis=0)
        
        # Empirical quantiles
        emp_q = np.quantile(emp_rts, q) if len(emp_rts) > 0 else np.full_like(q, np.nan)
        
        # Plot
        ax.fill_between(q, ppc_lo, ppc_hi, alpha=0.3, color='steelblue', label='90% PPC')
        ax.plot(q, ppc_median, '-', color='steelblue', label='PPC median')
        ax.plot(q, emp_q, 'o--', color='firebrick', label='Empirical')
        ax.set_xlabel('Probability (RT quantile)')
        ax.set_ylabel('RT (s)')
        ax.set_title(label)
        ax.legend()
    
    fig.suptitle('Posterior Predictive Check: RT Quantiles', fontsize=13)
    plt.tight_layout()
    return fig


def plot_conditional_accuracy(ppc_dict: dict,
                               empirical_rts: np.ndarray,
                               empirical_choices: np.ndarray,
                               n_bins: int = 5) -> plt.Figure:
    """
    Conditional Accuracy Function (CAF): accuracy as a function of RT bin.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Empirical CAF
    quantile_edges = np.quantile(empirical_rts, np.linspace(0, 1, n_bins + 1))
    emp_acc = []
    bin_centers = []
    for lo, hi in zip(quantile_edges[:-1], quantile_edges[1:]):
        mask = (empirical_rts >= lo) & (empirical_rts < hi)
        emp_acc.append(empirical_choices[mask].mean())
        bin_centers.append((lo + hi) / 2)
    
    ax.plot(bin_centers, emp_acc, 'o--', color='firebrick', label='Empirical')
    ax.set_xlabel('RT bin center (s)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Conditional Accuracy Function')
    ax.axhline(0.5, ls=':', color='gray')
    ax.legend()
    
    return fig
```

---

### 6.7 Simulation-Based Calibration (SBC)

```python
# analysis/sbc_check.py
"""
SBC verifies that the posterior is well-calibrated.
For each simulated dataset: sample θ*, simulate data, get posterior,
then compute rank of θ* among posterior samples.
Rank statistics should be uniform if calibrated.
"""
import numpy as np
import matplotlib.pyplot as plt
from configure_bayesflow import amortizer, generative_model

def run_sbc(n_sbc_runs: int = 1000, n_posterior_samples: int = 100) -> np.ndarray:
    """
    Returns rank statistics: shape (n_sbc_runs, n_params)
    """
    param_names = ['v', 'a', 't0', 'z', 'lam']
    n_params = len(param_names)
    ranks = np.zeros((n_sbc_runs, n_params), dtype=int)
    
    for i in range(n_sbc_runs):
        # Sample ground truth θ*
        sim_out = generative_model(batch_size=1)
        theta_star = sim_out['prior_draws'][0]   # shape (n_params,)
        obs_data = sim_out['sim_data']           # shape (1, n_trials, 2)
        
        # Get posterior samples
        post_samples = amortizer.sample({"sim_data": obs_data},
                                         n_samples=n_posterior_samples)  # (n_post, n_params)
        
        # Rank of θ* in posterior
        for p in range(n_params):
            ranks[i, p] = np.sum(post_samples[:, p] < theta_star[p])
    
    return ranks


def plot_sbc_ranks(ranks: np.ndarray, param_names: list) -> plt.Figure:
    """Plot rank histograms — should be approximately uniform."""
    n_params = ranks.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(ranks[:, i], bins=20, density=True, color='steelblue', edgecolor='white')
        ax.axhline(1.0 / 20, color='firebrick', linestyle='--', label='Uniform')
        ax.set_title(f'SBC: {name}')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Density')
    
    fig.suptitle('Simulation-Based Calibration', fontsize=13)
    plt.tight_layout()
    return fig
```

---

## 7. Evaluation & Discussion

### Model Evaluation Criteria

Since the likelihood is intractable, evaluation relies on:

1. **Posterior Predictive Checks (PPC)**
   - RT quantile plots (per condition, separated by correct/error)
   - Conditional accuracy functions
   - Accuracy rates and non-response rates
   - Defective CDFs

2. **Simulation-Based Calibration (SBC)**
   - Rank histograms should be uniform → ensures reliable posterior

3. **Qualitative Model Comparison**
   - Which boundary function best reproduces empirical RT distributions?
   - Does the pattern match Katsimpokis et al.? (linear collapse better for deadline, fixed better for speed instruction)
   - Does your non-linear variant improve or match the linear collapse under deadline?

4. **Parameter Interpretability**
   - Do posterior boundary parameters (e.g., `lam`, `k_sig`) take meaningful values?
   - Are credible intervals tight (informative posteriors)?

### Expected Discussion Points

- **Exponential vs. sigmoid collapse**: Exponential collapses immediately and may overfit fast responses; sigmoid preserves the boundary longer before a sharp collapse, which may fit deadline data better.
- **Urgency gating vs. boundary collapse**: Mechanistically different — urgency changes *internal evidence accumulation dynamics*, not the criterion. May explain cases where accuracy doesn't monotonically decrease with RT.
- **Condition specificity**: Replicate Katsimpokis et al.'s finding that the best boundary form depends on the SAT manipulation type.
- **Limitations**: Euler-Maruyama introduces discretization error; small `dt` reduces this but increases simulation time.

---

## 8. References

- Katsimpokis, D., Hawkins, G. E., & van Maanen, L. (2020). Not all speed-accuracy trade-off manipulations have the same psychological effect. *Computational Brain & Behavior*, 3(1), 1–13.
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, 20(4), 873–922.
- Boehm, U. et al. (2018). Estimating across-trial variability parameters of the Diffusion Decision Model. *Behavior Research Methods*.
- Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. *PNAS*.
- Radev, S. T. et al. (2022). BayesFlow: Learning complex stochastic models with invertible neural networks. *IEEE Transactions on Neural Networks and Learning Systems*.
- Drugowitsch, J. et al. (2012). The cost of accumulating evidence in perceptual decision making. *Journal of Neuroscience*.
- Hawkins, G. E. et al. (2015). Revisiting the evidence for collapsing boundaries and urgency signals in perceptual decision-making. *Journal of Neuroscience*.
