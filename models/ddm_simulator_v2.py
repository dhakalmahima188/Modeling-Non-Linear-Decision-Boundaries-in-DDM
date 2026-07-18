#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
os.environ["KERAS_BACKEND"] = "jax"
import bayesflow as bf


# ## Boundary shape functions
# 
# Each function returns the decision boundary height `B(t)` at every time point `t`, for one of four DDM variants:
# 
# - `fixed_boundary(t, a)` — constant boundary at value `a` (the classic, non-collapsing DDM).
# - `linear_collapse(t, a, k)` — boundary shrinks linearly: `B(t) = max(a - k*t, epsilon)`. `k` = collapse speed.
# - `exponential_collapse(t, a, lam)` — boundary shrinks exponentially: `B(t) = max(a * exp(-lam*t), epsilon)`. `lam` = decay rate; larger = faster collapse.
# - `sigmoid_collapse(t, a, k_sig, t_mid)` — boundary stays near `a`, then collapses sharply around time `t_mid`, with steepness `k_sig`.
# 
# All four floor the boundary at a tiny `epsilon` so it never reaches exactly 0 (avoids downstream division/logic issues).
# 
# **Theory**: collapsing boundaries model the idea that people get less cautious as time runs out, the amount of evidence needed to commit to a choice shrinks as a deadline approaches.
# 

# In[3]:


def fixed_boundary(t,a):
    """Constant boundary at value a."""
    return np.full_like(t, a)


def linear_collapse(t,a,k,epsilon=1e-4):
    """
    Linearly collapsing boundary.
    B(t) = max(a - k*t, epsilon)
    """
    return np.maximum(a - k * t, epsilon)


def exponential_collapse(t, a, lam, epsilon=1e-4):
    """
    Exponentially collapsing boundary.
    B(t) = max(a * exp(-lambda * t), epsilon)

    Parameters
    ----------
    lam : float
        Decay rate (lambda). Larger = faster collapse.
    """
    return np.maximum(a * np.exp(-lam * t), epsilon)


def sigmoid_collapse(t, a, k_sig, t_mid, epsilon=1e-4):
    """
    Sigmoid (logistic) collapsing boundary.
    B(t) = max(a * (1 - sigmoid(k * (t - t_mid))), epsilon)

    Parameters
    ----------
    k_sig  : float  Steepness of collapse (larger = sharper)
    t_mid  : float  Time of inflection point
    """
    sig = 1.0 / (1.0 + np.exp(-k_sig * (t - t_mid)))
    return np.maximum(a * (1.0 - sig), epsilon)


# ## Model registry
# 
# - `BOUNDARY_FNS` — a dictionary mapping model name (`"fixed"`, `"linear"`, `"exponential"`, `"sigmoid"`) to its function from the cell above. Lets other code pick a boundary function by string name instead of writing `if/elif` chains.
# - `SHAPE_PRIOR_RANGES` — for each model, which *extra* shape parameters it needs beyond the shared ones (`v_scale`, `t0`, `a`, `sv`), and the plausible (min, max) range each parameter can take. These ranges act as **priors**, the space of values considered plausible before seeing any data.
# 

# In[4]:


BOUNDARY_FNS = {
    "fixed": fixed_boundary,
    "linear": linear_collapse,
    "exponential": exponential_collapse,
    "sigmoid": sigmoid_collapse,
}

# Which shape-parameters each model needs, and their prior ranges.
# (v_scale, t0, a, sv are shared across all models.)
SHAPE_PRIOR_RANGES = {
    "fixed":       {},
    "linear":      {"k":     (0.1, 3.0)},
    "exponential": {"lam":   (0.05, 3.0)},
    "sigmoid":     {"k_sig": (0.5, 10.0), "t_mid": (0.2, 1.5)},
}


# ## `sample_cell_priors` : draw one random parameter set
# 
# Randomly draws a full set of DDM parameters for **one experimental cell** (one participant × cue × deadline combination), generalized to work with any of the four boundary models.
# 
# - `v_scale`, `t0`, `sv` : drawn from fixed distributions, shared across all models. `t0` = non-decision time (perception + motor response, unrelated to the evidence accumulation itself). `sv` = trial-to-trial variability in drift rate.
# - **Boundary separation `a`** : drawn, then shifted by `cue`: speed cue narrows it (`-0.15`), accuracy cue widens it (`+0.15`). This encodes the classic **speed–accuracy tradeoff (SAT)**: narrower boundary → faster but less accurate; wider boundary → slower but more accurate.
# - **Collapse-rate shape parameter** (`k`, `lam`, or `k_sig`, depending on model), if `has_deadline=1`, drawn from the *steeper* half of its prior range (faster collapse); otherwise from the *shallower* half. This encodes "under time pressure, people abandon caution faster." Note `t_mid` (sigmoid inflection *timing*) is **not** affected by deadline, only the *rate* of collapse is, mirroring the empirical dissociation reported in Katsimpokis et al. (2020).
# 

# In[5]:


def sample_cell_priors(rng, model="exponential", cue=None, has_deadline=None):
    """
    Priors for ONE (participant x cue x deadline) cell, generalized to any
    boundary model.

      - cue in {0, 1}, has_deadline in {0, 1} ->
        mirrors the empirical dissociation reported in Katsimpokis et al.
        (2020): cue (0=speed, 1=accuracy) shifts the initial boundary
        separation 'a' (accuracy -> wider, speed -> narrower), while
        has_deadline shifts the collapse-rate shape parameter (k / lam /
        k_sig) toward the steeper half of its prior range. The sigmoid
        inflection timing (t_mid) is left unaffected by deadline, since
        the paper's dissociation is specifically about *rate* of collapse,
        not *timing* of collapse.
    """
    params = {
        "v_scale": rng.normal(8.0, 2.0),                  # coherence -> drift slope
        "t0":      rng.uniform(0.15, 0.40),                # non-decision time
        "sv":      np.abs(rng.normal(0.15, 0.1)) + 0.01,   # trial-to-trial drift SD
    }

    # --- cue effect on initial boundary separation ---
    a_base = np.abs(rng.normal(1.0, 0.3)) + 0.2
    if cue is None:
        cue_shift = 0.0
    else:
        cue_shift = {0: -0.15, 1: 0.15}[cue]  # 0=speed, 1=accuracy
    params["a"] = max(a_base + cue_shift, 0.1)

    # --- deadline effect on collapse-rate shape parameter ---
    for name, (lo, hi) in SHAPE_PRIOR_RANGES[model].items():
        if has_deadline is not None and name in ("k", "lam", "k_sig"):
            mid = (lo + hi) / 2
            params[name] = rng.uniform(mid, hi) if has_deadline else rng.uniform(lo, mid)
        else:
            params[name] = rng.uniform(lo, hi)

    return params


# ## `simulate_ddm_batch` : the core trial-by-trial simulator
# 
# Runs a vectorized **Euler–Maruyama** simulation (a standard numerical method for stochastic differential equations) for `n_trials` of a single cell, all sharing one drift `v` and boundary time series.
# 
# **1. Trial-to-trial drift variability**
# Each trial gets its own drift rate drawn from `Normal(v, sv)`, since real decision-makers don't have identical evidence quality every trial.
# 
# **2. Initialize starting positions**
# Converts starting-point fraction `z` (0.5 = unbiased/centered) into an actual position between `-a` and `+a`.
# 
# **3. Setup tracking arrays**
# - `rts` — reaction times, start as NaN (unset).
# - `choices` — start at `-1` (non-response / not yet decided).
# - `active` — boolean mask marking trials still accumulating evidence.
# 
# **4. Determine steps to simulate**
# Runs for the full boundary length, or stops early at the deadline if one is set.
# 
# **5. Time-stepping loop** : for each timestep:
# - Early-exits once all trials are finished (`if not active.any(): break`).
# - Checks only `active` trials for boundary crossings: `x >= b` → upper/choice=1; `x <= -b` → lower/choice=0.
# - Records RT as `t0 + step * dt` (non-decision time + accumulation time) and removes finished trials from `active`.
# - Updates still-active trials' evidence:
# This is one discretized step of the DDM's stochastic differential equation `dx = v·dt + noise_sd·dW`. Scaling noise by `sqrt(dt)` (not `dt`) is required so the noise variance grows correctly (linearly) with time, matching Brownian motion.
# 
# **Returns**: `rts` (NaN = non-response) and `choices` (1 = upper/correct, 0 = lower/error, -1 = non-response/deadline reached).
# 
# 
# 

# In[6]:


def simulate_ddm_batch(v, a, t0, z, sv, boundary, dt, n_trials, deadline_step, noise_sd, rng):
    """
    Vectorized Euler-Maruyama DDM simulation for n_trials of ONE cell.
    `boundary` is a precomputed time series (shape: max_steps,).
    `v` may be scalar or per-trial array (drift variability handled by
    sampling v_trial ~ Normal(v, sv) once, outside the time loop).

    Returns
    -------
    rts     : (n_trials,) seconds, NaN = non-response
    choices : (n_trials,) 1=upper/correct, 0=lower/error, -1=non-response
    """
    v_trial = rng.normal(v, sv, n_trials) if sv > 0 else np.full(n_trials, v)

    x = np.full(n_trials, (z - 0.5) * 2 * a, dtype=float)
    rts = np.full(n_trials, np.nan)
    choices = np.full(n_trials, -1, dtype=int)
    active = np.ones(n_trials, dtype=bool)
    max_steps = len(boundary) if deadline_step is None else min(len(boundary), deadline_step)

    for step in range(max_steps):
        if not active.any():
            break
        b = boundary[step]
        upper_hit = active & (x >= b)
        lower_hit = active & (x <= -b)

        rts[upper_hit] = t0 + step * dt
        choices[upper_hit] = 1
        rts[lower_hit] = t0 + step * dt
        choices[lower_hit] = 0
        active &= ~(upper_hit | lower_hit)

        x[active] += v_trial[active] * dt + noise_sd * np.sqrt(dt) * rng.standard_normal(active.sum())

    return rts, choices


# ## `simulate_cell` : simulate all difficulty levels for one cell
# 
# Wraps `simulate_ddm_batch` across multiple **coherence** levels (stimulus difficulty) for a single cell, using one boundary-shape model.
# 
# **1. Build the boundary time series once**
# Computes `B(t)` at every time point **once** using this cell's `a` and shape parameter(s) — since the boundary shape doesn't depend on coherence, it's reused across all difficulty levels below. `SHAPE_PRIOR_RANGES[model]` picks out only the kwargs relevant to this model.
# 
# **2. Convert deadline time → deadline step**
# The inner simulator loops over discrete steps, not continuous time, so the deadline (in seconds) is converted to a step count.
# 
# **3. Loop over coherence levels**
# Coherence = stimulus difficulty (higher = easier/clearer stimulus). Drift rate scales linearly with it: harder stimulus → weaker evidence → smaller `v` → slower, less accurate decisions. `diff_idx` is a 1-indexed label used later for grouping/plotting, not part of the simulation math.
# 
# **4. Simulate and collect**
# Calls `simulate_ddm_batch` for `n_trials_per_diff` trials at this coherence (reusing the same boundary array and shared `a`, `t0`, `sv`, only `v` changes across the loop), then tags each trial with its coherence value and difficulty index.
# 
# **5. Return**
# Concatenates all difficulty levels into flat 1D arrays: `rts`, `choices`, `coherence_values`, `difficulty_idx`, each of length `n_trials_per_diff * len(coherences)`.
# 
# **Why structured this way**: the boundary is computed once per cell (not per trial or per coherence) since it's independent of difficulty, only drift `v` varies within the loop, directly implementing "harder stimuli → weaker evidence" while keeping boundary/timing parameters fixed within the cell.
# 

# In[7]:


def simulate_cell(params, model, coherences, n_trials_per_diff, dt, max_time,
                   deadline_time, z=0.5, noise_sd=1.0, rng=None):
    """Simulate all coherence levels for one cell using a given boundary model.

    Returns
    -------
    rts, choices, coherence_values, difficulty_idx
        difficulty_idx is a 1-indexed categorical label (1..len(coherences))
        for each trial, in the same order as `coherences` was given.
    """
    if rng is None:
        rng = np.random.default_rng()

    boundary_fn = BOUNDARY_FNS[model]
    time_axis = np.arange(int(max_time / dt)) * dt
    boundary = boundary_fn(time_axis, a=params["a"], **{k: params[k] for k in SHAPE_PRIOR_RANGES[model]})
    deadline_step = int(deadline_time / dt) if deadline_time is not None else None

    rts_all, choices_all, diff_all, diff_idx_all = [], [], [], []
    for diff_idx, coherence in enumerate(coherences, start=1):
        v = params["v_scale"] * coherence
        rts, choices = simulate_ddm_batch(
            v=v, a=params["a"], t0=params["t0"], z=z, sv=params["sv"],
            boundary=boundary, dt=dt, n_trials=n_trials_per_diff,
            deadline_step=deadline_step, noise_sd=noise_sd, rng=rng,
        )
        rts_all.append(rts); choices_all.append(choices)
        diff_all.append(np.full(n_trials_per_diff, coherence))
        diff_idx_all.append(np.full(n_trials_per_diff, diff_idx, dtype=int))

    return (np.concatenate(rts_all), np.concatenate(choices_all),
            np.concatenate(diff_all), np.concatenate(diff_idx_all))



# ## `simulate_full_dataset` : generate a full synthetic experiment
# 
# Top-level loop generating an entire synthetic dataset: for every participant × cue × deadline-condition combination, it:
# 
# 1. Converts string labels to integers (`cue: "speed"→0, "accuracy"→1`; `deadline_cond: "deadline"→1, "no_deadline"→0`).
# 2. Samples a fresh parameter set for that cell via `sample_cell_priors`.
# 3. Simulates trials for that cell across all coherence levels via `simulate_cell`.
# 4. Appends each trial as a row (`rt`, `acc`, participant id, deadline condition, cue, coherence, difficulty) to a growing list.
# 5. Also stores the true generating parameters for that cell (used later to check parameter recovery).
# 
# **Returns**: two DataFrames :
# - `df` : one row per simulated trial (the full synthetic experiment).
# - `true_params` : one row per cell, with the ground-truth parameters used to generate it.
# 
# Note: `acc` (accuracy) is `NaN` for non-responses (`choice == -1`), otherwise `1` if the upper/correct boundary was hit, else `0`.
# 

# In[8]:


def simulate_full_dataset(
    model="exponential",
    n_participants=24,
    coherences=(0.03, 0.07, 0.11, 0.15, 0.19, 0.23),
    cues=("speed", "accuracy"),
    deadline_conditions=("deadline", "no_deadline"),
    n_trials_per_cell=50,
    dt=0.002,
    deadline_time=1.35,
    max_time=3.0,
    noise_sd=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    rows, true_params = [], []

    for pp in range(1, n_participants + 1):
        for cue in cues:
            for deadline_cond in deadline_conditions:
                cue_int = 0 if cue == "speed" else 1
                deadline_int = 1 if deadline_cond == "deadline" else 0

                p = sample_cell_priors(rng, model=model, cue=cue_int, has_deadline=deadline_int)
                ddl = deadline_time if deadline_cond == "deadline" else None

                rts, choices, coh, diff_idx = simulate_cell(
                    p, model, coherences, n_trials_per_cell, dt, max_time, ddl,
                    noise_sd=noise_sd, rng=rng,
                )
                for rt, ch, c, d in zip(rts, choices, coh, diff_idx):
                    rows.append({
                        "rt": rt, "acc": np.nan if ch == -1 else int(ch == 1),
                        "pp": pp, "deadline": deadline_cond, "cue": cue,
                        "coherence": c, "difficulty": d,
                    })
                true_params.append({"pp": pp, "cue": cue, "deadline": deadline_cond, **p})

    return pd.DataFrame(rows), pd.DataFrame(true_params)


# 
# - `MODEL_NAMES` — the four boundary models, indexed for use with an integer `model_idx` (needed because BayesFlow's neural network works with numeric arrays, not strings).
# - `DT`, `MAX_TIME` — simulation time resolution and horizon.
# - `COHERENCES` — the six stimulus difficulty levels used in every simulated cell.
# - `N_TRIALS_PER_DIFF` — trials simulated per difficulty level (→ `N_TRIALS_PER_DIFF * len(COHERENCES)` trials per cell).
# - `DEADLINE_TIME` — deadline in seconds when a deadline condition is active.
# 
# 

# In[9]:


MODEL_NAMES = ["fixed", "linear", "exponential", "sigmoid"]

DT = 0.002
MAX_TIME = 3.0
COHERENCES = (0.03, 0.07, 0.11, 0.15, 0.19, 0.23)
N_TRIALS_PER_DIFF = 20          # -> 120 trials per simulated "cell"
DEADLINE_TIME = 1.35



# ## `prior()` and `likelihood()` — the BayesFlow simulator interface
# 
# BayesFlow's simulation-based inference needs two functions chained together: one that samples parameters, and one that generates data given those parameters.
# 
# **`prior()`**
# Randomly picks:
# - `model_idx` — which of the 4 boundary models to use (this is itself a quantity the network will learn to recover, i.e. model identification).
# - `cue`, `has_deadline` — known experimental design variables (not inferred — they're treated as given, like in a real experiment).
# - All DDM parameters via `sample_cell_priors`, which already applies the cue/deadline effects on `a` and the collapse-rate parameter.
# 
# Returns a flat dictionary with every parameter the network needs to learn to predict, defaulting model-irrelevant shape parameters (e.g. `k` when the model is `"sigmoid"`) to `0.0`.
# 
# **`likelihood(model_idx, cue, has_deadline, v_scale, t0, a, sv, k, lam, k_sig, t_mid)`**
# Given one draw from `prior()`, simulates that cell's data:
# - Selects the correct model name and only the shape parameter(s) it needs.
# - Sets the deadline (`DEADLINE_TIME` or `None`) based on `has_deadline`.
# - Calls `simulate_cell` to generate `rt`, `choice`, `coherence` arrays.
# - Recodes non-responses' RT as `-1.0` (instead of `NaN`) since neural networks generally can't handle `NaN` inputs directly.
# 
# Together, `prior()` and `likelihood()` define the full generative model BayesFlow will learn to invert (simulated data → true parameters).
# 

# In[10]:


def prior():
    """Samples model identity + all latent DDM parameters for one cell.

    cue and has_deadline are known experimental design variables (sampled
    here, not inferred). Their causal effect on 'a' and on the collapse-rate
    shape parameter is applied inside sample_cell_priors, mirroring the
    dissociation reported in Katsimpokis et al. (2020): cue -> boundary
    separation, deadline -> collapse rate.
    """
    rng = np.random.default_rng()
    model_idx = rng.integers(0, len(MODEL_NAMES))
    model = MODEL_NAMES[model_idx]

    cue = rng.integers(0, 2)          # 0 = speed, 1 = accuracy
    has_deadline = rng.integers(0, 2) # 0 = no deadline, 1 = deadline

    p = sample_cell_priors(rng, model=model, cue=int(cue), has_deadline=int(has_deadline))

    return {
        "model_idx": model_idx,
        "cue": cue,
        "has_deadline": has_deadline,
        "v_scale": p["v_scale"], "t0": p["t0"], "a": p["a"], "sv": p["sv"],
        "k":     p.get("k", 0.0),
        "lam":   p.get("lam", 0.0),
        "k_sig": p.get("k_sig", 0.0),
        "t_mid": p.get("t_mid", 0.0),
    }


def likelihood(model_idx, cue, has_deadline, v_scale, t0, a, sv, k, lam, k_sig, t_mid):
    """Simulate one cell's (rt, choice, coherence) trials.

    cue and has_deadline are pass-through design/conditioning variables:
    their causal effect on 'a' and the collapse-rate was already applied
    upstream in prior(). Here has_deadline is additionally used for the
    actual trial-cutoff truncation.
    """
    rng = np.random.default_rng()
    model = MODEL_NAMES[model_idx]
    shape_params = {"linear": {"k": k}, "exponential": {"lam": lam},
                    "sigmoid": {"k_sig": k_sig, "t_mid": t_mid}, "fixed": {}}[model]
    params = {"v_scale": v_scale, "t0": t0, "a": a, "sv": sv, **shape_params}

    ddl = DEADLINE_TIME if has_deadline else None
    rts, choices, coh, diff_idx = simulate_cell(
        params, model, COHERENCES, N_TRIALS_PER_DIFF, DT, MAX_TIME, ddl, rng=rng
    )
    rts_clean = np.where(choices == -1, -1.0, rts)
    return {
        "cue": cue,
        "has_deadline": has_deadline,
        "rt": rts_clean.astype("float32"),
        "choice": choices.astype("float32"),
        "coherence": coh.astype("float32"),
    }



# ## Build and test the BayesFlow simulator
# 
# 
# 
# Chains `prior` and `likelihood` into a single BayesFlow simulator object, then draws 8 simulated datasets as a sanity check. Printing each output's shape confirms the pipeline produces correctly structured batches before using it to train an actual neural network approximator.
# 

# In[11]:


simulator = bf.make_simulator([prior, likelihood])

sample = simulator.sample(batch_shape=(8,))
for k, v in sample.items():
    print(k, np.asarray(v).shape)


# In[12]:


def plot_boundary_shapes(a=1.0, k=1.0, lam=0.8, k_sig=5.0, t_mid=0.7,
                          max_time=2.0, dt=0.002, deadline=1.35):
    t = np.arange(int(max_time / dt)) * dt
    curves = {
        "fixed":       fixed_boundary(t, a=a),
        "linear":      linear_collapse(t, a=a, k=k),
        "exponential": exponential_collapse(t, a=a, lam=lam),
        "sigmoid":     sigmoid_collapse(t, a=a, k_sig=k_sig, t_mid=t_mid),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, b in curves.items():
        ax.plot(t, b, label=name, lw=2)
        ax.plot(t, -b, color=ax.lines[-1].get_color(), lw=2)
    ax.axvline(deadline, color="k", linestyle=":", alpha=0.6, label=f"deadline ({deadline}s)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Boundary position")
    ax.set_title("Boundary shapes (upper/lower)")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


plot_boundary_shapes()


# In[13]:


def _defective_cdf(rts, correct_mask, n_total):
    rt_c = rts[correct_mask].dropna()
    rt_e = rts[~correct_mask & rts.notna()].dropna()
    p_c, p_e = len(rt_c) / n_total, len(rt_e) / n_total
    x_c, y_c = np.sort(rt_c), np.arange(1, len(rt_c) + 1) / max(len(rt_c), 1) * p_c
    x_e, y_e = np.sort(rt_e), np.arange(1, len(rt_e) + 1) / max(len(rt_e), 1) * p_e
    return (x_c, y_c), (x_e, y_e)


def plot_insight_summary(df, title=""):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    colors = {"speed": "#d95f02", "accuracy": "#1b9e77"}
    linestyles = {"deadline": "-", "no_deadline": "--"}

    summary = df.groupby(["cue", "deadline"]).agg(
        mean_rt=("rt", "mean"), accuracy=("acc", "mean")
    ).reset_index()
    labels = [f"{r.cue}\n{r.deadline}" for r in summary.itertuples()]
    bar_colors = [colors[r.cue] for r in summary.itertuples()]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(labels, summary["mean_rt"], color=bar_colors, alpha=0.85)
    ax1.set_ylabel("Mean RT (s)"); ax1.set_title("Mean RT by condition")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(labels, summary["accuracy"], color=bar_colors, alpha=0.85)
    ax2.set_ylabel("Accuracy"); ax2.set_ylim(0, 1); ax2.set_title("Accuracy by condition")

    ax3 = fig.add_subplot(gs[0, 2])
    psych = df.groupby(["cue", "deadline", "coherence"]).agg(accuracy=("acc", "mean")).reset_index()
    for cue in colors:
        for dl in linestyles:
            sub = psych[(psych.cue == cue) & (psych.deadline == dl)].sort_values("coherence")
            ax3.plot(sub.coherence, sub.accuracy, marker="o", color=colors[cue],
                     linestyle=linestyles[dl], label=f"{cue}, {dl}")
    ax3.set_xlabel("Coherence"); ax3.set_ylabel("Accuracy")
    ax3.set_title("Psychometric function"); ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[1, 0])
    chrono = df.groupby(["cue", "deadline", "coherence"]).agg(mean_rt=("rt", "mean")).reset_index()
    for cue in colors:
        for dl in linestyles:
            sub = chrono[(chrono.cue == cue) & (chrono.deadline == dl)].sort_values("coherence")
            ax4.plot(sub.coherence, sub.mean_rt, marker="o", color=colors[cue],
                     linestyle=linestyles[dl], label=f"{cue}, {dl}")
    ax4.set_xlabel("Coherence"); ax4.set_ylabel("Mean RT (s)")
    ax4.set_title("Chronometric function"); ax4.legend(fontsize=7)

    ax5 = fig.add_subplot(gs[1, 1])
    for dl, color in zip(["deadline", "no_deadline"], ["steelblue", "firebrick"]):
        sub = df[df.deadline == dl]
        (xc, yc), (xe, ye) = _defective_cdf(sub["rt"], sub["acc"] == 1, len(sub))
        ax5.plot(xc, yc, color=color, label=f"correct, {dl}")
        ax5.plot(xe, ye, color=color, linestyle="--", label=f"error, {dl}")
    ax5.set_xlabel("RT (s)"); ax5.set_ylabel("Defective CDF")
    ax5.set_title("Slow-errors signature"); ax5.legend(fontsize=7)

    ax6 = fig.add_subplot(gs[1, 2])
    nonresp = df.assign(nonresp=df["acc"].isna())
    rate = nonresp.groupby(["cue", "deadline"])["nonresp"].mean().reset_index()
    bar_colors2 = [colors[r.cue] for r in rate.itertuples()]
    labels2 = [f"{r.cue}\n{r.deadline}" for r in rate.itertuples()]
    ax6.bar(labels2, rate["nonresp"], color=bar_colors2, alpha=0.85)
    ax6.set_ylabel("Non-response rate"); ax6.set_title("Missed-deadline rate")

    plt.suptitle(title or "Simulated SAT dataset: cue & deadline effects", fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()






# In[14]:


import os
os.makedirs("../data", exist_ok=True)

true_params = []
for model in ["fixed", "linear", "exponential", "sigmoid"]:
    df, params_df = simulate_full_dataset(model=model, n_participants=6, n_trials_per_cell=30, seed=1)
    df.to_csv(f"../data/simulated_data_{model}.csv", index=False)
    params_df["model"] = model
    true_params.append(params_df)

    if model == "exponential":
        plot_insight_summary(df, title=f"Simulated SAT dataset ({model} boundary)")

true_params_df = pd.concat(true_params, ignore_index=True)
true_params_df.to_csv("../data/true_params.csv", index=False)

