# Collapsing Decision Boundaries
```bash
https://bayesflow.org/v2.0.12/index.html Bayesflow 2, runs only on version 3.11-3.13, so change the version
brew install python@3.13
~/../../opt/homebrew/opt/python@3.13/bin/python3 -m venv .venv  
```

## Simulation Section : *Mahima & Bishesh*

### 1. What Is This Project About?
The study it builds on is **Katsimpokis, Hawkins & van Maanen (2020)**. They had people watch a screen full of moving dots and quickly decide: are most dots going left or right? Some people were just *told* to be fast. Others faced a hard computerised deadline. The study found that these two types of pressure lead to mathematically different decision strategies, and this project tries to push that finding further by testing more complex mathematical shapes.

Extend prior EAM modeling work (Katsimpokis et al., 2020) by implementing **non-linear collapsing boundaries** and/or **urgency gating mechanisms** within a simulation-based inference (SBI) framework using BayesFlow. Fit these models to empirical RT + choice data and compare their descriptive adequacy against fixed and linearly-collapsing boundary variants.

### Research Question
> Do non-linear boundary functions (exponential, sigmoid) or urgency gating improve model fit over linear collapsing boundaries, and does this depend on the type of speed-accuracy trade-off manipulation used?

---

## 2. Domains

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
| Drift variability | sv | Trial-to-trial variability in drift rate (needed to capture "slow errors") |

### 2.3 Boundary Functions

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

#### Exponential Collapsing Boundary
```
B(t) = max(a · exp(-λ·t), ε)
```
- `λ` = decay rate
- Collapses fast early, asymptotes toward `ε` (never reaches exactly 0)

#### Sigmoid / Logistic Collapsing Boundary
```
B(t) = max(a · (1 - 1/(1 + exp(-k·(t - t_mid)))), ε)
```
- `k` = steepness, `t_mid` = inflection point
- Stays near `a` initially, then collapses sharply around `t_mid`, floored at `ε`

All three collapsing boundaries share the same `ε` floor (default `1e-4`) to prevent the boundary from reaching exactly zero, which would make the process trivially terminate.

---

## 3. Simulation Workflow

### Step 1: Defining Prior Distributions

Specify plausible ranges for each model parameter, broad enough to be sensible
but tight enough to be identifiable.

**Key considerations:**
- Drift `v_scale`: coherence → drift slope, `v_scale ~ Normal(8.0, 2.0)`
- Boundary `a`: must be positive, `a ~ |Normal(1.0, 0.3)| + 0.2`
- Non-decision time `t0`: `t0 ~ Uniform(0.15, 0.40)`
- Drift variability `sv`: `sv ~ |Normal(0.15, 0.1)| + 0.01`
- Collapse-rate shape params (model-dependent, only one set applies per model):
  - Linear: `k ~ Uniform(0.1, 3.0)`
  - Exponential: `lam ~ Uniform(0.05, 3.0)`
  - Sigmoid: `k_sig ~ Uniform(0.5, 10.0)`, `t_mid ~ Uniform(0.2, 1.5)`

Two possible parameterizations were considered:

| Approach | Description |
|---|---|
| **A. Per-cell free priors (no built-in effect)** | Each (participant × cue × deadline) cell draws its own `v_scale, a, t0, sv, k/lam/k_sig` independently from shared priors, with **no** built-in cue/deadline effect. Any cue/deadline difference in the resulting data is emergent noise only — useful as a null-effect baseline to sanity-check that the pipeline doesn't manufacture spurious effects on its own. |
| **B. Cue → `a`, deadline → collapse-rate (implemented)** | Mirrors the *empirical dissociation* found by Katsimpokis et al. (2020) directly in the generative model: the cue (speed/accuracy instruction) shifts the initial boundary separation `a` (±0.15, accuracy wider / speed narrower), and `has_deadline` shifts the collapse-rate shape parameter (`k` / `lam` / `k_sig`) toward the steeper half of its prior range. The sigmoid inflection timing (`t_mid`) is deliberately left unaffected by deadline, since the paper's dissociation is specifically about *rate* of collapse, not *timing* of collapse. |

**Currently implemented: Approach B.** `sample_cell_priors(rng, model, cue=None, has_deadline=None)` — passing `cue=None, has_deadline=None` recovers Approach A (the null-effect baseline) for comparison/ablation purposes. `prior()` (the BayesFlow-facing function) always samples `cue`/`has_deadline` as known design variables and threads them through `sample_cell_priors`, so the training simulator inherits the same dissociation as the exploratory dataset in `simulate_full_dataset`.

<!-- > **Why introduce this at all?** Without it, the `cue` and `deadline` labels in the dataset were purely decorative — nothing in the simulator connected them to the data, so any "effect" visible in diagnostic plots was sampling noise. Approach B makes the simulator a faithful generative model of the paper's actual finding (cue → static, pre-trial caution; deadline → dynamic, within-trial urgency), so that downstream diagnostics/plots show genuine condition differences, and a trained inference network can meaningfully be asked to recover this structure rather than arbitrary noise. -->

> **Note:** the `±0.15` cue shift and the "steeper half of the range" deadline shift are currently *fixed* effect sizes, not inferable parameters, they represent a chosen ground truth for generating synthetic validation data, not something the network is asked to estimate. If the goal becomes recovering the *size* of the cue/deadline effect itself, promote these shifts to free parameters drawn from their own priors instead.

---

### Step 2: Implementation of the Simulator

The simulator generates synthetic RT + choice data given a parameter vector θ.

**Implemented:**
- Euler-Maruyama integration of the diffusion process
- All four boundary functions (fixed, linear, exponential, sigmoid)
- **Vectorized across trials**: the simulator loops over time steps once and updates all trials in a condition cell simultaneously via boolean masks, scaled to participant × cue × deadline × coherence combinations.
- Cue → boundary separation (`a`) and deadline → collapse-rate shape parameter, per the design in Step 1.

**Key considerations:**
- `dt = 0.001–0.002s`
- Max integration time capped at `deadline + buffer` (or 3s if no deadline)

---

## 4. Data Structure

### Data Dictionary: DDM Mapping

| Column | Meaning | Maps to in Our DDM |
| :--- | :--- | :--- |
| **pp** | participant ID | individual differences in `v, t0, a` |
| **cue** | "speed" or "accuracy" instruction | boundary height `a` (cue → threshold, per the paper's finding) |
| **deadline** | "deadline" or "no_deadline" | collapse rate (`λ`/`k`/`k_sig`) + hard cutoff time |
| **coherence** | % of dots moving coherently (Exp. 1 difficulty) | drift rate `v` — higher coherence = stronger signal |
| **difficulty** | categorical index (1–6) of the coherence level | labeled version of coherence, handy for grouping/plotting |
| **rt** | response time (s) | model output |
| **acc** | 1=correct, 0=error, NaN=non-response | model output |

---

## 6. How the Simulation Works (Euler-Maruyama)

We cannot solve the DDM equation perfectly, the mathematics becomes intractable, especially with collapsing boundaries. Instead, we **approximate it in tiny time steps**. This is called the Euler-Maruyama method.

Here is what happens in each simulated trial:

**Step 1 — Set the clock to zero and evidence to the starting value.**
X starts at (z − 0.5) × 2a. If z = 0.5 and a = 1.5, X starts at 0.

**Step 2 — Each timestep (`dt = 0.002s` by default), update the evidence.**
New evidence = old evidence + (drift × dt) + (noise × random jiggle)
The random jiggle comes from a standard normal random number scaled by the square root of the time step (`√dt`).

**Step 3 — Check whether a boundary has been crossed.**
If X ≥ +boundary → record RT = current time + t₀, record choice = correct.
If X ≤ −boundary → record RT = current time + t₀, record choice = error.
If the deadline passes with no crossing → record non-response.

**Step 4 — Repeat for hundreds of simulated trials.**
This produces a synthetic dataset of (RT, choice) pairs that we can compare to real data.

> **Analogy:** Like a video game that cannot run physics continuously, so it recalculates positions 60 times per second (here, ~500 times per second, since `dt = 0.002s`). The smaller the time step (dt), the more accurate the simulation, but smaller dt means more computation.

---


## Neural Network Section : *Esther & Paula*