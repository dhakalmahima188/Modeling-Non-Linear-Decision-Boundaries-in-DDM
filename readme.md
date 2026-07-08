# Collapsing Decision Boundaries

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



## 3. Simulation Workflow



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

## 6. How the Simulation Works (Euler-Maruyama)

We cannot solve the DDM equation perfectly, the mathematics becomes intractable, especially with collapsing boundaries. Instead, we **approximate it in tiny time steps**. This is called the Euler-Maruyama method.

Here is what happens in each simulated trial:

**Step 1 — Set the clock to zero and evidence to the starting value.**
X starts at (z − 0.5) × 2a. If z = 0.5 and a = 1.5, X starts at 0.

**Step 2 — Each 1 millisecond, update the evidence.**
New evidence = old evidence + (drift × 0.001) + (noise × random jiggle)
The random jiggle comes from a standard normal random number scaled by the square root of the time step.

**Step 3 — Check whether a boundary has been crossed.**
If X ≥ +boundary → record RT = current time + t₀, record choice = correct.
If X ≤ −boundary → record RT = current time + t₀, record choice = error.
If the deadline passes with no crossing → record non-response.

**Step 4 — Repeat for hundreds of simulated trials.**
This produces a synthetic dataset of (RT, choice) pairs that we can compare to real data.

> **Analogy:** Like a video game that cannot run physics continuously, so it recalculates positions 60 times per second. The smaller the time step (dt), the more accurate the simulation, but smaller dt means more computation.


---



## Neural Network Section : *Esther & Paula*
