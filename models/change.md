# Blockwise corrections for `ddm_simulator_v3.ipynb`

Cell numbers are 0-indexed as they appear in your notebook.
Verdicts: **REPLACE** = swap the cell contents · **DELETE** = remove the cell · **KEEP** = no change.

Summary of what to do:

| cells | action |
|---|---|
| 0 | KEEP (drop the `tensorflow` import) |
| 1, 2, 3 | REPLACE — boundary functions, priors, design constants |
| 4 | DELETE — `apply_condition_effects` bakes in the answer |
| 5 | REPLACE — two real bugs (deadline offset, per-coherence loop) |
| 6 | DELETE |
| 7 | REPLACE — merged into the new simulator |
| 8 | REPLACE — was giving each participant only one difficulty |
| 9, 10 | REPLACE — the collapse parameters were never inferable |
| 11, 12 | KEEP (small edits) |
| 13–19 | REPLACE — model-specific networks |
| 20, 21, 22 | REPLACE — `EPOCHS=1` |
| 23–28 | KEEP (loop over models) |
| 29 | REPLACE — fragile string mapping |
| 30, 31 | DELETE — chunking is the fatal one |
| 32, 33, 34 | REPLACE — `emp_rt["has_deadline"]` crashes |
| 35, 36 | REPLACE |
| 37, 38 | DELETE |

---

## Part A — Simulation

### Cell 0 — KEEP, minor

Remove `import tensorflow as tf`. You set `KERAS_BACKEND="jax"` on line 2, and cell 18
then calls `tf.stack` inside an adapter transform. Mixing the two is the source of a
whole class of confusing errors. Add `import inspect` — needed in cell 10.

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
import bayesflow as bf

import pickle, time
```

---

### Cell 1 — REPLACE

**What's wrong**

1. `sigmoid_collapse` returns `a * (1 - σ(k_sig·(t - t_mid)))`, so at `t = 0` it gives
   `a · σ(k_sig · t_mid)`, **not `a`**. With `t_mid = 0.2, k_sig = 5` the boundary starts
   27% below `a` at the very first timestep. `a` therefore means something different in
   this model than in `fixed`, and cross-model comparison of caution is confounded.
2. `exponential_collapse` has no asymptote — it always decays to zero. Real boundaries
   plateau; without a floor parameter, `lam` has to absorb both "how fast" and "how far".
3. `epsilon=1e-4` as a keyword on each function is fine but inconsistent across them.

**Replacement**

```python
FLOOR = 1e-3   # boundaries never collapse fully to zero


def b_fixed(t, a):
    """Constant boundary (baseline model)."""
    return np.full_like(t, a)


def b_linear(t, a, k):
    """Linear collapse, rate k per second."""
    return np.maximum(a - k * t, FLOOR)


def b_exponential(t, a, lam, a_min):
    """
    Exponential decay to an asymptote.  b(0) = a,  b(inf) = a * a_min.
    lam   : time constant in s (small = fast collapse)
    a_min : asymptote as a fraction of a
    """
    return np.maximum(a * (a_min + (1.0 - a_min) * np.exp(-t / lam)), FLOOR)


def b_sigmoid(t, a, k_sig, t_mid, a_min):
    """
    Logistic collapse, normalised so that b(0) = a exactly.
        g(t) = 1 / (1 + exp(-k_sig * (t - t_mid)))
        b(t) = a * [a_min + (1 - a_min) * (1 - g(t)) / (1 - g(0))]
    k_sig : steepness    t_mid : inflection point in s    a_min : asymptote fraction

    Since k_sig, t_mid > 0 we have g(0) <= 0.5, so the denominator is bounded
    below by 0.5 and this cannot blow up.
    """
    g0 = 1.0 / (1.0 + np.exp(k_sig * t_mid))
    g = 1.0 / (1.0 + np.exp(-k_sig * (t - t_mid)))
    return np.maximum(a * (a_min + (1.0 - a_min) * (1.0 - g) / (1.0 - g0)), FLOOR)


def b_weibull(t, a, lam, kappa, a_min):
    """
    Weibull-CDF collapse. Nests b_exponential at kappa = 1 and gives a
    'hold-then-drop' shape at kappa > 1.
    """
    return np.maximum(a * (a_min + (1.0 - a_min) * np.exp(-((t / lam) ** kappa))), FLOOR)


def b_urgency(t, a, gamma):
    """
    Urgency gating written as its equivalent collapsing bound: a linearly growing
    urgency signal u(t) = 1 + gamma*t multiplying the evidence against a fixed
    threshold is equivalent to dividing that threshold by u(t).
    """
    return np.maximum(a / (1.0 + gamma * t), FLOOR)
```

Keeping both `b_sigmoid` and `b_weibull` is deliberate: they are different hypotheses
(symmetric S-curve with an interpretable midpoint vs. asymmetric decay), and `t_mid`
is directly comparable against the deadline in your write-up.

---

### Cell 2 — REPLACE

**What's wrong**

The prior ranges are on the wrong scale for your `noise_sd = 0.1`. With `a_base ∈ (0.05, 0.20)`
and `k ∈ (0.01, 0.30)`, a collapse rate of 0.3/s against a boundary of 0.1 means the bound
hits the floor after 0.3 s — the top of your `k` prior is far past anything meaningful, so
most prior mass is in a region where the model is degenerate.

Switch to the `s = 1` convention (standard, better conditioned numerically) and rescale
everything consistently.

**Replacement**

```python
BOUNDARY_FNS = {
    "fixed":       b_fixed,
    "linear":      b_linear,
    "exponential": b_exponential,
    "sigmoid":     b_sigmoid,
    "weibull":     b_weibull,
    "urgency":     b_urgency,
}
MODEL_NAMES = list(BOUNDARY_FNS)

CORE_PRIORS = {
    "v_scale": (0.5, 9.0),     # drift = v_scale * coherence  ->  v in ~0.05-5.4
    "a":       (0.4, 2.0),     # boundary height at t = 0
    "t0":      (0.15, 0.45),   # non-decision time (s)
    "sv":      (0.0, 1.5),     # across-trial drift SD
}

SHAPE_PRIORS = {
    "fixed":       {},
    "linear":      {"k": (0.0, 3.0)},
    "exponential": {"lam": (0.05, 3.0), "a_min": (0.0, 0.9)},
    "sigmoid":     {"k_sig": (2.0, 15.0), "t_mid": (0.10, 1.20), "a_min": (0.0, 0.9)},
    "weibull":     {"lam": (0.05, 3.0), "kappa": (0.5, 5.0), "a_min": (0.0, 0.9)},
    "urgency":     {"gamma": (0.0, 6.0)},
}


def param_names(model):
    """Ordered list of inferred parameters for a boundary model."""
    return list(CORE_PRIORS) + list(SHAPE_PRIORS[model])


def sample_params(model, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    ranges = {**CORE_PRIORS, **SHAPE_PRIORS[model]}
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}


def boundary_trace(model, params, dt=None, max_time=None):
    dt = DT if dt is None else dt
    max_time = MAX_TIME if max_time is None else max_time
    t = np.arange(0.0, max_time, dt)
    shape = {k: params[k] for k in SHAPE_PRIORS[model]}
    return t, BOUNDARY_FNS[model](t, a=params["a"], **shape)
```

Note `a_base` is now just `a`. Search-and-replace that name everywhere downstream.

---

### Cell 3 — REPLACE

**What's wrong**

1. **`FLASH_PAIRS` has 5 levels, your data has 6.** You're missing `(0.65, 0.35)`.
   Cell 29 maps `"[0.65, 0.35]" -> 0.3`, so the empirical data contains a coherence
   of 0.3 that the simulator can never produce. The network is asked to score trials
   from a difficulty it has never seen in training.
2. `MAX_TIME = 3.0` with `DEADLINE_TIME = 1.35`. Under the deadline you integrate 1750
   steps and throw away everything past step 675. That's ~2.5x wasted compute per
   simulation, on the slowest part of the whole pipeline.
3. `DT = 0.002` is finer than you need for RTs quantised at ~10 ms.
4. `CUES` and `DEADLINE_CONDITIONS` drive the loops we're deleting.

**Replacement**

```python
DT = 0.005                 # Euler-Maruyama step (s)
DEADLINE = 1.35            # enforced response deadline -- OVERWRITTEN FROM DATA in cell 29
MAX_TIME = DEADLINE        # nothing past the deadline is observable
S_NOISE = 1.0              # diffusion coefficient (s = 1 convention)

# |p1 - p2| for the six Flash-task levels. Overwritten from data in cell 29.
COHERENCES = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])

N_TRIALS_PER_COH = 60      # -> 360 trials per dataset (= per participant)

# Fixed design cell. Pick one cue, deadline only, and keep them fixed.
CUE = 0                    # 0 = speed, 1 = accuracy
HAS_DEADLINE = 1
```

---

### Cell 4 — DELETE

`sample_participant_priors` + `apply_condition_effects` both go.

**Why this is the most important deletion.** `apply_condition_effects` contains:

```python
if has_deadline:
    params[name] = rng.uniform(midpoint, hi)   # collapse rate in the UPPER half
else:
    params[name] = rng.uniform(lo, midpoint)   # LOWER half
```

You are hardcoding "the deadline causes faster boundary collapse" into the generative
model, then fitting that model and concluding that deadlines cause faster collapse.
That's circular — the result is guaranteed by construction and carries no evidence.

Once the design cell is fixed (one cue, deadline only), the whole function is moot:
there are no conditions left to apply effects across. `difficulty_idx` also disappears,
because a participant now gets *all* difficulties rather than one.

---

### Cell 5 — REPLACE

**What's wrong**

1. **Deadline offset bug.** You compute `deadline_step = int(deadline_time / dt)` and stop
   the loop there, but you record `rts[hit] = t0 + step * dt`. So a trial is only censored
   once *decision time* exceeds 1.35 s, i.e. the effective deadline is `t0 + 1.35` ≈ 1.6 s,
   not 1.35 s. Censoring is the single most informative feature for a collapsing bound, and
   it's being applied at the wrong time for every trial. The available decision time must be
   `deadline - t0`.
2. **The boundary clock and the accumulation clock disagree.** `boundary[step]` is built on
   `t = step * dt` measured from stimulus onset, but accumulation starts at `t0`. Either is
   defensible, but you have to pick one. Standard is to start the collapse when accumulation
   starts; the code as written silently mixes them.
3. `noise_sd = 0.1` with drift reaching `v_scale * coherence = 2.0 * 0.6 = 1.2` makes the
   process nearly deterministic — the diffusion is 12x smaller than the drift. Under `s = 1`
   this is fixed automatically.
4. One call per coherence level (see cell 7) means 5-6 separate Python loops of ~1750 steps.
   Vectorising across all trials at once is a ~6x speedup for free.

**Replacement** — this single function now replaces both cell 5 and cell 7.

```python
def simulate_dataset(model, params, coherences=None, n_per_coh=None,
                     dt=None, deadline=None, s_noise=None, rng=None):
    """
    Simulate one participant in one condition cell.

    The boundary is SHARED across difficulties; only the drift rate changes.
    That shared-boundary constraint across difficulty levels is what makes the
    collapse rate identifiable -- with a single difficulty, fixed and collapsing
    boundaries mimic each other.

    Accuracy coding: drift is positive toward the correct response, so the upper
    boundary is 'correct'. This matches using `acc` as the choice variable.

    Returns
    -------
    rt        : (n,) response time in s, np.nan if no response
    choice    : (n,) 1 = correct, 0 = error, -1 = no response before deadline
    coherence : (n,) difficulty of each trial
    """
    rng = np.random.default_rng() if rng is None else rng
    coherences = COHERENCES if coherences is None else np.asarray(coherences, float)
    n_per_coh = N_TRIALS_PER_COH if n_per_coh is None else n_per_coh
    dt = DT if dt is None else dt
    deadline = DEADLINE if deadline is None else deadline
    s_noise = S_NOISE if s_noise is None else s_noise

    coh = np.repeat(coherences, n_per_coh)
    n = coh.size

    # FIX: decision time available is deadline MINUS non-decision time
    t_dec_max = max(deadline - params["t0"], dt)
    t_grid = np.arange(0.0, t_dec_max, dt)

    shape = {k: params[k] for k in SHAPE_PRIORS[model]}
    bnd = BOUNDARY_FNS[model](t_grid, a=params["a"], **shape)

    v = params["v_scale"] * coh
    if params.get("sv", 0.0) > 0:
        v = rng.normal(v, params["sv"])

    x = np.zeros(n)                       # z = 0.5 relative -> unbiased start
    rt = np.full(n, np.nan)
    choice = np.full(n, -1, dtype=np.int8)
    active = np.ones(n, dtype=bool)
    sqrt_dt = np.sqrt(dt)

    for step, b in enumerate(bnd):
        if not active.any():
            break
        hit_up = active & (x >= b)
        hit_lo = active & (x <= -b)
        t_now = params["t0"] + step * dt
        rt[hit_up] = t_now; choice[hit_up] = 1
        rt[hit_lo] = t_now; choice[hit_lo] = 0
        active &= ~(hit_up | hit_lo)
        na = int(active.sum())
        if na:
            x[active] += v[active] * dt + s_noise * sqrt_dt * rng.standard_normal(na)

    return rt, choice, coh


def pack_observations(rt, choice, coherence):
    """
    Shape trials into the (n_trials, 4) array the summary network consumes.
    Columns: [rt (0 if no response), correct, coherence, responded]

    Non-responses get an explicit indicator column instead of the sentinel
    rt = -1.0 used in your cell 10. A sentinel inside a continuous variable
    forces the network to learn that -1 means 'missing' rather than 'very fast',
    and the deadline-miss rate is one of the most informative statistics you have
    about the collapse rate -- it deserves its own channel.
    """
    choice = np.asarray(choice)
    responded = (choice >= 0).astype(np.float32)
    rt_safe = np.where(choice >= 0, rt, 0.0).astype(np.float32)
    correct = np.where(choice == 1, 1.0, 0.0).astype(np.float32)
    return np.stack([rt_safe, correct,
                     np.asarray(coherence, np.float32), responded], axis=-1)
```

---

### Cell 6 — DELETE

Bare `COHERENCES` display cell.

---

### Cell 7 — DELETE

Absorbed into `simulate_dataset` above. For the record, it had a real bug:

```python
else:
    pairs_to_run = list(zip(coherences, FLASH_PAIRS))
    start = 1                       # <-- should be 0
...
for difficulty_idx, (coherence, pair) in enumerate(pairs_to_run, start=start):
```

When no difficulty is fixed, difficulty indices come out as 1..5 instead of 0..4, so
`difficulty_idx` is off by one relative to `FLASH_PAIRS` / `COHERENCES` everywhere it's
used as an index downstream.

---

### Cell 8 — REPLACE

**What's wrong**

1. `fixed_difficulty_idx=difficulty_idx` with `participant_params["difficulty_idx"] = (pp - 1) % len(FLASH_PAIRS)`
   means **each participant sees exactly one difficulty level**. This is the identifiability
   problem in a nutshell: with one difficulty per dataset there is nothing forcing the
   boundary shape, and `a`, `v`, `t0` and the collapse rate trade off freely.
2. Indentation is broken around `nonresp_reason` — the `if choice == -1:` body sits at a
   different indent level from the `else:` body. It happens to parse, but it's a trap.
3. `nonresp_reason = "timeout_no_deadline"` conflates two different censoring mechanisms
   (a real deadline vs. hitting `MAX_TIME`). Gone once we drop the no-deadline condition.

**Replacement** — validation data only; this is no longer the training path.

```python
def simulate_reference_dataset(model, n_participants=24, seed=0):
    """Simulated 'ground truth' data for the fixed condition cell."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for pp in range(1, n_participants + 1):
        params = sample_params(model, rng)
        rt, choice, coh = simulate_dataset(model, params, rng=rng)
        for r, c, k in zip(rt, choice, coh):
            rows.append({
                "pp": pp, "cue": CUE, "has_deadline": HAS_DEADLINE,
                "difficulty": k, "rt": np.nan if c == -1 else r,
                "acc": np.nan if c == -1 else int(c == 1),
                "missed": int(c == -1),
            })
        truth.append({"pp": pp, "model": model, **params})
    return pd.DataFrame(rows), pd.DataFrame(truth)
```

---

### Cells 9 + 10 — REPLACE

**What's wrong — this is the core failure of the notebook**

`prior()` samples `model_idx` and the shape parameters get sampled *inside*
`apply_condition_effects`, which is called from `likelihood()`. Then cell 18 does:

```python
.drop(["difficulty_idx", "model_idx", "k", "lam", "k_sig", "t_mid"])
```

So `k`, `lam`, `k_sig` and `t_mid` — the collapse parameters, the entire subject of the
project — are generated as latent noise and then thrown away before the network ever
sees them. You are not estimating the boundary shape. You cannot be, given this adapter.

Second problem: `model_idx` is drawn uniformly over 4 boundary shapes and also dropped.
So a single network is being trained on a random mixture of four different generative
models with no indicator of which one produced each dataset. Even the four core
parameters become poorly identified, because the same `(v_scale, t0, a, sv)` maps to
very different data depending on the discarded shape.

Third: `likelihood()` loops over all four cue × deadline combinations and concatenates
them, so `cue` and `has_deadline` come out as length-`n_trials` vectors, which cell 18
then feeds in as `inference_conditions`. Inference conditions must be per-dataset, not
per-trial. With the design cell fixed, they vanish entirely.

**Replacement** — one prior/likelihood pair per boundary model.

```python
def make_prior(model):
    def prior():
        return sample_params(model, np.random.default_rng())
    return prior


def make_likelihood(model):
    names = param_names(model)

    def likelihood(**params):
        rt, choice, coh = simulate_dataset(model, params, rng=np.random.default_rng())
        return {"obs": pack_observations(rt, choice, coh)}

    # BayesFlow matches prior keys to likelihood arguments by name, so the
    # signature has to be explicit rather than **params
    likelihood.__signature__ = inspect.Signature(
        [inspect.Parameter(nm, inspect.Parameter.POSITIONAL_OR_KEYWORD) for nm in names]
    )
    return likelihood
```

---

### Cell 11 — KEEP, with edits

`plot_insight_summary` groups by `["cue", "deadline"]`, which are now constant. Strip
those groupings and keep the psychometric panel (accuracy × coherence) and the defective
CDFs — those are the panels that actually diagnose boundary shape. Add a miss-rate panel.

### Cell 12 — KEEP, with edits

Change the loop to `for model in MODEL_NAMES:` and call `simulate_reference_dataset`.
Add a boundary-shape sanity plot, which would have caught the `b(0) != a` bug immediately:

```python
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(18, 2.6), sharey=True)
for ax, mdl in zip(axes, MODEL_NAMES):
    for _ in range(12):
        p = sample_params(mdl, rng); p["a"] = 1.2
        t, b = boundary_trace(mdl, p)
        ax.plot(t, b, lw=1, alpha=0.6)
    ax.axhline(1.2, ls=":", c="k", lw=0.8)   # every model must start here
    ax.set_title(mdl); ax.set_xlabel("t (s)")
axes[0].set_ylabel("b(t)"); plt.tight_layout(); plt.show()
```

Also add a prior predictive check before you train anything:

```python
for mdl in MODEL_NAMES:
    miss, acc, med = [], [], []
    for _ in range(200):
        rt, ch, _ = simulate_dataset(mdl, sample_params(mdl, rng), rng=rng)
        ok = ch >= 0
        miss.append(1 - ok.mean())
        acc.append(np.mean(ch[ok] == 1) if ok.any() else np.nan)
        med.append(np.nanmedian(rt))
    q = lambda v: np.nanquantile(v, [.1, .5, .9]).round(2)
    print(f"{mdl:12s} miss {q(miss)}   acc {q(acc)}   median RT {q(med)}")
```

---

## Part B — BayesFlow

### Cells 13–19 — REPLACE with one factory

**What's wrong**

*Cell 17:* `num_params = 4` is hardcoded with the comment `# 4 Parameter: v_scale, t0, a, lam`.
The count actually varies by model: 4 (fixed), 5 (linear, urgency), 6 (exponential),
7 (sigmoid, weibull). In BayesFlow 2 you don't pass `num_params` at all — it's inferred
from the adapter output.

*Cell 18:* four separate problems.

- `tf.stack` inside `stack_trials_for_deepset` while the Keras backend is JAX. Adapter
  transforms run on numpy; pulling in TensorFlow here is asking for a backend clash.
- `.concatenate(["cue", "has_deadline"], into="inference_conditions")` — these are
  per-trial arrays of length `n_trials`, so this builds a 400-dimensional "condition"
  vector out of what should be two scalars. Gone now that the cell is fixed.
- `.drop([... "k", "lam", "k_sig", "t_mid"])` discards the project's target quantity.
- That same `.drop` lists keys that don't exist for most models — for `model="fixed"`
  there is no `k`, and the adapter will raise.

Do the stacking in the likelihood (`pack_observations`) instead of the adapter. It's
plain numpy, it runs once per simulation, and there's no framework to clash with.

**Replacement**

```python
def make_adapter(model):
    return (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("obs", "summary_variables")
        .concatenate(param_names(model), into="inference_variables")
        .keep(["summary_variables", "inference_variables"])
    )


def make_workflow(model, checkpoint_dir=None, summary_dim=32):
    simulator = bf.make_simulator([make_prior(model), make_likelihood(model)])
    kwargs = {}
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        kwargs = dict(checkpoint_filepath=checkpoint_dir, checkpoint_name=f"ddm_{model}")
    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=make_adapter(model),
        inference_network=bf.networks.CouplingFlow(
            num_coupling_layers=6, subnet_kwargs={"widths": (128, 128)}
        ),
        summary_network=bf.networks.DeepSet(summary_dim=summary_dim),
        inference_variables=param_names(model),
        **kwargs,
    )
```

`DeepSet` rather than `SetTransformer`: with ~360 trials the transformer's attention
matrix is 360×360 per dataset for no clear benefit here, and it's what forced the
chunking hack in cell 31.

Keep a smoke test where cell 20 was:

```python
wf = make_workflow("sigmoid")
d = wf.simulate(4)
print({k: np.asarray(v).shape for k, v in d.items()})
```

---

### Cell 21 — REPLACE

`EPOCHS = 1` with 300 batches. That's 6000 simulated datasets total — nowhere near
convergence, and every diagnostic downstream is reading an untrained network.

```python
EPOCHS, BATCH_SIZE, N_BATCHES = 30, 64, 200

workflows, histories = {}, {}
for mdl in MODEL_NAMES:
    print(f"=== {mdl}  {param_names(mdl)} ===")
    wf = make_workflow(mdl, checkpoint_dir=f"checkpoints/{mdl}")
    histories[mdl] = wf.fit_online(epochs=EPOCHS, batch_size=BATCH_SIZE,
                                   num_batches_per_epoch=N_BATCHES)
    workflows[mdl] = wf

with open("networks/histories.pkl", "wb") as f:
    pickle.dump({k: v.history for k, v in histories.items()}, f)
```

### Cell 22 — REPLACE

Load per model:

```python
approximators = {
    mdl: keras.saving.load_model(f"checkpoints/{mdl}/ddm_{mdl}.keras")
    for mdl in MODEL_NAMES
}
```

### Cells 23–28 — KEEP, wrapped in a loop

```python
val, post = {}, {}
for mdl in MODEL_NAMES:
    val[mdl] = workflows[mdl].simulate(300)
    post[mdl] = workflows[mdl].approximator.sample(num_samples=500, conditions=val[mdl])
    w = 3 * len(param_names(mdl))
    bf.diagnostics.plots.recovery(post[mdl], val[mdl], figsize=(w, 3))
    plt.suptitle(f"recovery — {mdl}"); plt.show()
    bf.diagnostics.plots.calibration_ecdf(post[mdl], val[mdl], figsize=(w, 3))
    plt.suptitle(f"calibration — {mdl}"); plt.show()
```

**Stop here and read the output before continuing.** If `k` / `lam` / `k_sig` / `kappa` /
`gamma` doesn't recover on simulated data, nothing you conclude from the empirical fit is
warranted — the comparison would be measuring the network, not the boundary. If recovery
is weak, the fix is more trials per difficulty or more difficulty levels, not more epochs.

---

## Part C — Empirical data

### Cell 29 — REPLACE

**What's wrong**

1. `mapping_diff` keys are inconsistent: `"[0.55, 0.45]"` and `"[0.6, 0.4]"` have a space
   after the comma, `"[0.75,0.25]"` and `"[0.8,0.2]"` don't. If the actual strings in the
   csv are uniformly formatted, two of these six never match.
2. `.replace()` leaves unmatched values as strings, so `difficulty` silently becomes a
   mixed str/float column. `.to_numpy(dtype=np.float64)` in cell 30 then raises, or worse,
   coerces silently.
3. `empirical_data["deadline"]` is overwritten with the integer mapping *and* copied to
   `has_deadline`, so the original label is destroyed and you can't sanity-check the split.

Parse the numbers instead of matching on strings:

```python
emp = pd.read_csv("../data/Katsimposis/experiment-2-Flash.csv")

emp["difficulty"] = (
    emp["difficulty"].astype(str).str.strip()
    .str.replace(r"[\[\]]", "", regex=True)
    .str.split(",")
    .apply(lambda s: round(abs(float(s[0]) - float(s[1])), 2))
)
emp["has_deadline"] = (emp["deadline"].astype(str).str.strip().str.lower() == "deadline").astype(int)
emp["cue"] = (emp["cue"].astype(str).str.strip().str.upper() == "AC").astype(int)

assert emp["difficulty"].notna().all()
print(sorted(emp["difficulty"].unique()))
print(emp.groupby(["cue", "has_deadline"]).size())
```

Then **lock the design constants to the data** — the simulator has to match the data it
will be applied to, and this must run *before* cell 21 trains anything:

```python
cell = emp[(emp.cue == CUE) & (emp.has_deadline == HAS_DEADLINE)]

DEADLINE = float(np.nanpercentile(cell["rt"], 99.9))
MAX_TIME = DEADLINE
COHERENCES = np.array(sorted(cell["difficulty"].unique()))
N_TRIALS_PER_COH = int(round(cell.groupby("pp").size().median() / len(COHERENCES)))

print(f"deadline={DEADLINE:.3f}s | {len(COHERENCES)} coherences | "
      f"{N_TRIALS_PER_COH} trials/coherence | {cell.pp.nunique()} participants")
```

Because every function reads these as `None`-defaults resolved at call time, rebinding
them here propagates without re-running Part A.

---

### Cells 30 + 31 — DELETE

**Why cell 31 is the fatal one.** It slices 19,033 trials into arbitrary 400-trial blocks:

```python
for i in range(0, num_trials, chunk_size):
    rt_chunk = rt_array[i:i+chunk_size]
```

Each block spans multiple participants, both cues, both deadline conditions and all six
difficulties. There is no single parameter vector that generated a block, so the posterior
per chunk is estimating a quantity that doesn't exist. Pooling those posteriors and taking
`np.mean` compounds it.

Two further bugs in the same cell:

- `choice_array = df['acc']` includes `NaN` for missed deadlines, which goes straight into
  the network.
- `all_samples.append(chunk_samples[first_key])` — in BayesFlow 2, `sample()` returns a
  dict keyed *per variable*. `first_key` grabs one parameter only, not the concatenated
  vector, so `final_parameter_estimates` is the mean of a single parameter repeated.

**Replacement** — one dataset = one participant:

```python
def prepare_empirical(df, cue=CUE, has_deadline=HAS_DEADLINE):
    sub = df[(df["cue"] == cue) & (df["has_deadline"] == has_deadline)]
    per_pp, ids = [], []
    for pp, g in sub.groupby("pp"):
        acc = g["acc"].to_numpy(float)
        choice = np.where(np.isnan(acc), -1, acc).astype(int)   # NaN = missed deadline
        per_pp.append(pack_observations(g["rt"].to_numpy(float), choice,
                                        g["difficulty"].to_numpy(float)))
        ids.append(pp)
    n_min = min(a.shape[0] for a in per_pp)
    return {"obs": np.stack([a[:n_min] for a in per_pp]).astype(np.float32)}, ids, n_min


emp_batch, pp_ids, n_trials = prepare_empirical(emp)
print(emp_batch["obs"].shape, "->", len(pp_ids), "participants x", n_trials, "trials")

emp_post = {mdl: workflows[mdl].approximator.sample(num_samples=1000, conditions=emp_batch)
            for mdl in MODEL_NAMES}

for mdl in MODEL_NAMES:
    print(f"{mdl:12s}", {k: round(float(np.mean(emp_post[mdl][k])), 3)
                         for k in param_names(mdl)})
```

Make sure `n_trials` here is close to `len(COHERENCES) * N_TRIALS_PER_COH`. A large
mismatch between training and application trial counts degrades the summary network.

---

### Cells 32 + 33 + 34 — REPLACE

**What's wrong**

*Cell 32:* `resimulate_from_posterior` reads `posterior_draws[i, 0], [i, 1], [i, 2]` as
`v_scale, t0, a_base` and hardcodes `sv_fixed=0.175`. But it never passes any shape
parameter, so every resimulation uses whatever `apply_condition_effects` happens to draw
at random — the posterior predictive is not conditioned on the fitted boundary at all.
Positional indexing into the posterior is also fragile; index by name.

*Cell 34:* this crashes.

```python
ax.hist(emp_rt["has_deadline"] == 1, ...)
```

`emp_rt` is a numpy array from `df["rt"].to_numpy()`. String-indexing it raises
`IndexError: only integers, slices... are valid indices`. Same on the next three lines and
on `resim_rt_all`. You meant to mask by a separate condition array.

**Replacement** — posterior predictive comparison on statistics the boundary actually
controls: RT quantiles across difficulty, and the miss rate.

```python
QUANTILES = [.1, .3, .5, .7, .9]

def summary_stats(rt, correct, coh, coherences):
    out = []
    for c_ in coherences:
        sel = coh == c_
        r = rt[sel & ~np.isnan(rt)]
        out += list(np.quantile(r, QUANTILES)) if r.size > 5 else [np.nan] * len(QUANTILES)
        out.append(np.nanmean(correct[sel]))
    out.append(float(np.mean(np.isnan(rt))))
    return np.array(out)


obs_stats = np.nanmean(np.array([
    summary_stats(np.where(g.acc.notna(), g.rt, np.nan), g.acc.to_numpy(float),
                  g.difficulty.to_numpy(), COHERENCES)
    for _, g in cell.groupby("pp")
]), axis=0)

ppc = {}
for mdl in MODEL_NAMES:
    draws = []
    for i in range(len(pp_ids)):
        for j in rng.choice(1000, 20, replace=False):
            p = {k: float(np.asarray(emp_post[mdl][k])[i, j].ravel()[0])
                 for k in param_names(mdl)}          # by NAME, not position
            rt, ch, coh = simulate_dataset(mdl, p, rng=rng)
            draws.append(summary_stats(np.where(ch >= 0, rt, np.nan),
                                       np.where(ch == 1, 1.0, 0.0), coh, COHERENCES))
    ppc[mdl] = np.array(draws)
    err = np.nanmean(np.abs(np.nanmean(ppc[mdl], axis=0) - obs_stats))
    print(f"{mdl:12s} mean absolute discrepancy = {err:.4f}")
```

### Cell 35 — REPLACE

The deadline split no longer exists. Plot predicted vs. observed RT quantiles across
difficulty, one panel per model — this is the plot that discriminates boundary shapes:

```python
nq = len(QUANTILES)
fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(3.4 * len(MODEL_NAMES), 3.2), sharey=True)
for ax, mdl in zip(axes, MODEL_NAMES):
    pred = np.nanmean(ppc[mdl], axis=0)
    for qi in range(nq):
        idx = [ci * (nq + 1) + qi for ci in range(len(COHERENCES))]
        ax.plot(COHERENCES, obs_stats[idx], "o-", color="steelblue",
                label="empirical" if qi == 0 else None)
        ax.plot(COHERENCES, pred[idx], "s--", color="darkorange",
                label="predicted" if qi == 0 else None)
    ax.set_title(mdl); ax.set_xlabel("coherence")
axes[0].set_ylabel("RT quantile (s)"); axes[0].legend()
plt.tight_layout(); plt.show()

print(f"{'model':12s} predicted miss   (empirical = {obs_stats[-1]:.3f})")
for mdl in MODEL_NAMES:
    print(f"{mdl:12s} {np.nanmean(ppc[mdl][:, -1]):.3f}")
```

### Cell 36 — REPLACE

Your prior predictive check runs *after* the posterior fit and calls `prior()` /
`likelihood()` with the old signature. Move it up to cell 12 (see Part A) — its whole
purpose is to catch prior/data mismatch before you spend time training.

### Cells 37 + 38 — DELETE

`test_data_large = workflow.simulate(20)` with a comment saying the trial structure still
needs adjusting. Superseded: cell 29 now sets `N_TRIALS_PER_COH` from the data, so
training and application counts match by construction.

---

## Order of operations

Cell 29's design-constant block must run **before** cell 21 trains. Suggested final order:

1. Cells 0–7 (imports, boundaries, priors, simulator)
2. Cell 29 (load data, lock `DEADLINE` / `COHERENCES` / `N_TRIALS_PER_COH`)
3. Cell 12 (boundary shapes + prior predictive check)
4. Cells 9–19 (BayesFlow interface)
5. Cell 21 (train), 23–28 (**recovery — check before proceeding**)
6. Cells 30–35 (empirical posteriors, posterior predictive comparison)