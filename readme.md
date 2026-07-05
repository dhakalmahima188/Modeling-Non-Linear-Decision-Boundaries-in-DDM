# Collapsing Decision Boundaries

## Simulation Section : *Mahima & Bishesh*

Write a generative model/simulator
- Parameter sampling: draw parameters (v, a₀, t0, z, and collapse/urgency parameters like k, τ, onset) from prior distributions.
- Trial simulation: numerically simulate the diffusion process using discretized time steps (Euler-Maruyama method):
X(t+Δt) = X(t) + v·Δt + σ·√Δt·ε, ε ~ N(0,1)
At each step, compare X(t) to the time-varying boundary a(t) (or apply urgency-gated evidence); record the crossing time and which boundary was hit.
- Output: simulated (choice, RT) pairs — the "observed data" surrogate — matching the format of empirical data.
- Repeat across many parameter draws to build a large training set of (parameters, simulated data) pairs.
  



## Neural Network Section : *Esther & Paula*
