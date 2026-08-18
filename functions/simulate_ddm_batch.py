import numpy as np

# Trial-level DDM simulation

def simulate_ddm_batch(
    v,
    a,
    t0,
    z,
    boundary,
    dt,
    n_trials,
    deadline_step,
    noise_sd,
    rng,
):
    """
    Simulate multiple DDM trials using Euler-Maruyama integration.

    Parameters
    ----------
    v : float
        Mean drift rate.

    a : float
        Initial boundary separation.

    t0 : float
        Non-decision time.

    z : float
        Relative starting point (0.5 = unbiased).


    boundary : ndarray
        Boundary height at every time step.

    Returns
    -------
    rts : ndarray
        Reaction times (NaN = non-response)

    choices : ndarray
        1 = upper boundary
        0 = lower boundary
       -1 = deadline reached
    """

    # Trial-specific drift rates
    
    v_trial = np.broadcast_to(np.asarray(v, dtype=float), (n_trials,)).copy()

    # Starting point
    x = np.full(
        n_trials,
        (z - 0.5) * 2 * a,
        dtype=float,
    )

    rts = np.full(n_trials, np.nan)
    choices = np.full(n_trials, -1, dtype=int)

    active = np.ones(n_trials, dtype=bool)

    max_steps = len(boundary)

    if deadline_step is not None:
        max_steps = min(max_steps, deadline_step)

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

        if active.any():
            x[active] += (
                v_trial[active] * dt
                + noise_sd
                * np.sqrt(dt)
                * rng.standard_normal(active.sum())
            )

    return rts, choices