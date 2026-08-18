from functions.sample_participant_priors import sample_participant_priors
from functions.simulate_cell import simulate_cell
from constants import COHERENCES, DT, DEADLINE_TIME, MAX_TIME, CUE, HAS_DEADLINE
import numpy as np
import pandas as pd


def simulate_full_dataset(model="exponential", n_participants=24,
                          coherences=COHERENCES, n_trials_per_cell=400,
                          dt=DT, deadline_time=DEADLINE_TIME,
                          max_time=MAX_TIME, noise_sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rows, true_params = [], []

    for pp in range(1, n_participants + 1):
        params = sample_participant_priors(rng)
        rts, choices, cohs, pairs, diffs = simulate_cell(
            params=params, model=model, coherences=coherences,
            n_trials_per_diff=n_trials_per_cell, dt=dt, max_time=max_time,
            deadline_time=deadline_time, noise_sd=noise_sd, rng=rng,
        )

        for rt, ch, co, pr, d in zip(rts, choices, cohs, pairs, diffs):
            rows.append({
                "pp": pp, "cue": CUE, "has_deadline": HAS_DEADLINE,
                "coherence": co, "pair": pr, "difficulty": d,
                "rt": rt,
                "acc": np.nan if ch == -1 else int(ch == 1),
                "nonresp_reason": "hit_deadline" if ch == -1 else None,
            })

        true_params.append({"pp": pp, "cue": CUE,
                            "has_deadline": HAS_DEADLINE,
                            "model": model, **params})

    return pd.DataFrame(rows), pd.DataFrame(true_params)

