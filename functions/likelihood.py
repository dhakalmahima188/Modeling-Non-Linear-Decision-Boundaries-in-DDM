import numpy as np
from constants import MODEL_NAMES, COHERENCES, DT, DEADLINE_TIME, MAX_TIME, N_TRIALS_PER_DIFF
from functions.simulate_cell import simulate_cell

def likelihood(model_idx, v_scale, t0,  a_base,
                lam, k_sig, t_mid):
    rng = np.random.default_rng()
    model = MODEL_NAMES[int(model_idx)]

    params = {"v_scale": v_scale, "t0": t0, 
              "a_base": a_base,
               "lam": lam, "k_sig": k_sig, "t_mid": t_mid}

    rts, choices, coh, _, _ = simulate_cell(
        params=params, model=model, coherences=COHERENCES,
        n_trials_per_diff=N_TRIALS_PER_DIFF, dt=DT, max_time=MAX_TIME,
        deadline_time=DEADLINE_TIME, rng=rng,
    )
    responded = (choices != -1).astype(np.float32)
    return {
        "rt":        np.where(choices == -1, -1.0, rts).astype(np.float32),
        "choice":    choices.astype(np.float32),
        "coherence": coh.astype(np.float32),
        "responded": responded
    }