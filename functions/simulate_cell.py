import numpy as np
from functions.boundary_functions import BOUNDARY_FNS
from constants import SHAPE_PRIOR_RANGES, FLASH_PAIRS
from functions.simulate_ddm_batch import simulate_ddm_batch
import itertools

# Simulate one experimental condition

def simulate_cell(
    params,
    model,
    coherences,
    n_trials_per_diff,
    dt,
    max_time,
    deadline_time=None,
    z=0.5,
    noise_sd=0.1,
    rng=None,
   
):
    """
    Simulate one participant under one experimental condition.

    The boundary is shared across all difficulty levels.
    Only the drift rate changes with difficulty.

    Parameters
    ----------
    params : dict
        Output of apply_condition_effects().

    model : str
        Boundary model.

    coherences : iterable
        Stimulus coherence values.

    Returns
    -------
    rts
    choices
    coherence_values
    difficulty_indices
    """

    if rng is None:
        rng = np.random.default_rng()



    time_axis = np.arange(int(max_time / dt)) * dt

    boundary_fn = BOUNDARY_FNS[model]

    a_base =  params["a_base"]      
    shape_kwargs = {n: params[n] for n in SHAPE_PRIOR_RANGES[model]}
    boundary = boundary_fn(time_axis, a=a_base, **shape_kwargs)

    deadline_step = None

    if deadline_time is not None:
        deadline_step = int(deadline_time / dt)


    
    all_rts = []
    all_choices = []
    all_coherences = []
    all_difficulties = []
    all_pairs = []

   
    pairs_to_run = list(zip(coherences, FLASH_PAIRS))
    start = 0                                 

    for difficulty_idx, (coherence, pair) in enumerate(pairs_to_run, start=start):

        drift = params["v_scale"] * coherence

        rts, choices = simulate_ddm_batch(
            v=drift,
            a=params["a_base"],
            t0=params["t0"],
            z=z,
            boundary=boundary,
            dt=dt,
            n_trials=n_trials_per_diff,
            deadline_step=deadline_step,
            noise_sd=noise_sd,
            rng=rng,
        )

        all_rts.append(rts)
        all_choices.append(choices)

        all_coherences.append(
            np.full(n_trials_per_diff, coherence)
        )
        all_pairs.append([list(pair)] * n_trials_per_diff)


        all_difficulties.append(
            np.full(
                n_trials_per_diff,
                difficulty_idx,
                dtype=int,
            )
        )

    return (
        np.concatenate(all_rts),
        np.concatenate(all_choices),
    np.concatenate(all_coherences),
        list(itertools.chain.from_iterable(all_pairs)),
        np.concatenate(all_difficulties),
    )