import numpy as np
from functions.sample_participant_priors import sample_participant_priors
from constants import MODEL_NAMES

def prior(rng=None):

    """
    BayesFlow prior.

    Samples ONE participant together with a boundary model.

    The experimental manipulations (cue and deadline)
    are deterministic and are applied later inside the
    likelihood.
    """

    if rng is None:
        rng = np.random.default_rng()

    model_idx = rng.integers(len(MODEL_NAMES))

    participant = sample_participant_priors(rng)

    return {
        "model_idx": model_idx,
        **participant,
    }