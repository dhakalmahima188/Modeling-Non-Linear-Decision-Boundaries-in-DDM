from scipy.stats import truncnorm
import numpy as np

# define function for generating priors
def sample_participant_priors(rng):
    """
    Sample participant-specific latent parameters.

    These are assumed to remain constant across all experimental
    conditions for a participant.
    """

    p = {
        "v_scale": truncnorm.rvs(0, np.inf, loc=0.04, scale=0.03, random_state=rng),
        # "t0": rng.uniform(0, 0.40),
        "t0": truncnorm.rvs(0, np.inf, loc=0.1, scale=0.1, random_state=rng),
        # "sv": truncnorm.rvs(0, np.inf, loc=0.10, scale=0.05, random_state=rng), # maybbe  we dont require sv at all.
        "a_base": rng.uniform(0, 1),
        "lam": truncnorm.rvs(0, np.inf, loc=1.5, scale=1, random_state=rng),
        "k_sig": truncnorm.rvs(0, np.inf, loc=1.5, scale=1, random_state=rng),
        "t_mid": truncnorm.rvs(0, np.inf, loc=0.5, scale=0.25, random_state=rng),
    }
    return p
