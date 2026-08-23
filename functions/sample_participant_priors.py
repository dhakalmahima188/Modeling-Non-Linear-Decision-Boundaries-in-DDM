from scipy.stats import truncnorm
import numpy as np


def truncnorm_positive(loc, scale, rng, lower=0.0, upper=np.inf):
    """Echte Grenzen (nicht die von scipy intern verwendeten standardisierten)."""
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, random_state=rng)


def sample_participant_priors(rng):
    p = {
        "v_scale": truncnorm_positive(loc=1.25, scale=0.75, rng=rng),
        "t0":      truncnorm_positive(loc=0.19, scale=0.138, rng=rng),
        "a_base":  rng.uniform(0, 1),
        "lam":     truncnorm_positive(loc=1.5, scale=1.0, rng=rng, upper=4.0),
        "k_sig":   truncnorm_positive(loc=1.5, scale=1.0, rng=rng, upper=3.5),
        "t_mid":   truncnorm_positive(loc=0.6, scale=0.4, rng=rng, upper=1.2),
    }
    return p