from functions.likelihood import likelihood
from constants import MODEL_NAMES
import numpy as np

def resimulate_from_posterior(posterior_draws, model="exponential",
                              difficulty_idx=0, n_resim=100, seed=None,
                              shape_kwargs=None):
    rng = np.random.default_rng(seed)
    shape_kwargs = shape_kwargs or {"lam": 2.5, "k_sig": 5.0, "t_mid": 0.8}

    idxs = rng.choice(len(posterior_draws),
                    size=min(n_resim, len(posterior_draws)), replace=False)
    out = []
    for i in idxs:
        v_scale_i, t0_i, a_base_i, lam_i = posterior_draws[i, :4]
        out.append(likelihood(
            model_idx=MODEL_NAMES.index(model),
            v_scale=v_scale_i, t0=t0_i, a_base=a_base_i,
            **shape_kwargs
        ))
    return out
