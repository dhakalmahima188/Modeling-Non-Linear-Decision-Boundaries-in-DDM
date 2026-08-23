from functions.likelihood import likelihood
from functions.get_param_order import get_param_order
from constants import MODEL_NAMES
import numpy as np

def resimulate_from_posterior(posterior_draws, model="exponential",
                              difficulty_idx=0, n_resim=100, seed=None,
                              shape_kwargs=None):
    model = model or MODEL_NAMES[0]
    param_order = get_param_order(model)
    n_params = len(param_order)
 
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(posterior_draws),
                       size=min(n_resim, len(posterior_draws)), replace=False)
 
    out = []
    for i in idxs:
        values = dict(zip(param_order, posterior_draws[i, :n_params]))
        out.append(likelihood(model_idx=0, **values))
    return out
