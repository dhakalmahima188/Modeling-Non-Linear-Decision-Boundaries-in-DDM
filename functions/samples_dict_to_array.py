import numpy as np
from constants import MODEL_NAMES
from functions.resimulate_from_posterior import get_param_order


def samples_dict_to_array(samples, model=None):
    """converts bayesflow posterior dict into  (n_draws, n_params)-Array"""
    model = model or MODEL_NAMES[0]
    param_order = get_param_order(model)
    return np.stack([np.asarray(samples[p]).reshape(-1) for p in param_order], axis=1)