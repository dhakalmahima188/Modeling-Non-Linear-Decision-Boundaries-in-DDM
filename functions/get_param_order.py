from constants import BASE_PARAMS, SHAPE_PRIOR_RANGES
def get_param_order(model):
    """
    Returns order of parameter columns for a given model:
    First base parameters, then model-specific shape parameters from SHAPE_PRIOR_RANGES.
    """
    return BASE_PARAMS + tuple(SHAPE_PRIOR_RANGES[model])