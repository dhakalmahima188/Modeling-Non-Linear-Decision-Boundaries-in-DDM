from functions.get_param_order import get_param_order
from constants import MODEL_NAMES
import pandas as pd
import numpy as np
def summarize_posterior(samples, model=None, ci=0.95):
    """
    samples: dict like returned by approximator.sample() e.g. {"v_scale": array(1, n_draws), "t0": ..., ...}
    returns da dataframe with one row per parameter
    """
    model = model or MODEL_NAMES[0]
    param_order = get_param_order(model)

    alpha = 1 - ci
    lower_q, upper_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    rows = []
    for p in param_order:
        draws = np.asarray(samples[p]).reshape(-1)
        rows.append({
            "param": p,
            "mean": draws.mean(),
            "median": np.median(draws),
            "sd": draws.std(),
            f"ci_{lower_q:.1f}": np.percentile(draws, lower_q),
            f"ci_{upper_q:.1f}": np.percentile(draws, upper_q),
        })
    return pd.DataFrame(rows)


def summarize_all_participants(results, model=None, ci=0.95):
    """
    results: Dict {pp_id: {"posterior": samples_dict, ...}}
    returns dataframe with one parameter and pp per row
    """
    summaries = []
    for pp_id, res in results.items():
        df = summarize_posterior(res["posterior"], model=model, ci=ci)
        df.insert(0, "pp", pp_id)
        summaries.append(df)
    return pd.concat(summaries, ignore_index=True)