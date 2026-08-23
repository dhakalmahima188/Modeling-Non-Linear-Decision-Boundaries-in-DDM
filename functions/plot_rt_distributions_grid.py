from constants import MODEL_NAMES
from functions.samples_dict_to_array import samples_dict_to_array
from _plot_rt_distribution import _plot_rt_distribution
import numpy as np
import matplotlib.pyplot as plt

def plot_rt_distributions_grid(empirical_data, posterior_samples_per_pp,
                                model=None, n_resim=100, seed=None,
                                n_cols=6, space_legend=1.06, space_figure=0.93):
    model = model or MODEL_NAMES[0]
    pp_ids = list(posterior_samples_per_pp.keys())
    n_pp = len(pp_ids)
    n_rows = int(np.ceil(n_pp / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows),
                              sharex=True, sharey=False)
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for i, pp_id in enumerate(pp_ids):
        ax = axes[i // n_cols, i % n_cols]
        emp = empirical_data[empirical_data["pp"] == pp_id]
        draws = samples_dict_to_array(posterior_samples_per_pp[pp_id], model=model)
        _plot_rt_distribution(ax, emp, draws, model=model, n_resim=n_resim,
                               seed=seed, pp_label=f"Participant {pp_id}")

    # leere Subplots ausblenden, falls n_pp nicht exakt n_rows*n_cols füllt
    for j in range(n_pp, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    # eine gemeinsame Legende statt 24x dieselbe
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=12,
            bbox_to_anchor=(0.5, space_legend))

    plt.tight_layout(rect=[0, 0, 1, space_figure])
    return fig