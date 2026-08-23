from constants import MODEL_NAMES
from functions.resimulate_from_posterior import resimulate_from_posterior
import numpy as np
def _plot_rt_distribution(ax, empirical_df, posterior_draws, model=None,
                           n_resim=100, seed=None, pp_label=""):
    """Draws empirical vs. simulated RT distribution (correct/error) into a given ax."""
    model = model or MODEL_NAMES[0]
    sims = resimulate_from_posterior(posterior_draws, model=model,
                                      n_resim=n_resim, seed=seed)

    rt_c_emp = empirical_df.loc[empirical_df["acc"] == 1, "rt"]
    rt_e_emp = empirical_df.loc[empirical_df["acc"] == 0, "rt"]
    ax.hist(rt_c_emp, bins=40, density=True, alpha=0.35, color="tab:blue", label="emp. correct")
    ax.hist(rt_e_emp, bins=40, density=True, alpha=0.35, color="tab:red", label="emp. error")

    sim_rt_c = np.concatenate([s["rt"][s["choice"] == 1] for s in sims])
    sim_rt_e = np.concatenate([s["rt"][s["choice"] == 0] for s in sims])
    ax.hist(sim_rt_c, bins=40, density=True, histtype="step", lw=2,
            color="tab:blue", linestyle="--", label="sim. correct")
    ax.hist(sim_rt_e, bins=40, density=True, histtype="step", lw=2,
            color="tab:red", linestyle="--", label="sim. error")

    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title(pp_label)