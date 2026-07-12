"""
Drift Diffusion Model (DDM) simulation with collapsing decision boundaries.

Supports fixed, linear, exponential, and sigmoid boundary shapes, simulates
trial-level reaction times and choices via Euler-Maruyama integration, and
can be run from the command line to simulate, save, and plot results.

Results are always saved to ../data/simulated_data_<boundary>.csv.

Example
-------
python ddm_simulation.py --boundary exponential --v 1.0 --a 1.0 --t0 0.2 \
    --z 0.5 --n-trials 500 --lam 0.3 --seed 42 --plot
"""

import argparse
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import exponential_collapse, sigmoid_collapse


# Registry mapping boundary names to their functions, used by the CLI.
BOUNDARY_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "exponential": exponential_collapse,
    "sigmoid": sigmoid_collapse,
}

# Simulator
def simulate_ddm_trials(
    v: float,
    a: float,
    t0: float,
    z: float,
    boundary_fn: Callable[..., np.ndarray],
    boundary_params: dict,
    n_trials: int = 200,
    dt: float = 0.001,
    max_time: float = 3.0,
    deadline: Optional[float] = None,
    noise_sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate DDM trials via Euler-Maruyama integration of the diffusion process.

    Parameters
    ----------
    v : float
        Drift rate.
    a : float
        Boundary separation passed to ``boundary_fn`` (initial/base boundary).
    t0 : float
        Non-decision time (seconds), added to the simulated decision time.
    z : float
        Relative starting point in [0, 1]; 0.5 is unbiased, 0 / 1 start at
        the lower / upper boundary respectively.
    boundary_fn : Callable[..., np.ndarray]
        Function computing the boundary trajectory, called as
        ``boundary_fn(time_axis, a=a, **boundary_params)``.
    boundary_params : dict
        Extra keyword arguments passed to ``boundary_fn`` (e.g. ``lam``,
        ``k_sig``, ``t_mid``).
    n_trials : int, default=200
        Number of trials to simulate.
    dt : float, default=0.001
        Integration time step (seconds).
    max_time : float, default=3.0
        Maximum simulated time per trial (seconds).
    deadline : float, optional
        If set, trials are truncated (counted as non-response) once
        simulated time reaches this value (seconds).
    noise_sd : float, default=1.0
        Standard deviation of the diffusion noise.
    rng : np.random.Generator, optional
        Random number generator. A new default generator is created if not given.

    Returns
    -------
    rts : np.ndarray, shape (n_trials,)
        Reaction times in seconds; ``np.nan`` for non-responses.
    choices : np.ndarray, shape (n_trials,)
        Choice per trial: 1 = correct (upper boundary), 0 = error (lower
        boundary), -1 = non-response.
    """
    if rng is None:
        rng = np.random.default_rng()

    rts = np.full(n_trials, np.nan)
    choices = np.full(n_trials, -1, dtype=int)

    max_steps = int(max_time / dt)
    time_axis = np.arange(max_steps) * dt
    boundary = boundary_fn(time_axis, a=a, **boundary_params)

    for i in range(n_trials):
        # Starting point: z is relative [0,1]; convert to absolute position.
        x = (z - 0.5) * 2 * a  # z=0.5 -> 0, z=0 -> -a, z=1 -> +a

        for step in range(max_steps):
            t_current = step * dt

            # Check deadline.
            if deadline is not None and t_current >= deadline:
                break  # non-response

            b = boundary[step]

            # Check boundaries.
            if x >= b:
                rts[i] = t0 + t_current
                choices[i] = 1  # correct
                break
            elif x <= -b:
                rts[i] = t0 + t_current
                choices[i] = 0  # error
                break

            # Euler-Maruyama update.
            x += v * dt + noise_sd * np.sqrt(dt) * rng.standard_normal()

    return rts, choices

# Priors 
def sample_prior_exponential_boundary(n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample from the prior over DDM parameters for the exponential-collapse boundary.

    Parameters
    ----------
    n_samples : int
        Number of parameter draws to sample.
    rng : np.random.Generator, optional
        Random number generator. A new default generator is created if not given.

    Returns
    -------
    np.ndarray, shape (n_samples, 5)
        Columns are [v, a, t0, z, lam], where:
        v   ~ Normal(0, 2)        drift rate
        a   ~ HalfNormal(1.0)     boundary separation (> 0)
        t0  ~ Uniform(0.05, 0.5)  non-decision time
        z   ~ Uniform(0.3, 0.7)   starting point
        lam ~ HalfNormal(0.5)     exponential decay rate (> 0)
    """
    if rng is None:
        rng = np.random.default_rng()

    v = rng.normal(0, 2, n_samples)
    a = np.abs(rng.normal(0, 1.0, n_samples)) + 0.1
    t0 = rng.uniform(0.05, 0.5, n_samples)
    z = rng.uniform(0.3, 0.7, n_samples)
    lam = np.abs(rng.normal(0, 0.5, n_samples)) + 0.01

    return np.stack([v, a, t0, z, lam], axis=-1)


def sample_prior_sigmoid_boundary(n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample from the prior over DDM parameters for the sigmoid-collapse boundary.

    Parameters
    ----------
    n_samples : int
        Number of parameter draws to sample.
    rng : np.random.Generator, optional
        Random number generator. A new default generator is created if not given.

    Returns
    -------
    np.ndarray, shape (n_samples, 6)
        Columns are [v, a, t0, z, k_sig, t_mid], where:
        v     ~ Normal(0, 2)        drift rate
        a     ~ HalfNormal(1.0)     boundary separation (> 0)
        t0    ~ Uniform(0.05, 0.5)  non-decision time
        z     ~ Uniform(0.3, 0.7)   starting point
        k_sig ~ HalfNormal(5.0)     collapse steepness (> 0)
        t_mid ~ Uniform(0.3, 1.5)   inflection point (seconds)
    """
    if rng is None:
        rng = np.random.default_rng()

    v = rng.normal(0, 2, n_samples)
    a = np.abs(rng.normal(0, 1.0, n_samples)) + 0.1
    t0 = rng.uniform(0.05, 0.5, n_samples)
    z = rng.uniform(0.3, 0.7, n_samples)
    k_sig = np.abs(rng.normal(0, 5.0, n_samples)) + 0.5
    t_mid = rng.uniform(0.3, 1.5, n_samples)

    return np.stack([v, a, t0, z, k_sig, t_mid], axis=-1)


def prior_fn(batch_size: int, boundary_type: str = "exponential") -> dict[str, np.ndarray]:
    """
    Draw a batch of prior samples for the DDM.

    Parameters
    ----------
    batch_size : int
        Number of parameter draws to sample.
    boundary_type : {"exponential", "sigmoid"}, default="exponential"
        Which boundary's prior to sample from.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with key ``"prior_draws"`` mapping to an array of shape
        ``(batch_size, 5)`` for exponential or ``(batch_size, 6)`` for sigmoid.
    """
    if boundary_type == "exponential":
        samples = sample_prior_exponential_boundary(batch_size)
    elif boundary_type == "sigmoid":
        samples = sample_prior_sigmoid_boundary(batch_size)
    else:
        raise ValueError(f"Unknown boundary_type: {boundary_type}")
    return {"prior_draws": samples}


def simulator_fn(
    prior_draws: np.ndarray,
    boundary_type: str = "exponential",
    n_trials: int = 200,
    deadline: float = 2.0,
    seed: Optional[int] = None
) -> dict[str, np.ndarray]:
    """
    Simulate a batch of DDM datasets from prior parameter draws (for use with sbi).

    Parameters
    ----------
    prior_draws : np.ndarray, shape (batch_size, n_params)
        Parameter draws, as produced by ``sample_prior_exponential_boundary``
        (5 columns) or ``sample_prior_sigmoid_boundary`` (6 columns).
    boundary_type : {"exponential", "sigmoid"}, default="exponential"
        Which collapsing-boundary function to use.
    n_trials : int, default=200
        Number of trials to simulate per parameter draw.
    deadline : float, default=2.0
        Response deadline (seconds) per trial.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with key ``"sim_data"`` mapping to an array of shape
        ``(batch_size, n_trials, 2)``, where the last axis holds
        ``[rt, choice]`` per trial (``rt`` is 0.0 for non-responses).
    """
    boundary_fn = exponential_collapse if boundary_type == "exponential" else sigmoid_collapse
    batch_size = prior_draws.shape[0]
    rng = np.random.default_rng(seed=seed)

    all_data = []
    for i in range(batch_size):
        if boundary_type == "exponential":
            v, a, t0, z, lam = prior_draws[i]
            boundary_params = {"lam": lam}
        else:  # sigmoid
            v, a, t0, z, k_sig, t_mid = prior_draws[i]
            boundary_params = {"k_sig": k_sig, "t_mid": t_mid}

        rts, choices = simulate_ddm_trials(
            v=v, a=a, t0=t0, z=z,
            boundary_fn=boundary_fn,
            boundary_params=boundary_params,
            n_trials=n_trials,
            dt=0.001,
            max_time=deadline + 0.5,
            deadline=deadline,
            rng=rng,
        )
        rts_clean = np.where(np.isnan(rts), 0.0, rts)
        trial_data = np.stack([rts_clean, choices.astype(float)], axis=-1)
        all_data.append(trial_data)

    return {"sim_data": np.array(all_data)}


# Plotting
def plot_ddm_results(df: pd.DataFrame, plot_output_path: Path ,title_prefix: str = "") -> None:
    """
    Plot RT distributions, response-type counts, and defective CDFs for DDM results.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``"rt"`` (float, seconds) and ``"choice"``
        (1 = correct, 0 = error, -1 = non-response).
    title_prefix : str, default=""
        Prefix used in the figure title and printed summary lines.

    Returns
    -------
    None
        Saves the figure to `plot_output_path` and prints summary statistics.
    """
    correct_rts = df.loc[df["choice"] == 1, "rt"].dropna()
    error_rts = df.loc[df["choice"] == 0, "rt"].dropna()
    n_nonresp = (df["choice"] == -1).sum()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title_prefix)

    axes[0].hist(correct_rts, bins=30, alpha=0.6, label="Correct", color="steelblue")
    axes[0].hist(error_rts, bins=30, alpha=0.6, label="Error", color="firebrick")
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RT Distribution")
    axes[0].legend()

    counts = df["choice"].value_counts().reindex([1, 0, -1], fill_value=0)
    axes[1].bar(
        ["Correct", "Error", "Non-response"],
        counts.values,
        color=["steelblue", "firebrick", "gray"],
    )
    axes[1].set_ylabel("Count")
    axes[1].set_title("Response Type Counts")

    p_correct = len(correct_rts) / len(df)
    p_error = len(error_rts) / len(df)

    x_c = np.sort(correct_rts)
    y_c = np.arange(1, len(x_c) + 1) / len(x_c) * p_correct
    x_e = np.sort(error_rts)
    y_e = np.arange(1, len(x_e) + 1) / len(x_e) * p_error

    axes[2].plot(x_c, y_c, color="steelblue", label="Correct")
    axes[2].plot(x_e, y_e, color="firebrick", label="Error")
    axes[2].set_xlabel("RT (s)")
    axes[2].set_ylabel("Defective CDF")
    axes[2].set_title("Defective CDFs")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_output_path}")

    print(f"[{title_prefix}] Non-response rate: {n_nonresp / len(df):.2%}")
    print(f"[{title_prefix}] Accuracy (of responses): {p_correct / (p_correct + p_error):.2%}")

# CLI
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser for running a DDM simulation.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with core DDM parameters and boundary-specific options.
    """
    parser = argparse.ArgumentParser(
        description="Simulate Drift Diffusion Model (DDM) trials with a chosen "
        "boundary shape, save results to CSV, and optionally plot them."
    )

    parser.add_argument(
        "--boundary",
        choices=sorted(BOUNDARY_FUNCTIONS.keys()),
        default="exponential",
        help="Boundary shape to use (default: exponential).",
    )

    # Core DDM parameters.
    parser.add_argument("--v", type=float, default=1.0, help="Drift rate (default: 1.0).")
    parser.add_argument("--a", type=float, default=1.0, help="Initial boundary separation (default: 1.0).")
    parser.add_argument("--t0", type=float, default=0.2, help="Non-decision time in seconds (default: 0.2).")
    parser.add_argument("--z", type=float, default=0.5, help="Relative starting point in [0,1] (default: 0.5).")
    parser.add_argument("--n-trials", type=int, default=500, help="Number of trials to simulate (default: 500).")
    parser.add_argument("--dt", type=float, default=0.001, help="Integration time step in seconds (default: 0.001).")
    parser.add_argument("--max-time", type=float, default=3.0, help="Maximum simulated time per trial (default: 3.0).")
    parser.add_argument("--deadline", type=float, default=2.0, help="Response deadline in seconds (default: 2.0).")
    parser.add_argument("--noise-sd", type=float, default=1.0, help="Diffusion noise standard deviation (default: 1.0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    # Boundary-specific parameters.
    parser.add_argument("--epsilon", type=float, default=1e-4, help="[exponential] Minimum boundary value (default: 1e-4).")
    parser.add_argument("--lam", type=float, default=0.3, help="[exponential] Decay rate lambda (default: 0.3).")
    parser.add_argument("--k-sig", type=float, default=5.0, help="[sigmoid] Collapse steepness (default: 5.0).")
    parser.add_argument("--t-mid", type=float, default=0.8, help="[sigmoid] Inflection time point (default: 0.8).")

    # Output options.
    parser.add_argument("--plot", action="store_true", help="Show summary plots after simulating.")

    return parser


def boundary_params_from_args(args: argparse.Namespace) -> dict:
    """
    Extract the boundary-specific keyword arguments for the chosen boundary type.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (as returned by ``build_arg_parser().parse_args()``).

    Returns
    -------
    dict
        Keyword arguments to pass to the selected boundary function
        (excluding ``t`` and ``a``, which are supplied separately).
    """
    if args.boundary == "fixed":
        return {}
    if args.boundary == "linear":
        return {"k": args.k, "epsilon": args.epsilon}
    if args.boundary == "exponential":
        return {"lam": args.lam, "epsilon": args.epsilon}
    if args.boundary == "sigmoid":
        return {"k_sig": args.k_sig, "t_mid": args.t_mid}
    raise ValueError(f"Unknown boundary type: {args.boundary}")


def main() -> None:
    """
    Parse command-line arguments, run a DDM simulation, save results to CSV,
    and optionally display summary plots.

    Returns
    -------
    None
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    boundary_fn = BOUNDARY_FUNCTIONS[args.boundary]
    boundary_params = boundary_params_from_args(args)

    rts, choices = simulate_ddm_trials(
        v=args.v,
        a=args.a,
        t0=args.t0,
        z=args.z,
        boundary_fn=boundary_fn,
        boundary_params=boundary_params,
        n_trials=args.n_trials,
        dt=args.dt,
        max_time=args.max_time,
        deadline=args.deadline,
        noise_sd=args.noise_sd,
        rng=rng,
    )

    df = pd.DataFrame({"rt": rts, "choice": choices})

    file_output_path = Path(f"../output/data/simulated_data_{args.boundary}.csv")
    file_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_output_path, index=False)
    print(f"Saved {args.n_trials} trials to {file_output_path}")

    if args.plot:
        plot_output_path = Path(f"../output/plots/simulated_data_{args.boundary}.png")
        plot_output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_ddm_results(df, plot_output_path=plot_output_path, title_prefix=f"{args.boundary.capitalize()} collapse")



if __name__ == "__main__":
    main()