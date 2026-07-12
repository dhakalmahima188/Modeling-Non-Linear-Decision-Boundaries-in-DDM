import numpy as np
 
 
def exponential_collapse(t: np.ndarray, a: float, lam: float, epsilon: float = 1e-4) -> np.ndarray:
    """
    Exponentially collapsing boundary: B(t) = a * exp(-lambda * t).
 
    Parameters
    ----------
    t : np.ndarray, shape (n_steps,)
        Time axis (seconds).
    a : float
        Initial boundary separation.
    lam : float
        Decay rate (lambda). Larger values collapse the boundary faster.
    epsilon : float, default=1e-4
        Minimum boundary value (floor), must be > 0.
 
    Returns
    -------
    np.ndarray, shape (n_steps,)
        Boundary value at each time step.
    """
    return np.maximum(a * np.exp(-lam * t), epsilon)
 
 
def sigmoid_collapse(t: np.ndarray, a: float, k_sig: float, t_mid: float) -> np.ndarray:
    """
    Sigmoid (logistic) collapsing boundary: B(t) = a * (1 - sigmoid(k_sig * (t - t_mid))).
 
    Parameters
    ----------
    t : np.ndarray, shape (n_steps,)
        Time axis (seconds).
    a : float
        Initial boundary separation.
    k_sig : float
        Steepness of the collapse (larger = sharper transition).
    t_mid : float
        Time (seconds) of the inflection point.
 
    Returns
    -------
    np.ndarray, shape (n_steps,)
        Boundary value at each time step.
    """
    sig = 1.0 / (1.0 + np.exp(-k_sig * (t - t_mid)))
    return np.maximum(a * (1.0 - sig), 1e-4)