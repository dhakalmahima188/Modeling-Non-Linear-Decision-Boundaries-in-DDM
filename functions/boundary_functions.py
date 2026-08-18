import numpy as np

# Boundary functions
def fixed_boundary(t, a):
    """Constant decision boundary."""
    return np.full_like(t, a)


def linear_collapse(t, a, k, epsilon=1e-4):
    """Linearly collapsing boundary."""
    return np.maximum(a - k * t, epsilon)


def exponential_collapse(t, a, lam, epsilon=1e-4):
    """Exponentially collapsing boundary."""
    return np.maximum(a * np.exp(-lam * t), epsilon)


def sigmoid_collapse(t, a, k_sig, t_mid, epsilon=1e-4):
    """Sigmoid collapsing boundary."""
    sig = 1.0 / (1.0 + np.exp(-k_sig * (t - t_mid)))
    return np.maximum(a * (1.0 - sig), epsilon)

BOUNDARY_FNS = {
    "fixed": fixed_boundary,
    "linear": linear_collapse,
    "exponential": exponential_collapse,
    "sigmoid": sigmoid_collapse,
}