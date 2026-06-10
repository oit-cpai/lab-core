"""Boltzmann (softmax) action-selection policy."""

import numpy as np


# ------------------------------------------------------------------
# Boltzmann (softmax)
# ------------------------------------------------------------------
def boltzmann_policy(Q, beta=1.0):
    """Boltzmann (softmax) policy.

    Parameters
    ----------
    Q : ndarray
        Action-value estimates (1-D).
    beta : float
        Inverse temperature (higher = more greedy).
    """
    Qmax = np.max(Q)
    expQ = np.exp(beta * (Q - Qmax))
    return expQ / np.sum(expQ)


