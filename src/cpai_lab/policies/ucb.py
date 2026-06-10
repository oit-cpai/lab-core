"""UCB action selection."""

import numpy as np

from ..utils.misc import argmax_random_tie


# ------------------------------------------------------------------
# UCB
# ------------------------------------------------------------------
def select_ucb_action(Q_values, N_values, t, c=2.0, rng=None):
    """Select an action using the UCB1 rule.

    Parameters
    ----------
    Q_values : ndarray
        Current action-value estimates.
    N_values : ndarray
        Per-action visit counts.
    t : int
        Current total step count (>= 1).
    c : float
        Exploration parameter.
    rng : numpy.random.Generator or None
        For tie-breaking.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_actions = Q_values.shape[0]

    untried = np.where(N_values == 0)[0]
    if len(untried) > 0:
        return int(rng.choice(untried))

    ucb_scores = np.zeros(n_actions)
    for a in range(n_actions):
        ucb_scores[a] = Q_values[a] + c * np.sqrt(np.log(t) / N_values[a])

    return argmax_random_tie(ucb_scores, rng=rng)
