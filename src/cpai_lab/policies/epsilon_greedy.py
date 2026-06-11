"""Epsilon-greedy action-selection policies."""

import numpy as np


# ------------------------------------------------------------------
# epsilon-greedy
# ------------------------------------------------------------------
def epsilon_greedy_policy(Q, epsilon=0.1):
    """Epsilon-greedy policy with random tie-breaking.

    Returns a probability vector over actions.
    """
    n_actions = Q.shape[0]
    policy = epsilon / n_actions * np.ones(n_actions)
    max_val = np.max(Q)
    best = np.flatnonzero(Q == max_val)
    policy[best] += (1 - epsilon) / len(best)
    return policy


def naive_epsilon_greedy_policy(Q, epsilon=0.1):
    """Naive epsilon-greedy using ``np.argmax`` (biased tie-break).

    Provided for comparison / teaching purposes.
    """
    n_actions = Q.shape[0]
    policy = np.ones(n_actions) * (epsilon / n_actions)
    best_action = np.argmax(Q)
    policy[best_action] += 1.0 - epsilon
    return policy

def sample_action(Q_s, epsilon=0.1, rng=None):
    """Sample one action from the epsilon-greedy distribution.

    Parameters
    ----------
    Q_s : array_like, shape (n_actions,)
        Q-values for the current state.
    epsilon : float
        Exploration rate.
    rng : numpy.random.Generator or None
        Random number generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    probs = epsilon_greedy_policy(np.asarray(Q_s), epsilon)
    return int(rng.choice(len(probs), p=probs))
