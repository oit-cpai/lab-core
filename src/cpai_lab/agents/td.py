"""Tabular temporal-difference (TD) learning: SARSA, Q-learning, Expected SARSA.

All three algorithms share the same update rule

    Q(s, a) <- Q(s, a) + alpha * delta

and differ only in how the TD error ``delta`` is computed.

The environment is expected to follow the gymnasium API:
``reset()`` returns ``(obs, info)`` and ``step(action)`` returns
``(obs, reward, terminated, truncated, info)``, with discrete
``observation_space`` / ``action_space`` (e.g. ``CliffWalking-v0``).
"""

import numpy as np

from ..policies import epsilon_greedy_policy, sample_action

TD_METHODS = ("sarsa", "qlearning", "expected_sarsa")


def compute_td_error(method, Q, s, a, r, s_next, a_next,
                     terminated, epsilon=0.1, gamma=1.0):
    """Compute the TD error for the given method.

    Parameters
    ----------
    method : str
        One of ``'sarsa'``, ``'qlearning'``, ``'expected_sarsa'``.
    Q : ndarray, shape (n_states, n_actions)
    s, a : int
        Current state and action.
    r : float
        Reward received.
    s_next, a_next : int
        Next state and next action (``a_next`` used by SARSA only).
    terminated : bool
        Whether ``s_next`` is terminal.
    epsilon : float
        Exploration rate (used by Expected SARSA).
    gamma : float
        Discount factor.

    Returns
    -------
    td_error : float
    """
    if terminated:
        td_error = r - Q[s, a]
    elif method == "sarsa":
        td_error = r + gamma * Q[s_next, a_next] - Q[s, a]
    elif method == "qlearning":
        td_error = r + gamma * np.max(Q[s_next, :]) - Q[s, a]
    elif method == "expected_sarsa":
        pi = epsilon_greedy_policy(Q[s_next, :], epsilon)
        expected_q = np.dot(pi, Q[s_next, :])
        td_error = r + gamma * expected_q - Q[s, a]
    else:
        raise ValueError(f"Unknown method: {method}")
    return td_error


def train_td(env, method, alpha=0.5, epsilon=0.1, gamma=1.0,
             num_episodes=500, seed=None):
    """Train a tabular agent on a gymnasium-style environment.

    Parameters
    ----------
    env : gymnasium.Env
        Environment with discrete observation/action spaces.
    method : str
        ``'sarsa'``, ``'qlearning'``, or ``'expected_sarsa'``.
    alpha : float
        Learning rate.
    epsilon : float
        Exploration rate for epsilon-greedy.
    gamma : float
        Discount factor.
    num_episodes : int
        Number of training episodes.
    seed : int or None
        Random seed (environment and action sampling).

    Returns
    -------
    Q : ndarray, shape (n_states, n_actions)
    rewards : ndarray, shape (num_episodes,)
        Total reward per episode.
    """
    if method not in TD_METHODS:
        raise ValueError(f"Unknown method: {method} (choose from {TD_METHODS})")

    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = np.zeros(num_episodes)

    for ep in range(num_episodes):
        s, _ = env.reset(seed=seed + ep if seed is not None else None)
        a = sample_action(Q[s, :], epsilon, rng=rng)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            s_next, r, terminated, truncated, _ = env.step(a)
            a_next = sample_action(Q[s_next, :], epsilon, rng=rng)

            td_error = compute_td_error(
                method, Q, s, a, r, s_next, a_next,
                terminated, epsilon, gamma,
            )
            Q[s, a] += alpha * td_error

            s, a = s_next, a_next
            rewards[ep] += r

    return Q, rewards
