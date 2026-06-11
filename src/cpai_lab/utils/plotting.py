"""Shared plotting helpers for bandit notebooks."""

import numpy as np
import matplotlib.pyplot as plt

from ..policies import epsilon_greedy_policy, naive_epsilon_greedy_policy
from ..policies import boltzmann_policy


def plot_epsilon_greedy_policy(Q, epsilon=0.1):
    """Visualise Q-values alongside the epsilon-greedy distribution."""
    policy = epsilon_greedy_policy(Q, epsilon)
    action = np.arange(Q.shape[0])

    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr[0].bar(action, Q)
    axarr[0].set_xlabel("action a")
    axarr[0].set_ylabel("action value Q(a)")
    axarr[1].bar(action, policy,
                 label=r"$\epsilon$ = %3.1f" % epsilon)
    axarr[1].set_xlabel("action a")
    axarr[1].set_ylabel(r"policy $\pi$ (a)")
    plt.legend()
    plt.show()


def plot_boltzmann_policy(Q, beta=1.0):
    """Visualise Q-values alongside the Boltzmann distribution."""
    policy = boltzmann_policy(Q, beta)
    action = np.arange(Q.shape[0])

    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    axarr[0].bar(action, Q)
    axarr[0].set_xlabel("action a")
    axarr[0].set_ylabel("action value Q(a)")
    axarr[1].bar(action, policy,
                 label=r"$\beta$ = %3.1f" % beta)
    axarr[1].set_xlabel("action a")
    axarr[1].set_ylabel(r"policy $\pi$ (a)")
    plt.legend()
    plt.show()


def compare_policies(Q, epsilon=0.1):
    """Compare random-tie-break vs naive epsilon-greedy side by side."""
    p_random = epsilon_greedy_policy(Q, epsilon)
    p_naive = naive_epsilon_greedy_policy(Q, epsilon)
    action = np.arange(Q.shape[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(action - width / 2, p_random, width, label="Random Tie-break")
    ax.bar(action + width / 2, p_naive, width,
           label="Naive (Smallest Index)")
    ax.set_xlabel("Action a")
    ax.set_ylabel("Probability")
    ax.set_title(f"Policy Comparison (epsilon={epsilon})")
    ax.set_xticks(action)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# ------------------------------------------------------------------
# generic learning-curve helpers (bandit / TD common)
# ------------------------------------------------------------------
def smooth(x, window=10):
    """Moving average with a sliding window."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def plot_learning_curves(results, window=10, ylim=None,
                         title='Learning Curves'):
    """Plot smoothed learning curves with mean +/- std shading.

    Parameters
    ----------
    results : dict
        ``{algorithm_name: ndarray of shape (num_runs, num_episodes)}``
    window : int
        Moving-average window size.
    ylim : tuple or None
        y-axis limits, e.g. ``(-200, 0)``.
    title : str
        Plot title.
    """
    colors = {'SARSA': 'tab:blue', 'Q-learning': 'tab:red',
              'Expected SARSA': 'tab:green'}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, data) in enumerate(results.items()):
        data = np.atleast_2d(np.asarray(data))
        smoothed = np.array([smooth(row, window) for row in data])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(len(mean))
        color = colors.get(name, f'C{i}')
        ax.plot(x, mean, label=name, color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Sum of rewards per episode')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# CliffWalking visualization
# ------------------------------------------------------------------
def plot_cliff_policy(Q, title='Learned Policy'):
    """Visualize the greedy policy as arrows on the CliffWalking 4x12 grid.

    Parameters
    ----------
    Q : ndarray, shape (48, 4)
        Learned Q-values.  Actions: 0=Up, 1=Right, 2=Down, 3=Left.
    title : str
        Plot title.
    """
    nrows, ncols = 4, 12
    # action -> (dy, dx) in plot coordinates (y-axis is inverted)
    arrow_map = {0: (-0.35, 0), 1: (0, 0.35), 2: (0.35, 0), 3: (0, -0.35)}

    fig, ax = plt.subplots(figsize=(12, 4))

    for s in range(nrows * ncols):
        row, col = divmod(s, ncols)

        # cliff cells
        if row == 3 and 1 <= col <= 10:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       color='#ffcccc'))
            ax.text(col, row, 'cliff', ha='center', va='center',
                    fontsize=6, color='#cc0000')
            continue

        # start / goal
        if (row, col) == (3, 0):
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       color='#cce5ff'))
            ax.text(col, row, 'S', ha='center', va='center',
                    fontsize=10, fontweight='bold')
            # still draw arrow
        if (row, col) == (3, 11):
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       color='#ccffcc'))
            ax.text(col, row, 'G', ha='center', va='center',
                    fontsize=10, fontweight='bold')
            continue

        best_a = np.argmax(Q[s, :])
        dy, dx = arrow_map[best_a]
        ax.annotate('', xy=(col + dx, row + dy),
                    xytext=(col, row),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # grid lines
    for r in range(nrows + 1):
        ax.axhline(r - 0.5, color='gray', linewidth=0.5)
    for c in range(ncols + 1):
        ax.axvline(c - 0.5, color='gray', linewidth=0.5)

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    plt.show()
