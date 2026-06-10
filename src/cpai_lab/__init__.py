"""cpai_lab — CPAI研究室の共通Pythonライブラリ。

Convenience imports so that ``from cpai_lab import GaussianBandit`` works.
"""

__version__ = "0.1.0"

from .envs import GaussianBandit
from .policies import (
    epsilon_greedy_policy,
    naive_epsilon_greedy_policy,
    boltzmann_policy,
    select_ucb_action,
)
from .agents import (
    uniform_random_method,
    value_based_method,
    ucb_method,
    run_experiment,
)
from .utils import (
    argmax_random_tie,
    plot_epsilon_greedy_policy,
    plot_boltzmann_policy,
    compare_policies,
)

__all__ = [
    "GaussianBandit",
    "argmax_random_tie",
    "epsilon_greedy_policy",
    "naive_epsilon_greedy_policy",
    "boltzmann_policy",
    "select_ucb_action",
    "uniform_random_method",
    "value_based_method",
    "ucb_method",
    "run_experiment",
    "plot_epsilon_greedy_policy",
    "plot_boltzmann_policy",
    "compare_policies",
]
