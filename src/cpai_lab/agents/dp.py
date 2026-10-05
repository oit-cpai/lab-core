"""動的計画法(価値反復・方策評価)。

遷移確率 ``P[s, a, s']`` と期待報酬 ``R[s, a]`` が分かっているときに、
最適な価値と方策を計算する。学習結果の検算(正解との比較)に使う。
``GridMazeEnv.transition_model()`` の返り値をそのまま渡せる。
"""

import numpy as np


def value_iteration(P, R, gamma=0.99, tol=1e-8, max_iter=100000):
    """価値反復で最適行動価値 Q*、状態価値 V*、貪欲方策を求める。

    Parameters
    ----------
    P : ndarray, shape (n_states, n_actions, n_states)
    R : ndarray, shape (n_states, n_actions)
    gamma : float
        割引率。
    tol : float
        価値の変化がこれより小さくなったら止める。

    Returns
    -------
    V : ndarray, shape (n_states,)
    Q : ndarray, shape (n_states, n_actions)
    policy : ndarray of int, shape (n_states,)
    """
    n_s, n_a, _ = P.shape
    V = np.zeros(n_s)
    for _ in range(max_iter):
        Q = R + gamma * P @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * P @ V
    return V, Q, Q.argmax(axis=1)


def policy_evaluation(P, R, policy, gamma=0.99, tol=1e-8, max_iter=100000):
    """決定的な方策 ``policy``(状態→行動)の状態価値を求める。"""
    n_s = P.shape[0]
    idx = np.arange(n_s)
    P_pi = P[idx, policy]          # (n_states, n_states)
    R_pi = R[idx, policy]          # (n_states,)
    V = np.zeros(n_s)
    for _ in range(max_iter):
        V_new = R_pi + gamma * P_pi @ V
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V
