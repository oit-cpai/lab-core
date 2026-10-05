"""卒研用の共通部品(迷路環境・遅れラッパー・動的計画法・保存処理)のテスト。"""

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from cpai_lab import (
    GridMazeEnv, generate_maze, maze_stats, ActionDelay,
    value_iteration, policy_evaluation, ExperimentRun, train_td,
)
from cpai_lab.envs import bfs_distances, shortest_path, grid_from_text


def test_maze_is_connected_and_reproducible():
    g1 = generate_maze(6, 8, straightness=0.5, loop_prob=0.1, seed=3)
    g2 = generate_maze(6, 8, straightness=0.5, loop_prob=0.1, seed=3)
    assert g1.shape == (13, 17) and (g1 == g2).all()
    dist = bfs_distances(g1, (1, 1))
    assert (dist[~g1] >= 0).all()              # 全ての通路に到達できる


def test_straightness_makes_longer_corridors():
    def mean_run(s):
        return np.mean([maze_stats(generate_maze(10, 10, straightness=s, seed=k))["mean_straight_length"]
                        for k in range(10)])
    assert mean_run(0.9) > mean_run(0.0)


def test_loops_reduce_dead_ends():
    def dead(p):
        return np.mean([maze_stats(generate_maze(10, 10, loop_prob=p, seed=k))["n_dead_ends"]
                        for k in range(10)])
    assert dead(0.3) < dead(0.0)


def test_env_api_and_optimal_steps():
    env = GridMazeEnv(maze_kwargs=dict(n_rows=4, n_cols=4, seed=1))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    path = shortest_path(env.grid, env.start, env.goal)
    moves = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    for a, b in zip(path[:-1], path[1:]):
        obs, r, term, trunc, _ = env.step(moves[(b[0] - a[0], b[1] - a[1])])
    assert term and env.t == env.optimal_steps()


def test_text_map():
    grid, s, g, traps = grid_from_text("""
        #####
        #S..#
        ###.#
        #G..#
        #####""")
    env = GridMazeEnv(grid=grid, start=s, goal=g)
    assert env.optimal_steps() == 6


def test_value_iteration_matches_bfs_when_deterministic():
    env = GridMazeEnv(maze_kwargs=dict(n_rows=4, n_cols=5, seed=2), step_reward=-1.0, goal_reward=-1.0)   # 1ステップごとに -1 → 価値 = -最短ステップ数
    P, R = env.transition_model()
    assert np.allclose(P.sum(axis=2), 1.0)
    V, Q, pi = value_iteration(P, R, gamma=1.0)
    s0 = env.pos_to_obs(env.start)
    assert np.isclose(-V[s0], env.optimal_steps())
    assert np.allclose(policy_evaluation(P, R, pi, gamma=1.0)[s0], V[s0])


def test_slip_transition_model_matches_simulation():
    env = GridMazeEnv(maze_kwargs=dict(n_rows=3, n_cols=3, seed=0, loop_prob=0.3), slip_prob=0.4)
    P, _ = env.transition_model()
    env.reset(seed=0)
    s, a, n = env.pos_to_obs(env.start), 3, 20000
    counts = np.zeros(P.shape[0])
    for _ in range(n):
        env.pos = env.start
        s2, *_ = env.step(a)
        counts[s2] += 1
    assert np.abs(counts / n - P[s, a]).max() < 0.02


def test_qlearning_reaches_optimum_on_small_maze():
    env = GridMazeEnv(maze_kwargs=dict(n_rows=3, n_cols=3, seed=0))
    Q, rewards = train_td(env, "qlearning", alpha=0.5, epsilon=0.1, gamma=0.99, num_episodes=300, seed=0)
    env.reset(seed=0)
    s, steps, done = env.pos_to_obs(env.start), 0, False
    while not done and steps < 100:
        s, r, term, trunc, _ = env.step(int(Q[s].argmax()))
        steps, done = steps + 1, term or trunc
    assert steps == env.optimal_steps()


def test_action_delay():
    base = gym.make("Pendulum-v1")
    env = ActionDelay(base, delay=2, augment_obs=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,) and env.observation_space.contains(obs)
    acts = [np.array([1.0], dtype=np.float32), np.array([-1.0], dtype=np.float32), np.array([0.5], dtype=np.float32)]
    applied = []
    for a in acts:
        obs, r, te, tr, info = env.step(a)
        applied.append(float(np.asarray(info["applied_action"]).ravel()[0]))
    assert applied == [0.0, 0.0, 1.0]
    assert np.allclose(obs[3:], [-1.0, 0.5])
    d0 = ActionDelay(gym.make("Pendulum-v1"), delay=0)
    o1, _ = d0.reset(seed=1); o2, _ = gym.make("Pendulum-v1").reset(seed=1)
    assert np.allclose(o1, o2)


def test_experiment_run(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = ExperimentRun("e0000000", "unit_test", {"a": 1}, base_dir=str(tmp_path))
    run.log(seed=0, episode=0, ret=1.0)
    run.log(seed=0, episode=1, ret=2.0)
    fig, ax = plt.subplots(); ax.plot([0, 1])
    run.save_figure(fig, "curve")
    run.save_summary([{"seed": 0, "mean": 1.5}])
    made = run.export_for_github()
    assert len(made) == 3
    run2 = ExperimentRun("e0000000", "unit_test", {}, base_dir=str(tmp_path))
    assert run2.dir != run.dir


CLIFF = """
###########
#.........#
#.#######.#
#S.......G#
#TTTTTTTTT#
###########"""


def test_traps_and_route_switch():
    grid, s, g, traps = grid_from_text(CLIFF)
    assert len(traps) == 9

    def first_move(p):
        env = GridMazeEnv(grid=grid, start=s, goal=g, traps=traps, slip_prob=p,
                          step_reward=-1.0, goal_reward=-1.0, trap_reward=-100.0)
        P, R = env.transition_model()
        assert np.allclose(P.sum(axis=2), 1.0)
        V, Q, pi = value_iteration(P, R, gamma=0.99)
        return int(pi[env.pos_to_obs(s)])
    assert first_move(0.0) == 3        # 滑らなければ近道(右)
    assert first_move(0.5) == 0        # よく滑るなら遠回り(上)
    env = GridMazeEnv(grid=grid, start=s, goal=g, traps=traps)
    env.reset(seed=0)
    *_, term, trunc, info = env.step(1)  # 下は落とし穴
    assert term and info["in_trap"]
    G = env.to_networkx()
    assert all(t not in G for t in traps)
