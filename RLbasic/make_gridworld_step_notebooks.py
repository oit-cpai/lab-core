#!/usr/bin/env python3
"""Generate the "step() を作る" GridWorld notebooks.

Produces two notebooks from a single source of truth:

  - lab-core/examples/notebooks/gridworld_step_exercise.ipynb   (学生用・穴埋め)
  - research-handbook/notebooks/gridworld_step_solution.ipynb    (解答付き)

The exercise version replaces the body of the cells tagged as "solution"
with TODO blanks; everything else is shared. Uses raw JSON (no nbformat).

Companion teaching material:
  research-handbook/reinforcement-learning/gridworld-step.md

Usage:  python make_gridworld_step_notebooks.py
"""
import json
import os
import uuid


# ── helpers ──────────────────────────────────────────────────────────
def _id():
    return uuid.uuid4().hex[:8]


def md(src):
    return {"cell_type": "markdown", "id": _id(), "metadata": {},
            "source": src.splitlines(True)}


def cc(src):
    return {"cell_type": "code", "id": _id(), "metadata": {},
            "source": src.splitlines(True), "outputs": [],
            "execution_count": None}


def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python",
                                           "version": "3.10.0"}},
            "cells": cells}


# ── shared content ───────────────────────────────────────────────────
TITLE = '''\
# 環境を作る:step() で学ぶエージェント-環境相互作用

## この notebook のゴール

1. エージェント-環境**相互作用ループ**(`reset` → `step` → … → 終了)を体験する
2. gymnasium API の `step(action)` が返す **5つ組** `(observation, reward, terminated, truncated, info)` を理解する
3. 状態を内部に持つグリッドワールドの **`step()` を自分で実装する**
4. 自作環境を、既存のTD学習エージェント `cpai_lab.agents.train_td` に**そのままつなぐ**

理論の解説は `research-handbook/reinforcement-learning/gridworld-step.md` を参照。
バンディット(`bandit.md`)・Cliff Walking(`cliff-walking.md`)の続編です。'''

SETUP = '''\
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 状態数・行動数を表す Discrete。Colab では gymnasium が入っているのでそれを使う。
try:
    from gymnasium.spaces import Discrete
except ImportError:
    # gymnasium が無い環境用の最小フォールバック(.n だけ使う)
    class Discrete:
        def __init__(self, n):
            self.n = n
        def sample(self, rng=np.random):
            return int(rng.integers(self.n))'''

ENV_DESC = '''\
## 1. 作る環境:4×4 グリッドワールド

```text
 S  .  .  .          S = 開始 (row=0, col=0)
 .  #  .  .          # = 壁(入れない)
 .  #  .  .          G = ゴール (row=3, col=3)
 .  .  .  G
```

| 項目 | 内容 |
|------|------|
| 状態 $s$ | 位置 `(row, col)`。エージェントへは整数index `row * n_cols + col` で渡す |
| 行動 $a$ | `0`:上、`1`:右、`2`:下、`3`:左 |
| 遷移 | 隣へ移動。グリッド外・壁マスへはその場に留まる |
| 報酬 | ゴール到達 $+1.0$、それ以外のステップ $-1.0$ |
| 終了 | ゴール到達で `terminated=True` |

gymnasium の約束に合わせ、`reset()` は `(obs, info)`、`step(a)` は
`(obs, reward, terminated, truncated, info)` を返します。'''

STEP_A_MD = '''\
## 2. Step A:状態を内部に保持する(`__init__` と `reset`)

環境は「いま自分がどの状態にいるか」を**インスタンス変数** `self.agent_pos` として
覚えておきます。これが「環境が状態を持つ」ということ。`reset()` でスタートに戻します。

**TODO 1**:`reset()` の中で `self.agent_pos` をスタート位置に戻し、
`(初期状態のindex, {})` を返してください。'''

# The class is built across cells; Step A defines __init__/reset (+ helpers),
# Step B adds _next_pos, Step C/D add step. To keep it runnable cell-by-cell
# we define the full class in one "solution" cell per stage by subclassing?
# Simpler: define the whole class once, blanking the relevant methods.

CLASS_SOLUTION = '''\
class GridWorld:
    \'\'\'4x4 deterministic gridworld(gymnasium 互換)。\'\'\'

    # 行動 → (Δrow, Δcol)
    MOVES = {0: (-1, 0),   # 上
             1: (0, 1),    # 右
             2: (1, 0),    # 下
             3: (0, -1)}   # 左
    ACTION_NAMES = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(self, n_rows=4, n_cols=4, start=(0, 0), goal=(3, 3),
                 walls=((1, 1), (2, 1)), step_reward=-1.0, goal_reward=1.0,
                 slip=0.0):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.start, self.goal = start, goal
        self.walls = set(walls)
        self.step_reward, self.goal_reward = step_reward, goal_reward
        self.slip = slip                       # 0.0 なら決定的
        self.observation_space = Discrete(n_rows * n_cols)
        self.action_space = Discrete(4)
        self.reset()

    # --- 座標 ↔ index ---
    def _to_index(self, pos):
        row, col = pos
        return row * self.n_cols + col

    # ===== Step A: reset =====
    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)   # 確率版で使う
        self.agent_pos = self.start              # 内部状態をスタートへ
        return self._to_index(self.agent_pos), {}

    # ===== Step B: 行動 → 次の座標 =====
    def _next_pos(self, pos, action):
        d_row, d_col = self.MOVES[action]
        new_row, new_col = pos[0] + d_row, pos[1] + d_col
        # グリッド外 → 動かない
        if not (0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols):
            return pos
        # 壁マス → 動かない
        if (new_row, new_col) in self.walls:
            return pos
        return (new_row, new_col)

    # ===== Step C/D: step =====
    def step(self, action):
        # (確率版)slip の確率で左右にそれる
        if self.slip > 0.0 and self.rng.random() < self.slip:
            action = int(self.rng.choice([(action - 1) % 4, (action + 1) % 4]))
        next_pos = self._next_pos(self.agent_pos, action)
        if next_pos == self.goal:
            reward, terminated = self.goal_reward, True
        else:
            reward, terminated = self.step_reward, False
        self.agent_pos = next_pos                # 内部状態を更新(忘れやすい!)
        return self._to_index(next_pos), reward, terminated, False, {}

    # --- 可視化用 ---
    def render(self):
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                if (r, c) == self.agent_pos:
                    row += " A "
                elif (r, c) == self.goal:
                    row += " G "
                elif (r, c) in self.walls:
                    row += " # "
                else:
                    row += " . "
            print(row)'''

CLASS_EXERCISE = '''\
class GridWorld:
    \'\'\'4x4 deterministic gridworld(gymnasium 互換)。\'\'\'

    # 行動 → (Δrow, Δcol)
    MOVES = {0: (-1, 0),   # 上
             1: (0, 1),    # 右
             2: (1, 0),    # 下
             3: (0, -1)}   # 左
    ACTION_NAMES = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(self, n_rows=4, n_cols=4, start=(0, 0), goal=(3, 3),
                 walls=((1, 1), (2, 1)), step_reward=-1.0, goal_reward=1.0,
                 slip=0.0):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.start, self.goal = start, goal
        self.walls = set(walls)
        self.step_reward, self.goal_reward = step_reward, goal_reward
        self.slip = slip                       # 0.0 なら決定的
        self.observation_space = Discrete(n_rows * n_cols)
        self.action_space = Discrete(4)
        self.reset()

    # --- 座標 ↔ index ---
    def _to_index(self, pos):
        row, col = pos
        return row * self.n_cols + col

    # ===== Step A: reset(TODO 1)=====
    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)   # 確率版で使う
        # TODO 1: 内部状態 self.agent_pos をスタート位置 self.start に戻し、
        #         (初期状態のindex, {}) を返す。ヒント: self._to_index(...)
        self.agent_pos = ___
        return ___, {}

    # ===== Step B: 行動 → 次の座標(TODO 2)=====
    def _next_pos(self, pos, action):
        # TODO 2: 行動 action の増分 (d_row, d_col) を MOVES から取り出し、
        #         pos に足して新しい座標を作る。ただし
        #           - グリッドの外に出るなら pos を返す(動かない)
        #           - 壁マス self.walls に入るなら pos を返す(動かない)
        #           - それ以外は新しい座標を返す
        d_row, d_col = ___
        new_row, new_col = ___, ___
        if not (0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols):
            return ___
        if (new_row, new_col) in self.walls:
            return ___
        return ___

    # ===== Step C/D: step(TODO 3)=====
    def step(self, action):
        # (確率版)slip の確率で左右にそれる。まずは slip=0.0 のまま進めてよい。
        if self.slip > 0.0 and self.rng.random() < self.slip:
            action = int(self.rng.choice([(action - 1) % 4, (action + 1) % 4]))
        # TODO 3:
        #   1) self._next_pos で次の座標 next_pos を求める
        #   2) next_pos がゴールなら reward=goal_reward, terminated=True
        #      そうでなければ reward=step_reward, terminated=False
        #   3) 内部状態 self.agent_pos を next_pos に更新する(重要!)
        #   4) (次状態のindex, reward, terminated, False, {}) を返す
        next_pos = ___
        if ___:
            reward, terminated = ___, ___
        else:
            reward, terminated = ___, ___
        self.agent_pos = ___
        return ___, ___, ___, False, {}

    # --- 可視化用 ---
    def render(self):
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                if (r, c) == self.agent_pos:
                    row += " A "
                elif (r, c) == self.goal:
                    row += " G "
                elif (r, c) in self.walls:
                    row += " # "
                else:
                    row += " . "
            print(row)'''

CHECK_BASIC_MD = '''\
### 動作確認

環境を作って、`reset` と数回の `step` の出力を観察します。'''

CHECK_BASIC = '''\
env = GridWorld()
obs, info = env.reset()
print(f"初期状態 index = {obs}")
env.render()
print()

# 下 → 下 → 右 と動かしてみる
for name, a in [("下", 2), ("下", 2), ("右", 1)]:
    obs, reward, terminated, truncated, info = env.step(a)
    print(f"行動={name} → obs={obs}, reward={reward}, "
          f"terminated={terminated}, truncated={truncated}")
env.render()'''

SANITY_MD = '''\
### Sanity check(自動テスト)

`reset()` / `_next_pos()` / `step()` が正しく実装されているか確認します。
すべてパスすれば Step A〜D は完成です。'''

SANITY = '''\
_env = GridWorld()

# reset はスタート index=0 を返す
_obs, _ = _env.reset()
assert _obs == 0, f"reset の戻り値が不正: {_obs}"

# 左上で「上」→ グリッド外なので動かない、報酬 -1
_obs, _r, _term, _trunc, _ = _env.step(0)
assert _obs == 0, f"境界の処理が不正: {_obs}"
assert _r == -1.0 and not _term

# 壁の処理: (0,1) から「下」→ (1,1) は壁なので動かない
_env.reset()
_env.step(1)                       # (0,0)->(0,1)
_obs, _r, _term, _trunc, _ = _env.step(2)   # (0,1)->下は壁(1,1)
assert _obs == _env._to_index((0, 1)), f"壁の処理が不正: index={_obs}"

# ゴール到達: (3,2) から「右」→ (3,3) で terminated, 報酬 +1
_env.reset()
_env.agent_pos = (3, 2)
_obs, _r, _term, _trunc, _ = _env.step(1)
assert _obs == _env._to_index((3, 3)), f"ゴール到達が不正: index={_obs}"
assert _r == 1.0 and _term, "ゴールなのに reward/terminated が不正"

print("✓ すべての sanity check をパスしました")'''

RANDOM_MD = '''\
## 3. ランダム方策で相互作用ループを回す

学習を入れる前に、でたらめに行動するループで「相互作用が正しく回るか」を確かめます。
これが環境デバッグの基本。第2節の相互作用ループ
`while not (terminated or truncated): ...` をそのまま書きます。

**TODO 4**:`while` ループの中で、ランダムな行動を選び `env.step` を1ステップ進め、
累積報酬を足してください。'''

RANDOM_SOLUTION = '''\
def run_random_episode(env, max_steps=100, seed=0):
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    total, terminated, truncated = 0.0, False, False
    t = 0
    while not (terminated or truncated):
        action = int(rng.integers(env.action_space.n))      # でたらめに行動
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        t += 1
        if t >= max_steps:
            truncated = True                                # 時間切れで打ち切り
    return t, total, terminated

steps, total, reached = run_random_episode(GridWorld())
print(f"{steps} ステップで終了 / 累積報酬={total} / ゴール到達={reached}")'''

RANDOM_EXERCISE = '''\
def run_random_episode(env, max_steps=100, seed=0):
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    total, terminated, truncated = 0.0, False, False
    t = 0
    while not (terminated or truncated):
        # TODO 4:
        #   1) 0〜env.action_space.n の範囲でランダムに action を選ぶ
        #   2) env.step(action) で5つ組を受け取る
        #   3) total に reward を足す
        action = ___
        obs, reward, terminated, truncated, info = ___
        total += ___
        t += 1
        if t >= max_steps:
            truncated = True                                # 時間切れで打ち切り
    return t, total, terminated

steps, total, reached = run_random_episode(GridWorld())
print(f"{steps} ステップで終了 / 累積報酬={total} / ゴール到達={reached}")'''

STOCH_MD = '''\
## 4. 確率的にする(滑る床)

ここまでは決定的でした。`slip > 0` にすると、確率 `slip` で意図した方向ではなく
左右どちらかにそれます(`step()` の中の乱数処理がここで効きます)。
確率的だと同じ方策でも結果がばらつくので、複数 seed で平均を取ります。'''

STOCH = '''\
det = [run_random_episode(GridWorld(slip=0.0), seed=s)[1] for s in range(20)]
sto = [run_random_episode(GridWorld(slip=0.3), seed=s)[1] for s in range(20)]
print(f"決定的   (slip=0.0): 累積報酬 平均 {np.mean(det):.1f}")
print(f"確率的   (slip=0.3): 累積報酬 平均 {np.mean(sto):.1f}")'''

TRAIN_MD = '''\
## 5. 既存エージェントで学習させる

自作環境は gymnasium の約束(`reset`/`step`/`observation_space.n`/`action_space.n`)を
守っているので、`cpai_lab.agents.train_td` を**1行も変えずに**つなげます。
Cliff Walking のときと同じ呼び出しです。'''

TRAIN = '''\
# Colab では lab-core を pip install(初回のみ)
try:
    from cpai_lab.agents import train_td
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/oit-cpai/lab-core.git"])
    from cpai_lab.agents import train_td

env = GridWorld(slip=0.0)
Q_sarsa, r_sarsa = train_td(env, "sarsa", alpha=0.5, epsilon=0.1,
                            gamma=0.95, num_episodes=500, seed=0)
env = GridWorld(slip=0.0)
Q_qlearn, r_qlearn = train_td(env, "qlearning", alpha=0.5, epsilon=0.1,
                              gamma=0.95, num_episodes=500, seed=0)

# 学習曲線(50エピソードの移動平均)
def moving_avg(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")

plt.figure(figsize=(8, 4))
plt.plot(moving_avg(r_sarsa), label="SARSA")
plt.plot(moving_avg(r_qlearn), label="Q-learning")
plt.xlabel("episode"); plt.ylabel("総報酬(移動平均)")
plt.title("学習曲線"); plt.legend(); plt.grid(); plt.show()'''

POLICY_MD = '''\
### 学習した方策を見る

各状態で Q が最大の行動(greedy 方策)を矢印で表示します。
ゴールへ向かう経路が学べていれば成功です。'''

POLICY = '''\
def show_greedy_policy(env, Q):
    arrows = env.ACTION_NAMES
    for r in range(env.n_rows):
        line = ""
        for c in range(env.n_cols):
            if (r, c) == env.goal:
                line += " G "
            elif (r, c) in env.walls:
                line += " # "
            else:
                idx = env._to_index((r, c))
                line += f" {arrows[int(np.argmax(Q[idx]))]} "
        print(line)

print("Q-learning が学習した方策:")
show_greedy_policy(GridWorld(), Q_qlearn)'''

WRAP_MD = '''\
## まとめ

- 強化学習は **`reset` → `step` → … の相互作用ループ**
- `step(action)` は **`(observation, reward, terminated, truncated, info)`** を返す
- 環境は **状態を内部に保持**し、行動から次状態・報酬・終了を計算する
- gymnasium のインタフェースを守れば、**既存エージェント(`train_td`)がそのまま動く**
- 決定的と確率的の違いは **`step()` に乱数が入るかどうか**だけ

### 発展課題

- 報酬設計(`goal_reward`/`step_reward`)を変えると方策はどう変わる?
- 崖マスを追加して Cliff Walking を再現してみる
- `truncated` を環境側で実装する(最大ステップ超過で打ち切り)
- 詳しい解説は `research-handbook/reinforcement-learning/gridworld-step.md`'''


# ── assemble ─────────────────────────────────────────────────────────
def build(cells_class, cells_random):
    return nb([
        md(TITLE),
        cc(SETUP),
        md(ENV_DESC),
        md(STEP_A_MD),
        cc(cells_class),
        md(CHECK_BASIC_MD),
        cc(CHECK_BASIC),
        md(SANITY_MD),
        cc(SANITY),
        md(RANDOM_MD),
        cc(cells_random),
        md(STOCH_MD),
        cc(STOCH),
        md(TRAIN_MD),
        cc(TRAIN),
        md(POLICY_MD),
        cc(POLICY),
        md(WRAP_MD),
    ])


solution = build(CLASS_SOLUTION, RANDOM_SOLUTION)
exercise = build(CLASS_EXERCISE, RANDOM_EXERCISE)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # .../cpai
out_ex = os.path.join(ROOT, "lab-core", "examples", "notebooks",
                      "gridworld_step_exercise.ipynb")
out_sol = os.path.join(ROOT, "research-handbook", "notebooks",
                       "gridworld_step_solution.ipynb")

with open(out_ex, "w", encoding="utf-8") as f:
    json.dump(exercise, f, ensure_ascii=False, indent=1)
with open(out_sol, "w", encoding="utf-8") as f:
    json.dump(solution, f, ensure_ascii=False, indent=1)

print("wrote:", out_ex)
print("wrote:", out_sol)
