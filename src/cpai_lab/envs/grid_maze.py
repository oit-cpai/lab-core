"""格子迷路の生成と、Gymnasium形式の迷路環境。

卒研で使う2次元格子の地図(迷路・障害物のある地図)を、共通の形で作るための部品。

- ``generate_maze``  : 通路の長さ(直進のしやすさ)と分岐(ループ)の多さを指定して迷路を作る
- ``grid_from_text`` : 文字列で書いた地図を読み込む
- ``bfs_distances`` / ``shortest_path`` : 最短距離・最短経路(検算に使う)
- ``maze_stats``     : 迷路の特徴(行き止まり・分岐点・直線の長さなど)
- ``GridMazeEnv``    : Gymnasium形式の環境。滑り確率 ``slip_prob`` で遷移を確率的にできる

地図は ``bool`` の2次元配列で表す(``True`` = 壁、``False`` = 通れるマス)。
座標は ``(行, 列)``。行は下向き、列は右向きに増える。
"""

from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None

# 行動: 0=上, 1=下, 2=左, 3=右
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# 地図の生成・読み込み
# ---------------------------------------------------------------------------

def generate_maze(n_rows, n_cols, straightness=0.0, loop_prob=0.0, seed=None):
    """迷路を生成する。

    セル単位で ``n_rows`` × ``n_cols`` の迷路を作り、壁を含めた
    ``(2*n_rows+1)`` × ``(2*n_cols+1)`` の配列として返す。

    Parameters
    ----------
    n_rows, n_cols : int
        セルの数(通路のマス目の数ではない点に注意)。
    straightness : float, 0〜1
        穴掘りのときに、直前と同じ方向へ進む確率。大きいほど長い直線通路ができる。
    loop_prob : float, 0〜1
        生成後に、通路どうしを隔てる壁を壊す確率。大きいほど分岐(ループ)が増え、
        行き止まりが減る。0 なら分岐のない「完全迷路」(どの2点間も経路が1本)。
    seed : int or None
        乱数のseed。同じseedなら同じ迷路になる。

    Returns
    -------
    grid : ndarray of bool, shape (2*n_rows+1, 2*n_cols+1)
        ``True`` が壁。
    """
    rng = np.random.default_rng(seed)
    H, W = 2 * n_rows + 1, 2 * n_cols + 1
    grid = np.ones((H, W), dtype=bool)
    visited = np.zeros((n_rows, n_cols), dtype=bool)

    # 穴掘り法(深さ優先)。直前の方向を優先することで直線の長さを調整する
    stack = [((0, 0), None)]
    visited[0, 0] = True
    grid[1, 1] = False
    while stack:
        (r, c), last = stack[-1]
        options = []
        for d, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and not visited[nr, nc]:
                options.append(d)
        if not options:
            stack.pop()
            continue
        if last in options and rng.random() < straightness:
            d = last
        else:
            d = int(rng.choice(options))
        dr, dc = MOVES[d]
        nr, nc = r + dr, c + dc
        grid[2 * r + 1 + dr, 2 * c + 1 + dc] = False   # 間の壁を壊す
        grid[2 * nr + 1, 2 * nc + 1] = False
        visited[nr, nc] = True
        stack.append(((nr, nc), d))

    # ループを作る:隣り合う2つの通路を隔てている内部の壁を、確率 loop_prob で壊す
    if loop_prob > 0:
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if not grid[r, c]:
                    continue
                horizontal = (r % 2 == 1 and c % 2 == 0)   # 左右のセルを隔てる壁
                vertical = (r % 2 == 0 and c % 2 == 1)     # 上下のセルを隔てる壁
                if (horizontal or vertical) and rng.random() < loop_prob:
                    grid[r, c] = False
    return grid


def grid_from_text(text):
    """文字列の地図を読み込む。

    ``#`` が壁、``.`` が通路、``S`` がスタート、``G`` がゴール、``T`` が落とし穴
    (``S``・``G``・``T`` は通路扱い)。

    Returns
    -------
    grid : ndarray of bool
    start, goal : tuple or None
        ``S`` / ``G`` の位置。書かれていなければ ``None``。
    traps : list of tuple
        ``T`` の位置のリスト(``GridMazeEnv(traps=...)`` に渡す)。
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    width = max(len(ln) for ln in lines)
    grid = np.ones((len(lines), width), dtype=bool)
    start = goal = None
    traps = []
    for r, ln in enumerate(lines):
        for c, ch in enumerate(ln):
            if ch in ".SGT":
                grid[r, c] = False
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
            elif ch == "T":
                traps.append((r, c))
    return grid, start, goal, traps


def grid_to_text(grid, start=None, goal=None, path=None, agent=None, traps=()):
    """地図を文字列にする(表示・確認用)。``path`` のマスは ``*``、落とし穴は ``T``。"""
    chars = np.where(grid, "#", ".").astype(object)
    for p in traps:
        chars[p] = "T"
    if path is not None:
        for p in path:
            chars[p] = "*"
    if start is not None:
        chars[start] = "S"
    if goal is not None:
        chars[goal] = "G"
    if agent is not None:
        chars[agent] = "A"
    return "\n".join("".join(row) for row in chars)


# ---------------------------------------------------------------------------
# 最短距離と迷路の特徴
# ---------------------------------------------------------------------------

def _neighbors(grid, pos):
    H, W = grid.shape
    r, c = pos
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and not grid[nr, nc]:
            yield (nr, nc)


def bfs_distances(grid, source):
    """``source`` から各マスまでの最短ステップ数(幅優先探索)。届かないマスは -1。"""
    dist = np.full(grid.shape, -1, dtype=int)
    dist[source] = 0
    queue = deque([source])
    while queue:
        p = queue.popleft()
        for q in _neighbors(grid, p):
            if dist[q] < 0:
                dist[q] = dist[p] + 1
                queue.append(q)
    return dist


def shortest_path(grid, start, goal):
    """``start`` から ``goal`` までの最短経路(マスのリスト)。届かなければ ``None``。"""
    parent = {start: None}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        if p == goal:
            break
        for q in _neighbors(grid, p):
            if q not in parent:
                parent[q] = p
                queue.append(q)
    if goal not in parent:
        return None
    path, p = [], goal
    while p is not None:
        path.append(p)
        p = parent[p]
    return path[::-1]


def count_turns(path):
    """経路の曲がり角の数(進む方向が変わった回数)。"""
    turns, prev = 0, None
    for a, b in zip(path[:-1], path[1:]):
        d = (b[0] - a[0], b[1] - a[1])
        if prev is not None and d != prev:
            turns += 1
        prev = d
    return turns


def maze_stats(grid, start=None, goal=None):
    """迷路の特徴をまとめて返す。

    Returns
    -------
    dict
        ``n_free``(通れるマス数)、``n_dead_ends``(行き止まりの数)、
        ``n_junctions``(3方向以上に分かれるマスの数)、
        ``mean_straight_length``(縦横の直線通路の平均の長さ。2マス以上の直線のみ)、
        ``start``・``goal`` を与えた場合は ``shortest_length``(最短ステップ数)と
        ``shortest_turns``(最短経路の曲がり角の数)。
    """
    free = ~grid
    degree = np.zeros(grid.shape, dtype=int)
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if free[r, c]:
                degree[r, c] = sum(1 for _ in _neighbors(grid, (r, c)))

    runs = []
    for line in list(free) + list(free.T):      # 各行と各列の、連続した通路の長さ
        n = 0
        for v in list(line) + [False]:
            if v:
                n += 1
            else:
                if n >= 2:
                    runs.append(n)
                n = 0

    stats = {
        "n_free": int(free.sum()),
        "n_dead_ends": int(((degree == 1) & free).sum()),
        "n_junctions": int(((degree >= 3) & free).sum()),
        "mean_straight_length": float(np.mean(runs)) if runs else 0.0,
    }
    if start is not None and goal is not None:
        path = shortest_path(grid, start, goal)
        stats["shortest_length"] = len(path) - 1 if path else -1
        stats["shortest_turns"] = count_turns(path) if path else -1
    return stats


# ---------------------------------------------------------------------------
# Gymnasium 環境
# ---------------------------------------------------------------------------

class GridMazeEnv(gym.Env if gym is not None else object):
    """2次元格子の迷路環境(Gymnasium形式)。

    - 観測:エージェントの位置を番号にしたもの ``r * W + c``(``Discrete(H*W)``)。
      壁のマスの番号には到達しない。
    - 行動:0=上, 1=下, 2=左, 3=右(``Discrete(4)``)。
    - 遷移:確率 ``slip_prob`` で、選んだ行動の代わりに4方向から一様に選んだ行動になる。
      壁や外へ向かう移動ではその場にとどまる。
    - 報酬:ゴールに着いたステップで ``goal_reward``、落とし穴に入ったステップで
      ``trap_reward``、それ以外は ``step_reward``。
    - 終了:ゴールまたは落とし穴で ``terminated``、``max_steps`` に達したら ``truncated``。

    Parameters
    ----------
    grid : ndarray of bool, optional
        地図。省略すると ``maze_kwargs`` で ``generate_maze`` を呼んで作る。
    start, goal : tuple, optional
        省略すると左上の通路と右下の通路。
    slip_prob : float
        行動がランダムにずれる確率(0 なら決定的)。
    step_reward, goal_reward : float
    traps : list of tuple, optional
        落とし穴のマス。入るとエピソードが終わる(崖のようなもの)。
    trap_reward : float
    max_steps : int, optional
        省略すると通れるマス数の4倍。
    maze_kwargs : dict, optional
        ``generate_maze`` に渡す引数(``n_rows``, ``n_cols``, ``straightness``,
        ``loop_prob``, ``seed``)。

    Examples
    --------
    >>> env = GridMazeEnv(maze_kwargs=dict(n_rows=5, n_cols=5, straightness=0.5, seed=0))
    >>> obs, info = env.reset(seed=0)
    >>> print(env.render())
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid=None, start=None, goal=None, slip_prob=0.0,
                 step_reward=-0.01, goal_reward=1.0, traps=None, trap_reward=-1.0,
                 max_steps=None, maze_kwargs=None, render_mode="ansi"):
        if grid is None:
            kw = dict(n_rows=5, n_cols=5)
            kw.update(maze_kwargs or {})
            grid = generate_maze(**kw)
        self.grid = np.asarray(grid, dtype=bool)
        H, W = self.grid.shape
        free = np.argwhere(~self.grid)
        self.start = tuple(start) if start is not None else tuple(free[0])
        self.goal = tuple(goal) if goal is not None else tuple(free[-1])
        if self.grid[self.start] or self.grid[self.goal]:
            raise ValueError("start / goal が壁の上にあります")
        self.slip_prob = float(slip_prob)
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.traps = [tuple(p) for p in (traps or [])]
        self.trap_reward = float(trap_reward)
        if self.start in self.traps or self.goal in self.traps:
            raise ValueError("start / goal が落とし穴の上にあります")
        self.max_steps = int(max_steps) if max_steps is not None else 4 * len(free)
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(H * W)
        self.action_space = spaces.Discrete(4)
        self.pos = self.start
        self.t = 0

    # --- 位置と番号の変換 ---
    def pos_to_obs(self, pos):
        return int(pos[0] * self.grid.shape[1] + pos[1])

    def obs_to_pos(self, obs):
        return divmod(int(obs), self.grid.shape[1])

    # --- Gymnasium API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.t = self.start, 0
        return self.pos_to_obs(self.pos), {"pos": self.pos}

    def _move(self, pos, action):
        dr, dc = MOVES[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        H, W = self.grid.shape
        if 0 <= nr < H and 0 <= nc < W and not self.grid[nr, nc]:
            return (nr, nc), False
        return pos, True

    def step(self, action):
        action = int(action)
        slipped = False
        if self.slip_prob > 0 and self.np_random.random() < self.slip_prob:
            action = int(self.np_random.integers(4))
            slipped = True
        self.pos, hit_wall = self._move(self.pos, action)
        self.t += 1
        at_goal = self.pos == self.goal
        in_trap = self.pos in self.traps
        terminated = at_goal or in_trap
        truncated = (not terminated) and self.t >= self.max_steps
        reward = self._reward(self.pos)
        info = {"pos": self.pos, "slipped": slipped, "hit_wall": hit_wall,
                "at_goal": at_goal, "in_trap": in_trap}
        return self.pos_to_obs(self.pos), reward, terminated, truncated, info

    def _reward(self, pos):
        if pos == self.goal:
            return self.goal_reward
        if pos in self.traps:
            return self.trap_reward
        return self.step_reward

    def render(self):
        return grid_to_text(self.grid, self.start, self.goal, agent=self.pos, traps=self.traps)

    # --- 検算・理論計算のための道具 ---
    def optimal_steps(self):
        """滑りがない場合の、スタートからゴールまでの最短ステップ数。"""
        return int(bfs_distances(self.grid, self.start)[self.goal])

    def transition_model(self):
        """遷移確率 ``P[s, a, s']`` と期待報酬 ``R[s, a]`` を返す(動的計画法用)。

        ゴールと落とし穴は吸収状態(どの行動でもその場にとどまり、報酬0)として扱う。
        状態の番号は観測と同じ ``r * W + c``。壁のマスは自分自身に戻る形にしてある。
        """
        H, W = self.grid.shape
        n_s = H * W
        P = np.zeros((n_s, 4, n_s))
        R = np.zeros((n_s, 4))
        g = self.pos_to_obs(self.goal)
        absorbing = {g} | {self.pos_to_obs(p) for p in self.traps}
        for s in range(n_s):
            pos = self.obs_to_pos(s)
            if self.grid[pos] or s in absorbing:
                P[s, :, s] = 1.0
                continue
            for a in range(4):
                probs = np.zeros(4)
                probs[a] += 1.0 - self.slip_prob
                probs += self.slip_prob / 4.0
                for a2 in range(4):
                    pos2 = self._move(pos, a2)[0]
                    s2 = self.pos_to_obs(pos2)
                    P[s, a, s2] += probs[a2]
                    R[s, a] += probs[a2] * self._reward(pos2)
        return P, R

    def to_networkx(self):
        """通れるマスを頂点、隣り合うマスを辺とする ``networkx.Graph`` を返す(落とし穴は除く)。"""
        import networkx as nx
        G = nx.Graph()
        for r, c in np.argwhere(~self.grid):
            if (int(r), int(c)) in self.traps:
                continue
            G.add_node((int(r), int(c)))
            for q in _neighbors(self.grid, (int(r), int(c))):
                if q not in self.traps:
                    G.add_edge((int(r), int(c)), q)
        return G
