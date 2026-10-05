# cpai_lab.envs

教育・卒研用の小規模環境。

- `bandit.py` : `GaussianBandit` — 報酬がガウス分布に従う多腕バンディット
- `grid_maze.py` : 2次元格子の迷路・地図(卒研の共通部品)
  - `generate_maze(n_rows, n_cols, straightness, loop_prob, seed)` — 直線の長さと分岐の多さを指定して迷路を作る
  - `grid_from_text` / `grid_to_text` — 文字列の地図の読み書き
  - `bfs_distances` / `shortest_path` / `count_turns` / `maze_stats` — 最短距離・曲がり角・行き止まりなど(検算用)
  - `GridMazeEnv` — Gymnasium形式の迷路環境。`slip_prob` で遷移を確率的にできる。`transition_model()` で遷移確率、`optimal_steps()` で最短ステップ数、`to_networkx()` でグラフ

グリッドワールド系の課題には gymnasium の `CliffWalking-v0` などを利用する。
新しい環境を追加するときは gymnasium API(`reset` / `step`)に合わせること。
