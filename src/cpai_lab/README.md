# cpai_lab

CPAI研究室の共通Pythonライブラリ本体。

| サブパッケージ | 役割 |
|---|---|
| `envs/` | 環境(GaussianBandit、格子迷路 `GridMazeEnv` と迷路の生成器) |
| `policies/` | 行動選択方策(ε-greedy・Boltzmann・UCB) |
| `agents/` | 学習アルゴリズム・実験ランナー(bandit用メソッド、TD学習、動的計画法) |
| `wrappers/` | 既存の環境に機能を足す部品(入力遅れ `ActionDelay`) |
| `utils/` | 可視化・共通ユーティリティ・実験結果の保存(`ExperimentRun`) |

使用例は repo ルートの README とdocstringを参照。
