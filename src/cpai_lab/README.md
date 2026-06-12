# cpai_lab

CPAI研究室の共通Pythonライブラリ本体。

| サブパッケージ | 役割 |
|---|---|
| `envs/` | 環境(GaussianBandit など) |
| `policies/` | 行動選択方策(ε-greedy・Boltzmann・UCB) |
| `agents/` | 学習アルゴリズム・実験ランナー(bandit用メソッド、TD学習) |
| `utils/` | 可視化・共通ユーティリティ |

使用例は repo ルートの README とdocstringを参照。
