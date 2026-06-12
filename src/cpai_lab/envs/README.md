# cpai_lab.envs

教育・卒研用の小規模環境。

- `bandit.py` : `GaussianBandit` — 報酬がガウス分布に従う多腕バンディット

グリッドワールド系の課題には gymnasium の `CliffWalking-v0` などを利用する。
新しい環境を追加するときは gymnasium API(`reset` / `step`)に合わせること。
