# cpai_lab.utils

可視化と共通ユーティリティ。

- `plotting.py` : 方策の可視化(`plot_epsilon_greedy_policy` など)、学習曲線(`smooth`・`plot_learning_curves`)、CliffWalking方策表示(`plot_cliff_policy`)
- `experiment.py` : `ExperimentRun` — 実験結果を Google Drive(Colab)またはローカルに保存し、卒研repo用の `for_github/` を作る
- `misc.py` : `argmax_random_tie`(同点をランダムに破るargmax)
