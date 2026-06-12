# cpai_lab.policies

行動選択方策。Q値(1次元配列)を受け取り、行動の確率分布や行動を返す。

- `epsilon_greedy.py` : `epsilon_greedy_policy`(同点はランダム)、`naive_epsilon_greedy_policy`(比較用)、`sample_action`
- `boltzmann.py` : `boltzmann_policy`(softmax、逆温度β)
- `ucb.py` : `select_ucb_action`(UCB1、未試行行動を優先)

バンディット演習とTD学習(cliff walking)で共通に使う。
