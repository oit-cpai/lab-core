# cpai_lab.agents

学習アルゴリズムと実験ランナー。

- `bandit.py` : バンディット用(`uniform_random_method`・`value_based_method`・`ucb_method`・`run_experiment`)
- `td.py` : 表形式TD学習(`compute_td_error`・`train_td`)。SARSA / Q-learning / Expected SARSA。環境はgymnasium APIを想定
