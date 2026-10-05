# cpai_lab.agents

学習アルゴリズムと実験ランナー。

- `bandit.py` : バンディット用(`uniform_random_method`・`value_based_method`・`ucb_method`・`run_experiment`)
- `dp.py` : 動的計画法(`value_iteration`・`policy_evaluation`)。遷移確率が分かっているときの正解の計算(検算用)
- `td.py` : 表形式TD学習(`compute_td_error`・`train_td`)。SARSA / Q-learning / Expected SARSA。環境はgymnasium APIを想定
