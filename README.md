# lab-core

CPAI研究室の**演習・実験用notebook**と、それらが使う共通Pythonライブラリ `cpai_lab` をまとめたリポジトリです。理論の解説は `research-handbook`、Colabなどの操作手順は `technical-handbook` にあります。

## このリポジトリにあるもの

| 場所 | 内容 |
|---|---|
| [`examples/notebooks/`](examples/notebooks/) | 演習notebook(穴埋め式)と、卒研の実験テンプレート [`experiment_template.ipynb`](examples/notebooks/experiment_template.ipynb)。一覧と対応する解説は [`examples/notebooks/README.md`](examples/notebooks/README.md) |
| [`src/cpai_lab/`](src/cpai_lab/) | 演習notebookが使う共通ライブラリ(環境・方策・エージェント・可視化) |
| `RLbasic/` | 旧教育資料(整理中) |

notebookは直接編集せず、**自分の卒研repoにコピーして使ってください**。手順は `technical-handbook/colab/use-github-repo.md` にあります。

## インストール

Colab / ローカル共通:

```bash
pip install git+https://github.com/oit-cpai/lab-core.git
```

開発用(クローン済みの場合):

```bash
pip install -e .
```

## 使い方

```python
from cpai_lab.envs import GaussianBandit
from cpai_lab.policies import epsilon_greedy_policy, boltzmann_policy
from cpai_lab.agents import value_based_method, run_experiment
from cpai_lab.utils import plot_epsilon_greedy_policy
```

トップレベルからの一括importも可能:

```python
from cpai_lab import GaussianBandit, epsilon_greedy_policy
```

## 構成

```text
lab-core/
  pyproject.toml
  requirements.txt
  src/
    cpai_lab/
      envs/        # 環境(GaussianBandit など)
      policies/    # 行動選択方策(ε-greedy, Boltzmann, UCB)
      agents/      # 学習メソッド・実験ランナー
      utils/       # 可視化・共通ユーティリティ
  examples/
    notebooks/     # 学生向けサンプル・演習notebook
  RLbasic/         # (整理中)旧教育用notebook
```

