# lab-core

CPAI研究室の共通Pythonライブラリ `cpai_lab` と、再利用可能なサンプルnotebookを管理するリポジトリ。

理論説明・教材は `research-handbook`、技術手順は `technical-handbook` を参照。

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

## 運用ルール

- 再利用可能なPythonコードはここに置く。理論説明は `research-handbook` へ。
- 学生個人の実験結果・卒論メモは各学生repoへ。
- 大容量データ・モデル重み・未公開研究計画に関わるものは置かない。
