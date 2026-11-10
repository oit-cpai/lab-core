# examples/notebooks

学生向けの演習notebook置き場。

> [!WARNING]
> **このフォルダの演習 notebook は、2027年3月末(いまの4年生の卒業)のあとに削除する。**
>
> 基礎の演習は、研究の基礎コース [`cpai-foundations`](https://github.com/oit-cpai/cpai-foundations) に、新しい単元として作り直している。新しい単元では、NumPy・Matplotlib・SciPy で自分で実装し、最後に `cpai_lab` との関係を学ぶ。
>
> - **3年生(早期配属)**:このフォルダの notebook ではなく、`cpai-foundations` の単元を使う。単元ができるまでは、学習のロードマップの指示に従う。
> - **4年生**:卒研で使っている notebook は、卒業まで、このまま使ってよい。自分の卒研 repo にコピーしたものは消えない。
> - **卒研の実験のテンプレート `experiment_template.ipynb`** は、`cpai_lab` の使い方の例として、このフォルダに残す。

| いまの notebook | 移る先(cpai-foundations の単元) |
|---|---|
| `bandit_lecture` | R1 探索と活用(バンディット) |
| `gridworld_step_exercise` | R2 環境を作る |
| `dp_gridworld` | R4 動的計画法 |
| `cliff_walking_lecture` | R7 TD学習・Sarsa・Q学習 |
| `arm_kinematics` | [K1 順運動学](https://github.com/oit-cpai/cpai-foundations/blob/main/kinematics/01-forward-kinematics.md)・[K2 解析的な逆運動学](https://github.com/oit-cpai/cpai-foundations/blob/main/kinematics/02-inverse-kinematics.md)(作成済み)、K4 数値逆運動学 |
| `polynomial_regression` | S4 多項式回帰・リッジ回帰と過学習 |
| `logistic_regression` | S5 ロジスティック回帰と分類 |
| `reinforce_cartpole`、`dqn_cartpole`、`diff_drive`、`kmeans_gmm`、`pca`、`mlp_backprop` | 発展の教材。扱いは、決まりしだいここで知らせる |

`cpai-foundations` の単元ができたら、上の表に単元へのリンクを加える。

| notebook | 内容 | 解説資料(research-handbook) |
|---|---|---|
| `bandit_lecture.ipynb` | バンディット問題(ε-greedy・Q値更新・UCB) | `reinforcement-learning/bandit.md` |
| `gridworld_step_exercise.ipynb` | 環境を作る:`step()` を自作し既存エージェントにつなぐ | `reinforcement-learning/gridworld-step.md` |
| `dp_gridworld.ipynb` | 動的計画法(方策評価・方策反復・価値反復) | `reinforcement-learning/dynamic-programming.md` |
| `cliff_walking_lecture.ipynb` | TD学習(SARSA・Q-learning・Expected SARSA) | `reinforcement-learning/cliff-walking.md`・`td-learning.md` |
| `reinforce_cartpole.ipynb` | REINFORCE(方策勾配・ベースライン、numpy実装) | `reinforcement-learning/policy-gradient.md` |
| `dqn_cartpole.ipynb` | DQN(replay buffer・target network、PyTorch/Colab推奨) | `reinforcement-learning/dqn.md` |
| `arm_kinematics.ipynb` | 2リンクアームの運動学(FK・IK・ヤコビアン・特異点) | `robotics/kinematics.md` |
| `diff_drive.ipynb` | 差動二輪の運動学と経路追従(デッドレコニング・pure pursuit) | `robotics/mobile-robots.md` |
| `polynomial_regression.ipynb` | 多項式回帰と過学習(正規方程式・ridge・交差検証) | `machine-learning/ml-basics.md`・`linear-regression.md` |
| `logistic_regression.ipynb` | ロジスティック回帰(交差エントロピー・勾配降下) | `machine-learning/classification.md` |
| `kmeans_gmm.ipynb` | k-meansとGMM+EM(負担率・対数尤度の単調増加) | `machine-learning/clustering.md` |
| `pca.ipynb` | 主成分分析(固有分解・寄与率・再構成誤差) | `machine-learning/dimensionality-reduction.md` |
| `mlp_backprop.ipynb` | MLPと誤差逆伝播のnumpy実装(勾配チェック込み) | `deep-learning/nn-basics.md`・`backprop.md` |
| `experiment_template.ipynb` | 卒研の実験テンプレート:Google Driveへの結果保存(条件・ログ・図・要約)と卒研repo用ファイルの書き出し。FrozenLakeのQ学習を例に、環境・手法を置き換えて使う | `technical-handbook/colab/experiment-workflow.md`(技術手順) |

## 使い方

ここのnotebookは**直接編集せず**、自分の卒研repoにコピーして使う。
手順は `technical-handbook/colab/use-github-repo.md` を参照。
穴埋め課題(`---- ここから課題 ----` の区間や TODO)を埋めながら上から順に実行する。

解答付き(instructor)版は `research-handbook/notebooks/` にある。まず自力で取り組むこと。
