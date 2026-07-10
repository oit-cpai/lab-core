# examples/notebooks

学生向けの演習notebook置き場。

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

## 使い方

ここのnotebookは**直接編集せず**、自分の卒研repoにコピーして使う。
手順は `technical-handbook/colab/use-github-repo.md` を参照。
穴埋め課題(`---- ここから課題 ----` の区間や TODO)を埋めながら上から順に実行する。

解答付き(instructor)版は `research-handbook/notebooks/` にある。まず自力で取り組むこと。
