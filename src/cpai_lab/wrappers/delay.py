"""入力遅れ(行動の遅延)を加える Gymnasium ラッパー。"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionDelay(gym.Wrapper):
    """選んだ行動が ``delay`` ステップ遅れて環境に反映されるようにする。

    エピソードの最初の ``delay`` ステップは、``noop_action``(省略時は0の行動)が
    実行される。

    Parameters
    ----------
    env : gymnasium.Env
    delay : int
        遅れのステップ数。0 なら元の環境と同じ。
    noop_action : optional
        遅れている間に実行する行動。省略すると、Box なら全成分0、Discrete なら 0。
    augment_obs : bool
        ``True`` のとき、観測の後ろに「まだ反映されていない行動(古い順に ``delay`` 個)」
        をつなげる。観測と行動がどちらも Box の場合のみ使える。
        遅れのあるMDPは、この拡張した状態を使うとマルコフ性が回復する
        (Katsikopoulos & Engelbrecht, 2003)。

    Examples
    --------
    >>> env = ActionDelay(gym.make("Pendulum-v1"), delay=3, augment_obs=True)
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape    # 元の観測3次元 + 遅れている行動 3×1次元
    (6,)
    """

    def __init__(self, env, delay=1, noop_action=None, augment_obs=False):
        super().__init__(env)
        if delay < 0:
            raise ValueError("delay は0以上")
        self.delay = int(delay)
        if noop_action is None:
            if isinstance(env.action_space, spaces.Box):
                noop_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            else:
                noop_action = 0
        self.noop_action = noop_action
        self.augment_obs = augment_obs
        if augment_obs:
            if not (isinstance(env.observation_space, spaces.Box)
                    and isinstance(env.action_space, spaces.Box)):
                raise ValueError("augment_obs=True は観測・行動がどちらも Box のときだけ使えます")
            o, a = env.observation_space, env.action_space
            low = np.concatenate([o.low.ravel()] + [a.low.ravel()] * self.delay)
            high = np.concatenate([o.high.ravel()] + [a.high.ravel()] * self.delay)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.pending = deque()

    def _augment(self, obs):
        if not self.augment_obs:
            return obs
        parts = [np.asarray(obs, dtype=np.float32).ravel()]
        parts += [np.asarray(a, dtype=np.float32).ravel() for a in self.pending]
        return np.concatenate(parts).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pending = deque([self.noop_action] * self.delay)
        return self._augment(obs), info

    def step(self, action):
        self.pending.append(action)
        applied = self.pending.popleft()
        obs, reward, terminated, truncated, info = self.env.step(applied)
        info = dict(info)
        info["applied_action"] = applied
        return self._augment(obs), reward, terminated, truncated, info
