# cpai_lab.wrappers

Gymnasium の既存環境に機能を足すラッパー。

- `delay.py` : `ActionDelay(env, delay, augment_obs=False)` — 選んだ行動が `delay` ステップ遅れて反映される。`augment_obs=True` で、まだ反映されていない行動を観測の後ろにつなげる
