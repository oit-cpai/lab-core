"""実験結果の保存(Google Drive / ローカル)。

``technical-handbook/colab/experiment-workflow.md`` の保存手順を関数にまとめたもの。
1回の実行ごとに ``<保存先>/<学籍番号>/<日付_時刻_実験名>/`` を作り、
設定・ログ・図・要約を同じ形式で残す。

使い方
------
>>> from cpai_lab.utils.experiment import ExperimentRun
>>> run = ExperimentRun("e16xxxxx", "qlearning_maze", config={"alpha": 0.1, "seeds": [0, 1, 2]})
>>> run.log(seed=0, episode=0, ret=-0.3, steps=31)     # 1行ずつ metrics.csv に追記
>>> run.save_figure(fig, "learning_curve")              # learning_curve.png
>>> run.save_summary([{"seed": 0, "success_rate": 0.9}]) # summary.csv
>>> run.export_for_github()                             # 卒研repo用の for_github/ を作る
"""

import csv
import datetime
import json
import os
import platform
import shutil

DRIVE_BASE = "/content/drive/MyDrive/cpai-results"
LOCAL_BASE = "./cpai-results"


def _versions(extra_packages=()):
    from importlib import metadata
    names = ["numpy", "gymnasium", "matplotlib", "cpai-lab"] + list(extra_packages)
    out = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


class ExperimentRun:
    """1回の実験の保存先と保存処理。

    Parameters
    ----------
    student_id : str
        学籍番号(卒研repo名。例 ``"e1623049"``)。
    experiment : str
        実験名。半角英小文字と ``_`` で(例 ``"qlearning_maze"``)。
    config : dict
        実験条件。``config.json`` に保存される(ライブラリのバージョンも自動で追加)。
    base_dir : str, optional
        保存先の親フォルダ。省略すると、Colab では Google Drive をマウントして
        ``マイドライブ/cpai-results``、それ以外では ``./cpai-results``。
    extra_packages : list of str
        バージョンを記録したい追加のパッケージ名(例 ``["pogema"]``)。
    """

    def __init__(self, student_id, experiment, config=None, base_dir=None,
                 extra_packages=()):
        if base_dir is None:
            base_dir = LOCAL_BASE
            try:
                from google.colab import drive  # noqa: F401
                drive.mount("/content/drive")
                base_dir = DRIVE_BASE
            except ImportError:
                pass
        self.name = f"{datetime.datetime.now():%Y-%m-%d_%H%M}_{experiment}"
        path = os.path.join(base_dir, student_id, self.name)
        k = 2
        while os.path.exists(path):             # 同じ分に2回実行しても上書きしない
            path = os.path.join(base_dir, student_id, f"{self.name}_{k}")
            k += 1
        self.name = os.path.basename(path)
        self.dir = path
        os.makedirs(self.dir)
        self.config = dict(config or {})
        self.config["versions"] = _versions(extra_packages)
        self.config["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.path("config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2, default=str)
        self._fields = None
        self._figures = []
        print("保存先:", self.dir)

    def path(self, filename):
        """この実行のフォルダ内のファイルパス。"""
        return os.path.join(self.dir, filename)

    def log(self, **row):
        """``metrics.csv`` に1行追記する(列は最初の呼び出しで決まる)。"""
        file = self.path("metrics.csv")
        if self._fields is None:
            self._fields = list(row.keys())
            with open(file, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        with open(file, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow(row)

    def log_rows(self, rows):
        """複数行をまとめて追記する。"""
        for row in rows:
            self.log(**row)

    def save_figure(self, fig, name, dpi=150):
        """matplotlib の図を ``<name>.png`` として保存する。"""
        fig.savefig(self.path(f"{name}.png"), dpi=dpi, bbox_inches="tight")
        self._figures.append(name)

    def save_summary(self, rows):
        """要約(dict のリスト、または pandas.DataFrame)を ``summary.csv`` に保存する。"""
        if hasattr(rows, "to_csv"):
            rows.to_csv(self.path("summary.csv"), index=False)
            return
        rows = list(rows)
        with open(self.path("summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def save_array(self, name, array):
        """numpy 配列を ``<name>.npy`` として保存する(Q表など)。"""
        import numpy as np
        np.save(self.path(f"{name}.npy"), array)

    def export_for_github(self):
        """卒研repoにアップロードするファイルを ``for_github/`` にまとめる。

        ``configs/<実行名>.json``、``results/<実行名>_summary.csv``、
        ``figures/<実行名>_<図の名前>.png`` を作り、作ったファイルの一覧を返す。
        ``for_github`` の中の3フォルダを、卒研repoの「Upload files」にまとめてドラッグする。
        """
        gh = self.path("for_github")
        made = []
        jobs = [("config.json", "configs", f"{self.name}.json"),
                ("summary.csv", "results", f"{self.name}_summary.csv")]
        jobs += [(f"{n}.png", "figures", f"{self.name}_{n}.png") for n in self._figures]
        for src, sub, dst in jobs:
            if os.path.exists(self.path(src)):
                os.makedirs(os.path.join(gh, sub), exist_ok=True)
                shutil.copy(self.path(src), os.path.join(gh, sub, dst))
                made.append(f"{sub}/{dst}")
        for m in made:
            print(m)
        return made
