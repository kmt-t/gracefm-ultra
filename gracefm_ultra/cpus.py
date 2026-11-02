"""使ってよい CPU コア。環境変数 ULTRA_CPUS（既定 0,1,5,6,7,11）。他のプロジェクトと共有するマシンなので、重い処理は必ず pin() で収める。"""
import os

CPUS = {int(v) for v in os.environ.get("ULTRA_CPUS", "0,1,5,6,7,11").split(",")}


def pin(cpus=None):
    """このプロセスの全スレッドを cpus（既定 CPUS）に固定する。sched_setaffinity(0, …) は呼んだスレッドだけなので、numpy/torch の import で
    すでにできているスレッドも含めて、/proc/self/task の全てに設定する。この後にできるスレッドは、呼び出し元の設定を引き継ぐ"""
    cpus = CPUS if cpus is None else set(cpus)
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except (ProcessLookupError, PermissionError):
            pass
