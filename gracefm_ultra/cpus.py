"""使ってよい CPU コア。環境変数 ULTRA_CPUS（既定 0,1,5,6,7,11）のうち、このマシンにあるもの。他のプロジェクトと共有するマシンなので、重い処理は必ず pin() で収める。"""
import os

from threadpoolctl import threadpool_limits

CPUS = {int(v) for v in os.environ.get("ULTRA_CPUS", "0,1,5,6,7,11").split(",")} & os.sched_getaffinity(0)


def pin(cpus=None):
    """このプロセスの全スレッドを cpus（既定 CPUS）に固定する。sched_setaffinity(0, …) は呼んだスレッドだけなので、numpy/torch の import で
    すでにできているスレッドも含めて、/proc/self/task の全てに設定する。この後にできるスレッドは、呼び出し元の設定を引き継ぐ。
    numpy (OpenBLAS) や OpenMP のスレッド数も cpus の数に揃える (多いと、少ないコアの上で空回りして何十倍も遅くなる)"""
    cpus = CPUS if cpus is None else set(cpus)
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except (ProcessLookupError, PermissionError):
            pass
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):                     # この後に読み込まれるライブラリの分
        os.environ[k] = str(len(cpus))
    threadpool_limits(len(cpus))
