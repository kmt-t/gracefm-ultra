#!/usr/bin/env python3
"""
プロファイルごとの学習を、データ生成から ONNX の書き出しと Rust の受信の確認まで通す。

  .venv/bin/python tools/train_profile.py B200 B200-L
  .venv/bin/python tools/train_profile.py --all
  .venv/bin/python tools/train_profile.py M700 --stages postnet,export,verify   # 途中から
  .venv/bin/python tools/train_profile.py B200M --quick --data /tmp/q --runs /tmp/q/runs --models /tmp/q/models   # 短いデータと短い学習で動作確認

段: data → mendnet → postdata → postnet → export → verify。eval は時間がかかるので --stages で指定したときだけ。
データ生成は済んだファイルを飛ばすので、止まったらそのまま再実行できる。ログは runs/train_<プロファイル>.log。
1 プロファイルで最大 127 GB × (標本化周波数 / 32 kHz) のデータを作る。--clean で書き出しのあとデータを消す。
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS  # noqa: E402
from gracefm.profiles import PROFILES  # noqa: E402

STAGES = ("data", "mendnet", "postdata", "postnet", "export", "verify")
EXTRA_STAGES = ("eval",)
PY = sys.executable
GB_PER_RUN = 127                # B200 の 1 プロファイルぶんのデータ [GB]
RESIDENT = 180                  # 32 kHz のとき MendNet がメモリに置く区間の数 (46 GB)
ESN0_AWGN = {"qam16h": "16,18,20,22,24,26,28,30", "qpsk": "4,6,8,10,12,14,16,18"}   # 評価の Es/N0 [dB]
ESN0_MIN = {"qam16h": 6, "qpsk": 0}                                                  # 学習のシナリオの Es/N0 の下限 [dB]


def run(log, *cmd):
    print("  " + " ".join(os.path.basename(c) if os.path.isabs(c) else c for c in cmd), flush=True)
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"失敗 (終了コード {r.returncode})。ログ: {log}")


def check_zips(pattern):
    """生成したファイルの CRC を確かめて、壊れたものは消す (再実行で作り直す)"""
    bad = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        try:
            with zipfile.ZipFile(f) as z:
                broken = z.testzip()
        except zipfile.BadZipFile:
            broken = "zip"
        if broken:
            bad.append(f)
            os.remove(f)
    return bad


def train_profile(name, a):
    p = PROFILES[name]
    data = os.path.join(a.data, name)
    mend, post = os.path.join(a.runs, f"mendnet_{name}.pt"), os.path.join(a.runs, f"post_{name}.pt")
    models = os.path.join(a.models, name)
    log = os.path.join(a.runs, f"train_{name}.log")
    common = ["--profile", name, "--data", a.data]
    tool = lambda t: [PY, "-u", os.path.join(HERE, t)]  # noqa: E731

    if "data" in a.stages:
        need = GB_PER_RUN * p.fs / 32000
        if not a.quick and shutil.disk_usage(a.data).free < need * 2**30 and not os.path.exists(data):
            raise SystemExit(f"{name}: データに約 {need:.0f} GB 要りますが、{a.data} の空きが足りません")
        g = tool("gen_mend.py") + common + ["--workers", str(len(CPUS))]
        emin = str(ESN0_MIN[p.modulation])
        awgn = ["--sub", "mend_awgn", "--awgn", ESN0_AWGN[p.modulation]]
        q = ["--segments", "1", "--seconds", "4", "--scenarios", "2"] if a.quick else []
        for split in ("val", "train"):
            run(log, *g, "--split", split, "--sub", "mend", "--segments", "4", "--esn0-min", emin, *q)
            if not a.quick:
                run(log, *g, "--split", split, "--sub", "mend", "--src", a.commons, "--segments", "2")
            if split == "val":
                run(log, *g, "--split", split, *awgn, "--segments", "4", *q[:4])
                if not a.quick:
                    run(log, *g, "--split", split, *awgn, "--src", a.commons, "--segments", "2")

    if "mendnet" in a.stages:
        res = max(int(RESIDENT * 32000 / p.fs), 1)
        run(log, *tool("train_mendnet.py"), *common, "--minutes", str(a.mend_minutes), "--resident", str(res), "--swap", "8", "--out", mend)

    if "postdata" in a.stages:
        g = tool("gen_post.py") + common + ["--ckpt", mend]
        for _ in range(2):
            run(log, *g, "--src", "mend_awgn", "--split", "val", "--out", "awgn_val")
            run(log, *g, "--src", "mend", "--split", "val", "--out", "val")
            run(log, *g, "--src", "mend", "--split", "train", "--out", "train")
            bad = check_zips(os.path.join(data, "post", "*", "*.npz"))
            if not bad:
                break
            print(f"  CRC エラー {len(bad)} 件を作り直す", flush=True)
        else:
            raise SystemExit(f"{name}: 作り直しても CRC エラーが残ります: {bad}")
        run(log, *tool("check_gate.py"), *common)

    if "postnet" in a.stages:
        run(log, *tool("train_post.py"), *common, "--minutes", str(a.post_minutes), "--out", post)

    if "export" in a.stages:
        run(log, *tool("export_onnx.py"), *common, "--mend", mend, "--post", post, "--out", models)

    if "verify" in a.stages:
        run(log, *tool("test_rx_onnx.py"), "--profile", name, "--mend", mend, "--post", post, "--models", models)

    if "eval" in a.stages:
        run(log, *tool("eval_post.py"), *common, "--ckpt", post, "--awgn")
        run(log, *tool("eval_post.py"), *common, "--ckpt", post)

    if a.clean:
        shutil.rmtree(data)


def main():
    root = os.path.join(HERE, "..")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profiles", nargs="*")
    ap.add_argument("--all", action="store_true", help="すべてのプロファイル")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"段 (カンマ区切り): {', '.join(STAGES + EXTRA_STAGES)}")
    ap.add_argument("--mend-minutes", type=float, default=360)
    ap.add_argument("--post-minutes", type=float, default=180)
    ap.add_argument("--data", default=os.path.join(root, "data"))
    ap.add_argument("--runs", default=os.path.join(root, "runs"))
    ap.add_argument("--models", default=os.path.join(root, "engine", "models"))
    ap.add_argument("--commons", default=os.path.join(root, "data", "audio", "commons"), help="tools/collect_audio.py で集めた音源")
    ap.add_argument("--clean", action="store_true", help="終わったらそのプロファイルのデータを消す")
    ap.add_argument("--quick", action="store_true", help="動作確認: 短いデータと 1 分足らずの学習")
    a = ap.parse_args()
    names = list(PROFILES) if a.all else a.profiles
    if not names or set(names) - set(PROFILES):
        ap.error(f"プロファイルは {list(PROFILES)} から選んでください")
    a.stages = a.stages.split(",")
    if set(a.stages) - set(STAGES + EXTRA_STAGES):
        ap.error(f"段は {STAGES + EXTRA_STAGES} から選んでください")
    if a.quick:
        a.mend_minutes, a.post_minutes = 0.5, 0.5
    elif "data" in a.stages and not os.path.isdir(a.commons):
        ap.error(f"音源がありません: {a.commons}（tools/collect_audio.py で集める）")
    os.makedirs(a.runs, exist_ok=True)
    os.makedirs(a.data, exist_ok=True)
    for n in names:
        print(f"== {n}", flush=True)
        train_profile(n, a)


if __name__ == "__main__":
    main()
