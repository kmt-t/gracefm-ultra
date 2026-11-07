#!/usr/bin/env python3
"""
学習用の音源を Wikimedia Commons から取る。一覧は tools/audio_sources.json（ファイル名、出典、作者、ライセンス、sha1）。
音源の組ごとに <out>/<組>/ に保存する。1 本ずつ順に取り、間隔を空ける。sha1 が合うものは取り直さないので、途中で止めても続きから再開する。
Commons で差し替えられて sha1 が合わないものや消えたものは飛ばして、最後に件数を出す。

  .venv/bin/python tools/collect_audio.py --contact you@example.com
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import CPUS, pin  # noqa: E402
from gracefm_ultra.gfm import AUDIO  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"


def get(url, ua, params=None, rate=0.0):
    """url を取る。rate > 0 なら、その速度 [byte/s] を超えないように読む。拒否されたら (429 など) 間を空けて再試行する"""
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for k in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=120) as r:
                if rate <= 0:
                    return r.read()
                buf, t0 = bytearray(), time.time()
                while chunk := r.read(65536):
                    buf += chunk
                    time.sleep(max(0.0, len(buf) / rate - (time.time() - t0)))
                return bytes(buf)
        except Exception as e:  # noqa: BLE001
            wait = 120 * 2 ** k
            print(f"  再試行 {k + 1}: {e} ({wait} 秒待つ)", flush=True)
            time.sleep(wait)
    raise RuntimeError(url)


def sha1_of(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha1").hexdigest()


def fetch(src, path, ua, rate):
    """src を path に保存する。取れなければ理由を返す"""
    title = urllib.parse.unquote(src["source_page"].rsplit("/wiki/", 1)[1])
    d = json.loads(get(API, ua, {"action": "query", "format": "json", "titles": title, "prop": "imageinfo", "iiprop": "url|sha1"}))
    info = next(iter(d["query"]["pages"].values())).get("imageinfo")
    if not info:
        return "Commons にない"
    if info[0]["sha1"] != src["sha1"]:
        return "差し替えられている"
    data = get(info[0]["url"], ua, rate=rate)
    if hashlib.sha1(data).hexdigest() != src["sha1"]:
        return "sha1 が合わない"
    with open(path, "wb") as f:
        f.write(data)
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=os.path.join(HERE, "audio_sources.json"))
    ap.add_argument("--out", default=AUDIO)
    ap.add_argument("--wait", type=float, default=30.0, help="ダウンロードの間隔 [秒]")
    ap.add_argument("--rate", type=float, default=1.0, help="ダウンロードの速度の上限 [MB/s]")
    ap.add_argument("--contact", required=True, help="User-Agent に入れる連絡先 (Wikimedia の規則)")
    a = ap.parse_args()
    pin({max(CPUS)})
    ua = f"gracefm-ultra/0.1 (training data collection; contact: {a.contact})"
    with open(a.list, encoding="utf-8") as f:
        sets = json.load(f)
    skipped = []
    for name, sources in sets.items():
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        for i, src in enumerate(sources):
            path = os.path.join(a.out, name, src["file"])
            if os.path.exists(path) and sha1_of(path) == src["sha1"]:
                continue
            time.sleep(a.wait)
            why = fetch(src, path, ua, a.rate * 1e6)
            print(f"[{name} {i + 1}/{len(sources)}] {src['file']} {why or 'OK'}", flush=True)
            if why:
                skipped.append(src["file"])
    print(f"完了: 飛ばしたもの {len(skipped)} 本", flush=True)


if __name__ == "__main__":
    main()
