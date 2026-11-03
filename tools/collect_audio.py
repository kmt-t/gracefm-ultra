#!/usr/bin/env python3
"""
学習用の音源 (可逆圧縮の FLAC) を Wikimedia Commons から集める。ライセンスは CC0・パブリックドメイン・CC BY に限る (SA・NC・ND は除く)。
1 本ずつ順に取り、間隔を空ける。ステレオで 30 秒以上、標本化周波数 32 kHz 以上のものだけ残す。
鳴き声の録音集 (SKIP_TITLE) は取らない。
出力: <out>/*.flac と <out>/manifest.json (既存の train_lossless/manifest.json と同じ項目)。途中で止めても、続きから再開する。

  .venv/bin/python tools/collect_audio.py --out data/audio/commons --max-gb 30
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import pin  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = "gracefm-ultra-research/0.1 (training data collection; contact: {})"
OK_LICENSE = re.compile(r"^(CC0|Public domain|PD|CC BY \d)", re.I)          # ゆるいものだけ (SA・NC・ND は除く)
BAD_LICENSE = re.compile(r"(SA|NC|ND)", re.I)
SKIP_TITLE = re.compile(r"BioAcoustica")                                  # 昆虫・動物の鳴き声の録音集 (音楽の学習に向かない)
MIN_BYTES, MIN_SECONDS, MIN_FS = 8_000_000, 30.0, 32000


def get(url, params=None, rate=0.0):
    """url を取る。rate > 0 なら、その速度 [byte/s] を超えないように読む。拒否されたら (429 など) 間を空けて再試行する"""
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for k in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r:
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


def candidates():
    """Commons の FLAC (MIN_BYTES 以上) を、ページごとに (title, info) で返す"""
    cont = {}
    while True:
        p = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": 50,
             "gsrsearch": f"filemime:audio/flac filesize:>{MIN_BYTES // 1000}", "prop": "imageinfo",
             "iiprop": "size|url|sha1|extmetadata|mediatype", **cont}
        d = json.loads(get(API, p))
        for page in d.get("query", {}).get("pages", {}).values():
            yield page["title"], page["imageinfo"][0]
        if "continue" not in d:
            return
        cont = d["continue"]
        time.sleep(5)


def strip(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "audio", "commons"))
    ap.add_argument("--max-gb", type=float, default=30.0, help="保存する合計の上限 [GB]")
    ap.add_argument("--wait", type=float, default=30.0, help="ダウンロードの間隔 [秒]")
    ap.add_argument("--rate", type=float, default=1.0, help="ダウンロードの速度の上限 [MB/s]")
    ap.add_argument("--contact", required=True, help="User-Agent に入れる連絡先 (Wikimedia の規則)")
    a = ap.parse_args()
    pin({11})
    global UA
    UA = UA.format(a.contact)
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {"sources": [], "rejected": []}
    done = {s["sha1"] for s in man["sources"]} | {s["sha1"] for s in man["rejected"]}
    total = sum(os.path.getsize(os.path.join(a.out, s["file"])) for s in man["sources"])
    for title, ii in candidates():
        em = ii.get("extmetadata", {})
        lic = strip(em.get("LicenseShortName", {}).get("value"))
        if ii["sha1"] in done or ii["mediatype"] != "AUDIO" or not OK_LICENSE.match(lic) or BAD_LICENSE.search(lic) or SKIP_TITLE.search(title):
            continue
        if total > a.max_gb * 1e9:
            print("上限に達した", flush=True)
            break
        name = re.sub(r"[^\w.-]+", "_", title.removeprefix("File:"))[:150]
        path = os.path.join(a.out, name if name.endswith(".flac") else name + ".flac")
        time.sleep(a.wait)
        data = get(ii["url"], rate=a.rate * 1e6)
        if hashlib.sha1(data).hexdigest() != ii["sha1"]:
            print(f"sha1 が合わない: {title}", flush=True)
            continue
        open(path, "wb").write(data)
        try:
            info = sf.info(path)
            why = "" if info.channels == 2 and info.duration >= MIN_SECONDS and info.samplerate >= MIN_FS else f"{info.channels}ch {info.duration:.0f}s {info.samplerate}Hz"
        except Exception as e:  # noqa: BLE001
            info, why = None, f"読めない: {e}"
        rec = {"name": os.path.splitext(name)[0], "file": os.path.basename(path), "source_page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
               "artist": strip(em.get("Artist", {}).get("value"))[:120], "license": lic, "sha1": ii["sha1"], "sha256": hashlib.sha256(data).hexdigest()}
        if why:
            os.remove(path)
            man["rejected"].append({**rec, "reason": why})
            print(f"除外 {title}: {why}", flush=True)
        else:
            man["sources"].append({**rec, "sample_rate_hz": info.samplerate, "channels": info.channels, "subtype": info.subtype, "duration_s": round(info.duration, 1)})
            total += len(data)
            print(f"[{len(man['sources'])}] {title} {lic} {info.duration:.0f}s {total / 1e9:.1f} GB", flush=True)
        json.dump(man, open(mpath, "w"), ensure_ascii=False, indent=1)
    print(f"完了: {len(man['sources'])} 本、{total / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
