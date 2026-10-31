"""
評価・測定用の素材 (GraceFM の eval_audio 以下) の読み込み。32 kHz、振幅 0.5 倍のステレオを返す。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(HERE, "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
sys.path.insert(0, os.path.join(HERE, ".."))
from tools.common import wavio  # noqa: E402

FS = 32000

MATERIALS = {
    "piano": ("lossless/Emotional_piano.wav", 2.0),
    "percussion": ("lossless/Ceramic_Djembe.wav", 1.0),
    "guitar": ("lossless/344211_giomilko_c-major-9-bossa-nova-guitar.wav", 1.0),
    "speech": ("libri1.ogg", 0.3),
    "demo": ("@demo", 0.0),
    "ambient": ("train_lossless/Raspberrymusic_-_Ambient_(10_minutes).flac", 60.0),
}
FOUR = ["piano", "percussion", "guitar", "speech"]


def load(name, seconds):
    rel, start = MATERIALS[name]
    path = os.path.join(GRACEFM, "eval_audio", rel)
    if rel == "@demo":                                              # GraceFM の合成試験信号 (48 kHz)
        from tools.common.demo_signal import make_demo
        x = make_demo(fs=48000, dur=10.0)
        x = np.stack(x, axis=1) if isinstance(x, tuple) else x
        fs = 48000
        x = wavio.resample(x, fs, FS)
        return x[int(start * FS):int((start + seconds) * FS)] * 0.5
    if path.endswith(".ogg") or path.endswith(".flac"):
        import soundfile as sf
        info = sf.info(path)
        first = int(start * info.samplerate) if rel.startswith("train_lossless") else 0              # 長い曲は途中から使う
        x, fs = sf.read(path, dtype="float64", always_2d=True, start=first, frames=int((seconds + 2) * info.samplerate))
        x = x[:, :2] if x.shape[1] > 1 else np.repeat(x, 2, axis=1)
    else:
        x, fs = wavio.read(path)
    x = wavio.resample(x, fs, FS)
    a0 = 0 if rel.startswith("train_lossless") else int(start * FS)
    return x[a0:a0 + int(seconds * FS)] * 0.5
