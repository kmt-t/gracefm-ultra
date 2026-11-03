#!/usr/bin/env python3
"""
MendNet の学習。data/<profile>/mend/{train,val} (gen_hn.py --mend) の、実際の復号の結果から窓 (WIN フレーム) を切り出す。
窓は、使えない (壊れた・欠落) フレームを含むものを選ぶ。損失は使えないフレームの帯域ごとの相対二乗誤差 + 接続部の変化量の誤差。

  .venv/bin/python tools/train_mend.py --profile B200 --minutes 30 --out runs/mend_B200.pt
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gracefm_ultra.cpus import pin  # noqa: E402
from gracefm_ultra.mendnet import C_IN, C_IN_HINT, WIN, MendNet, bit_index, build_input, mend_loss  # noqa: E402

MIN_USABLE = 10


class Windows:
    """区間・全シナリオを (CPU に) 保持して、窓をバッチで切り出す。resident を与えると、files のうち resident 本だけを保持し、
    swap で一部を入れ替える (学習データがメモリに収まらないとき)"""

    KEYS = ("truth", "dec", "var", "status", "gap", "hint_h", "hint_q")
    BIT_KEYS = ("llr", "bits", "step", "dstep")

    def __init__(self, files, device, bits=False, resident=None, rng=None):
        self.files, self.bits, self.device = list(files), bits, device
        keys = self.KEYS + (self.BIT_KEYS if bits else ())
        self.data = {k: [] for k in keys}
        self.loaded, self._bad = [], []
        n = len(self.files) if resident is None else min(resident, len(self.files))
        order = np.arange(len(self.files)) if resident is None else (rng or np.random.default_rng(0)).permutation(len(self.files))
        for fi in order[:n]:
            self._load(len(self.loaded), int(fi))
        self._rebuild()
        self.ncomp = self.data["var"][0].shape[2]

    def _load(self, slot, fi):
        """files[fi] を slot に読み込む (slot が末尾なら追加)"""
        z = np.load(self.files[fi])
        for k in self.data:
            v = torch.from_numpy(z[k])
            if slot == len(self.loaded):
                self.data[k].append(v)
            else:
                self.data[k][slot] = v
        if self.bits and not hasattr(self, "bit_idx"):
            idx, ok = bit_index(z["coef_of_bit"])
            self.bit_idx, self.bit_ok = torch.from_numpy(idx), torch.from_numpy(ok)
            self.bit_pos = torch.from_numpy(z["pos"][idx])                       # (2, NB) 仮数内の位置 (大きいほど上位)
            self.bit_coef = torch.from_numpy(z["coef_of_bit"][idx] % 64)         # (2, NB) 成分内の係数の番号
        s, fr = np.nonzero(z["status"] >= 2)
        bad = np.stack([np.full_like(s, slot), s, fr], 1)
        if slot == len(self.loaded):
            self.loaded.append(fi)
            self._bad.append(bad)
        else:
            self.loaded[slot] = fi
            self._bad[slot] = bad

    def _rebuild(self):
        self.bad = np.concatenate(self._bad)                       # (区間, シナリオ, フレーム) の、使えないフレーム
        self.truth, self.dec, self.var, self.status = self.data["truth"], self.data["dec"], self.data["var"], self.data["status"]

    def swap(self, k, rng):
        """保持している区間のうち k 本を、保持していない区間と入れ替える"""
        out = [i for i in range(len(self.files)) if i not in set(self.loaded)]
        if not out:
            return
        for slot, fi in zip(rng.choice(len(self.loaded), min(k, len(out)), replace=False), rng.choice(out, min(k, len(out)), replace=False)):
            self._load(int(slot), int(fi))
        self._rebuild()

    def locate(self, bs, rng, anywhere=False):
        """窓の位置 [(区間, シナリオ, 先頭フレーム), …] を bs 個選ぶ"""
        locs = []
        while len(locs) < bs:
            si, sc, fr = self.bad[rng.integers(len(self.bad))]
            if anywhere and rng.random() < 0.5:                  # 使えないフレームのない窓も (使えるフレームの補正の学習)
                si = int(rng.integers(len(self.truth)))
                sc, fr = int(rng.integers(self.dec[si].shape[0])), int(rng.integers(self.truth[si].shape[0]))
            a = int(np.clip(fr - rng.integers(0, WIN), 0, self.truth[si].shape[0] - WIN))
            if (self.status[si][sc, a:a + WIN] <= 1).sum() >= MIN_USABLE:
                locs.append((int(si), int(sc), a))
        return locs

    def gather(self, locs, comps):
        """locs の窓を、窓ごとの成分 comps (0 = M、1 = S) で切り出して積む"""
        keys = ("coef", "var", "status", "gap", "comp", "target", "hint_h", "hint_q") + (("llr", "bits", "dstep", "step") if self.bits else ())
        out = {k: [] for k in keys}
        for (si, sc, a), ch in zip(locs, comps):
            sl = slice(ch * 64, ch * 64 + 64)
            out["coef"].append(self.data["dec"][si][sc, a:a + WIN, sl])
            out["target"].append(self.data["truth"][si][a:a + WIN, sl])
            out["var"].append(self.data["var"][si][sc, a:a + WIN, ch])
            out["status"].append(self.data["status"][si][sc, a:a + WIN])
            out["gap"].append(self.data["gap"][si][sc, a:a + WIN])
            out["hint_h"].append(self.data["hint_h"][si][sc, a:a + WIN, sl])
            out["hint_q"].append(self.data["hint_q"][si][sc, a:a + WIN, ch])
            out["comp"].append(torch.tensor(ch))
            if self.bits:
                bi = self.bit_idx[ch]
                out["llr"].append(self.data["llr"][si][sc, a:a + WIN][:, bi] * self.bit_ok[ch])
                out["bits"].append(self.data["bits"][si][a:a + WIN][:, bi])
                out["dstep"].append(self.data["dstep"][si][sc, a:a + WIN, sl])
                out["step"].append(self.data["step"][si][a:a + WIN, sl])
        batch = {k: torch.stack(v).to(self.device) for k, v in out.items()}
        if self.bits:
            batch["ok"] = self.bit_ok.to(self.device)[batch["comp"]][:, None, :]
            batch["pos"] = self.bit_pos.to(self.device)[batch["comp"]]
            batch["coef_of_bit"] = self.bit_coef.to(self.device)[batch["comp"]]
        return batch

    def batch(self, bs, rng, anywhere=False):
        """窓を bs 個、成分は窓ごとにランダムで切り出す"""
        locs = self.locate(bs, rng, anywhere)
        return self.gather(locs, rng.integers(self.ncomp, size=bs))


def forward(net, b, hint, refine):
    x, sig, usable = build_input(b["coef"], b["var"], b["status"].long(), b["gap"], b["comp"], *((b["hint_h"], b["hint_q"]) if hint else ()))
    y = net(x)
    return mend_loss(y, b["target"], sig, usable, b["coef"], refine=refine)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="B200")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", default="", help="このチェックポイントの重みから始める")
    ap.add_argument("--hint", action="store_true", help="v2: 壊れたフレームの手がかりを入力に足す")
    ap.add_argument("--refine", action="store_true", help="使えるフレームの係数も補正する (HaarNet の役割を持たせる)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "runs", f"mend_{a.profile}.pt")
    pin()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    d = os.path.join(a.data, a.profile, "mend")
    tr = Windows(sorted(glob.glob(os.path.join(d, "train", "*.npz"))), dev)
    va = Windows(sorted(glob.glob(os.path.join(d, "val", "*.npz"))), dev)
    print(f"使えないフレーム 学習 {len(tr.bad)}、検証 {len(va.bad)}", flush=True)
    cfg = dict(c_in=C_IN_HINT if a.hint else C_IN, ch=192, blocks=12)
    net = MendNet(**cfg).to(dev)
    print(f"パラメータ {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev)["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    vrng = np.random.default_rng(1)
    vb = [va.batch(a.bs, vrng) for _ in range(8)]
    t0, step, run = time.time(), 0, []
    total = a.minutes * 60
    while time.time() - t0 < total:
        frac = (time.time() - t0) / total
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * frac)) * 0.97 + 0.03)
        loss, _ = forward(net, tr.batch(a.bs, rng, a.refine), a.hint, a.refine)
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.mean().item())
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                net.eval()
                v = torch.cat([forward(net, b, a.hint, a.refine)[1] for b in vb]).mean(0).cpu().numpy()
                net.train()
            print(f"step {step:5d}  {time.time() - t0:5.0f}s  loss {np.mean(run[-100:]):.4f}  検証 帯域別 {np.round(v, 3)}", flush=True)
        if step % 500 == 0:
            torch.save({"model": net.state_dict(), "cfg": cfg, "hint": a.hint, "refine": a.refine, "step": step}, out)
    torch.save({"model": net.state_dict(), "cfg": cfg, "hint": a.hint, "refine": a.refine, "step": step}, out)
    print(f"終了 step {step}", flush=True)


if __name__ == "__main__":
    main()
