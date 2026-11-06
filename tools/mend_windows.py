"""
MendNet の学習・評価の窓。data/<profile>/mend/{train,val} (gen_mend.py) の、実際の復号の結果から窓 (WIN フレーム) を切り出す。
窓は、使えない (壊れた・欠落) フレームを含むものを選ぶ。
"""
import numpy as np
import torch

from gracefm_ultra.mendnet import WIN, bit_index

MIN_USABLE = 10


class Windows:
    """区間・全シナリオを (CPU に) 保持して、窓をバッチで切り出す。resident を与えると、files のうち resident 本だけを保持し、
    swap で一部を入れ替える (学習データがメモリに収まらないとき)"""

    KEYS = ("truth", "dec", "var", "status", "gap", "hint_h", "hint_q", "llr", "bits", "step", "dstep")

    def __init__(self, files, device, resident=None, rng=None):
        self.files, self.device = list(files), device
        self.data = {k: [] for k in self.KEYS}
        self.loaded, self._bad = [], []
        n = len(self.files) if resident is None else min(resident, len(self.files))
        order = np.arange(len(self.files)) if resident is None else (rng or np.random.default_rng(0)).permutation(len(self.files))
        for fi in order[:n]:
            self._load(len(self.loaded), int(fi))
        self._rebuild()
        self.comps = np.flatnonzero(self.bit_ok.numpy().any(1))                   # 送っている成分

    def _load(self, slot, fi):
        """files[fi] を slot に読み込む (slot が末尾なら追加)"""
        z = np.load(self.files[fi])
        for k in self.data:
            v = torch.from_numpy(z[k])
            if slot == len(self.loaded):
                self.data[k].append(v)
            else:
                self.data[k][slot] = v
        if not hasattr(self, "bit_idx"):
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
        keys = ("coef", "var", "status", "gap", "comp", "target", "hint_h", "hint_q", "llr", "bits", "dstep", "step")
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
            bi = self.bit_idx[ch]
            out["llr"].append(self.data["llr"][si][sc, a:a + WIN][:, bi] * self.bit_ok[ch])
            out["bits"].append(self.data["bits"][si][a:a + WIN][:, bi])
            out["dstep"].append(self.data["dstep"][si][sc, a:a + WIN, sl])
            out["step"].append(self.data["step"][si][a:a + WIN, sl])
        batch = {k: torch.stack(v).to(self.device) for k, v in out.items()}
        batch["ok"] = self.bit_ok.to(self.device)[batch["comp"]][:, None, :]
        batch["pos"] = self.bit_pos.to(self.device)[batch["comp"]]
        batch["coef_of_bit"] = self.bit_coef.to(self.device)[batch["comp"]]
        return batch

    def batch(self, bs, rng, anywhere=False):
        """窓を bs 個、成分は窓ごとにランダムで切り出す"""
        locs = self.locate(bs, rng, anywhere)
        return self.gather(locs, rng.choice(self.comps, size=bs))
