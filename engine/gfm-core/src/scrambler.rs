//! スクランブル: 送信の `scramble`（ビット）と、受信の `descramble`（LLR と硬判定）。同じ PRBS を使う。
//! 系列は 1 + x^14 + x^15、フレームごとに先頭から。詰め物も含めてスクランブルする。

use crate::modem::prbs;

pub struct Scrambler {
    seq: Vec<u8>,
}

impl Scrambler {
    pub fn new(n: usize) -> Scrambler {
        Scrambler { seq: prbs(n) }
    }

    /// 送信: ビット列（詰め物込み）に PRBS を XOR する
    pub fn scramble(&self, bits: &mut [u8]) {
        for (b, s) in bits.iter_mut().zip(&self.seq) {
            *b ^= *s;
        }
    }

    /// 受信: 復調した LLR と硬判定のスクランブルを戻す（PRBS が 1 の位置で LLR の符号と硬判定を反転する）
    pub fn descramble(&self, llr: &mut [i64], hard: &mut [u8]) {
        for i in 0..llr.len() {
            if self.seq[i] == 1 {
                llr[i] = -llr[i];
            }
            hard[i] ^= self.seq[i];
        }
    }

    /// 単独の LLR の符号を戻す（位置 i）
    pub fn sign(&self, i: usize) -> bool {
        self.seq[i] == 1
    }
}
