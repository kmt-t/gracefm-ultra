//! numpy の .npz（無圧縮 zip + .npy）の書き出し。

pub struct Npy {
    pub descr: &'static str,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn u8(v: &[u8], shape: &[usize]) -> Npy {
        Npy { descr: "|u1", shape: shape.to_vec(), data: v.to_vec() }
    }
    pub fn bool(v: &[bool], shape: &[usize]) -> Npy {
        Npy { descr: "|b1", shape: shape.to_vec(), data: v.iter().map(|&b| b as u8).collect() }
    }
    pub fn i8(v: &[i8], shape: &[usize]) -> Npy {
        Npy { descr: "|i1", shape: shape.to_vec(), data: v.iter().map(|&b| b as u8).collect() }
    }
    pub fn i64(v: &[i64], shape: &[usize]) -> Npy {
        Npy { descr: "<i8", shape: shape.to_vec(), data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn f32(v: &[f32], shape: &[usize]) -> Npy {
        Npy { descr: "<f4", shape: shape.to_vec(), data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let shape = match self.shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.shape[0]),
            _ => format!("({})", self.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
        };
        let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", self.descr, shape);
        let total = 10 + header.len() + 1;
        let pad = (64 - total % 64) % 64;
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        let mut out = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(header.as_bytes());
        out.extend(&self.data);
        out
    }
}

fn crc32_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
        t[i as usize] = c;
    }
    t
}

fn crc32(data: &[u8], table: &[u32; 256]) -> u32 {
    let mut c = 0xFFFFFFFFu32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFFFFFF
}

pub fn write_npz(path: &str, entries: &[(&str, Npy)]) -> std::io::Result<()> {
    use std::io::Write;
    let table = crc32_table();
    let mut out: Vec<u8> = Vec::new();
    let mut central: Vec<u8> = Vec::new();
    for (name, arr) in entries {
        let fname = format!("{}.npy", name);
        let body = arr.to_bytes();
        if body.len() >= u32::MAX as usize {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "配列が 4 GB を超えます"));
        }
        let crc = crc32(&body, &table);
        let offset = out.len() as u32;
        let size = body.len() as u32;
        out.extend(0x04034b50u32.to_le_bytes());
        out.extend([20u8, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]); // version, flags, method(0), time, date(1980-01-01)
        out.extend(crc.to_le_bytes());
        out.extend(size.to_le_bytes());
        out.extend(size.to_le_bytes());
        out.extend((fname.len() as u16).to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend(fname.as_bytes());
        out.extend(&body);
        central.extend(0x02014b50u32.to_le_bytes());
        central.extend([20u8, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]); // made, needed, flags, method, time, date
        central.extend(crc.to_le_bytes());
        central.extend(size.to_le_bytes());
        central.extend(size.to_le_bytes());
        central.extend((fname.len() as u16).to_le_bytes());
        central.extend([0u8; 12]); // extra, comment, disk, 内部属性 (各 2 byte)、外部属性 (4 byte)
        central.extend(offset.to_le_bytes());
        central.extend(fname.as_bytes());
    }
    let cd_off = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend(central);
    out.extend(0x06054b50u32.to_le_bytes());
    out.extend([0u8; 4]);
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend((entries.len() as u16).to_le_bytes());
    out.extend(cd_size.to_le_bytes());
    out.extend(cd_off.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    let mut f = std::fs::File::create(path)?;
    f.write_all(&out)
}
