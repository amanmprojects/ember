//! GGUF file parser.
//!
//! The file is memory-mapped; tensor data is never copied. A GGUF file is:
//!
//! ```text
//! magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
//! n_kv x (key: string, type: u32, value)
//! n_tensors x (name: string, n_dims: u32, dims: [u64; n_dims], type: u32, offset: u64)
//! padding to `general.alignment` (default 32)
//! tensor data
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::quant::GgmlType;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::I8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::I16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::I32(v) => v as u64,
            Value::U64(v) => v,
            Value::I64(v) => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::String(s) => {
                let short: String = s.chars().take(80).collect();
                if short.len() < s.len() {
                    write!(f, "{short:?}... ({} chars)", s.chars().count())
                } else {
                    write!(f, "{s:?}")
                }
            }
            Value::Array(a) => {
                let head: Vec<String> = a.iter().take(6).map(|v| v.to_string()).collect();
                if a.len() > 6 {
                    write!(f, "[{}, ... ({} items)]", head.join(", "), a.len())
                } else {
                    write!(f, "[{}]", head.join(", "))
                }
            }
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: dims[0] is the innermost (contiguous) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset into the file.
    pub offset: usize,
    pub size: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    kv_index: HashMap<String, usize>,
    tensor_index: HashMap<String, usize>,
    mmap: Mmap,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!("unexpected end of file at offset {}", self.pos);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.arr::<1>()?[0]),
            1 => Value::I8(self.arr::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => Value::Bool(self.arr::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(elem_ty)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            t => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is treated as read-only; if it is modified while
        // mapped we may read garbage, which is the standard mmap caveat.
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::WillNeed);

        let mut r = Reader { buf: &mmap, pos: 0 };
        if r.take(4)? != b"GGUF" {
            bail!("{} is not a GGUF file", path.display());
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        let mut metadata = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let val = r.value(ty).with_context(|| format!("reading metadata {key}"))?;
            metadata.push((key, val));
        }
        let kv_index: HashMap<_, _> = metadata.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect();

        let alignment = kv_index.get("general.alignment").and_then(|&i| metadata[i].1.as_u64()).unwrap_or(32) as usize;

        let mut raw = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ty_id = r.u32()?;
            let ty =
                GgmlType::from_id(ty_id).with_context(|| format!("tensor {name}: unsupported ggml type {ty_id}"))?;
            let offset = r.u64()? as usize;
            raw.push((name, dims, ty, offset));
        }
        let data_start = r.pos.div_ceil(alignment) * alignment;

        let mut tensors = Vec::with_capacity(n_tensors);
        for (name, dims, ty, offset) in raw {
            let n: u64 = dims.iter().product();
            let (blk, bytes) = ty.block();
            if !(n as usize).is_multiple_of(blk) {
                bail!("tensor {name}: {n} elements not divisible by block size {blk}");
            }
            let size = n as usize / blk * bytes;
            let offset = data_start + offset;
            if offset + size > mmap.len() {
                bail!("tensor {name} extends past end of file");
            }
            tensors.push(TensorInfo { name, dims, ty, offset, size });
        }
        let tensor_index = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();

        Ok(Self { version, metadata, tensors, kv_index, tensor_index, mmap })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv_index.get(key).map(|&i| &self.metadata[i].1)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(Value::as_f32)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.size]
    }

    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }
}
