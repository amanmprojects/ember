//! Tensor operations: quantized matmul plus the small elementwise kernels a
//! transformer needs.

use rayon::prelude::*;

use crate::quant::{self, ActKind, Acts, BlockQ8, BlockQ8K, GgmlType, QK, QK_K};

/// A 2-D weight matrix borrowed from the mmapped file. Row `r` holds `cols`
/// elements in GGML block format.
#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
    row_bytes: usize,
}

impl<'a> Matrix<'a> {
    pub fn new(ty: GgmlType, rows: usize, cols: usize, data: &'a [u8]) -> Self {
        let (blk, bytes) = ty.block();
        assert_eq!(cols % blk, 0, "row length {cols} not a multiple of block {blk}");
        let row_bytes = cols / blk * bytes;
        assert_eq!(data.len(), rows * row_bytes);
        Self { ty, rows, cols, data, row_bytes }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &'a [u8] {
        &self.data[r * self.row_bytes..(r + 1) * self.row_bytes]
    }

    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        quant::dequantize(self.ty, self.row(r), out);
    }
}

/// Activations for `n` rows, quantized for a particular weight format.
/// Reused across calls to avoid reallocating every matmul.
#[derive(Default)]
pub struct ActBuf {
    q8: Vec<BlockQ8>,
    q8k: Vec<BlockQ8K>,
}

impl ActBuf {
    fn prepare<'s>(&'s mut self, kind: ActKind, x: &'s [f32], n: usize, cols: usize) -> Acts<'s> {
        match kind {
            ActKind::F32 => Acts::F32(x, cols),
            ActKind::Q8 => {
                let per = cols / QK;
                self.q8.resize(n * per, BlockQ8 { d: 0.0, s: 0.0, qs: [0; QK] });
                self.q8[..n * per]
                    .par_chunks_mut(per)
                    .zip(x.par_chunks(cols))
                    .for_each(|(o, xr)| quant::quantize_q8(xr, o));
                Acts::Q8(&self.q8[..n * per], per)
            }
            ActKind::Q8K => {
                let per = cols / QK_K;
                self.q8k.resize(n * per, BlockQ8K { d: 0.0, qs: [0; QK_K], bsums: [0; 16] });
                self.q8k[..n * per]
                    .par_chunks_mut(per)
                    .zip(x.par_chunks(cols))
                    .for_each(|(o, xr)| quant::quantize_q8k(xr, o));
                Acts::Q8K(&self.q8k[..n * per], per)
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    // A method (not field access) so closures capture the whole Send wrapper.
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// `out[t, r] = dot(w[r, :], x[t, :])` for `t < n`.
///
/// `x` is `n x w.cols`, `out` is `n x w.rows`, both row-major. Work is split
/// across threads by weight rows; each thread streams its slice of the
/// weights once and reuses every row against all `n` activation rows, so a
/// batched prefill reads the weights once instead of `n` times.
pub fn matmul(w: &Matrix, x: &[f32], n: usize, out: &mut [f32], buf: &mut ActBuf) {
    assert_eq!(x.len(), n * w.cols);
    assert_eq!(out.len(), n * w.rows);
    let acts = buf.prepare(w.ty.act_kind(), x, n, w.cols);
    let rows = w.rows;
    let threads = rayon::current_num_threads();
    // Enough chunks for load balancing across heterogeneous (P/E) cores.
    let chunk = rows.div_ceil(threads * 8).clamp(4, 256);
    let out_ptr = SendPtr(out.as_mut_ptr());

    (0..rows.div_ceil(chunk)).into_par_iter().for_each(|c| {
        let r0 = c * chunk;
        let r1 = (r0 + chunk).min(rows);
        let mut tmp = vec![0.0f32; n];
        let mut scratch = Vec::new();
        for r in r0..r1 {
            quant::row_dot(w.ty, w.row(r), acts, &mut tmp, &mut scratch);
            for (t, &v) in tmp.iter().enumerate() {
                // SAFETY: each (t, r) is written by exactly one task and is
                // in bounds by the asserts above.
                unsafe { *out_ptr.get().add(t * rows + r) = v };
            }
        }
    });
}

pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = xi * scale * wi;
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for (xi, &wi) in x.iter_mut().zip(w) {
        *xi *= scale * wi;
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Precomputed RoPE rotation table: cos/sin for every (position, pair).
pub struct Rope {
    pub dim: usize,
    pub neox: bool,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    /// `freq_factors` (Llama 3 style long-context scaling) divides each
    /// pair's frequency; `linear_scale` divides every frequency.
    pub fn new(
        dim: usize,
        max_pos: usize,
        base: f32,
        linear_scale: f32,
        neox: bool,
        freq_factors: Option<&[f32]>,
    ) -> Self {
        let half = dim / 2;
        let mut cos = vec![0.0; max_pos * half];
        let mut sin = vec![0.0; max_pos * half];
        for i in 0..half {
            let mut freq = base.powf(-2.0 * i as f32 / dim as f32) / linear_scale;
            if let Some(f) = freq_factors {
                freq /= f[i];
            }
            for p in 0..max_pos {
                let a = p as f32 * freq;
                cos[p * half + i] = a.cos();
                sin[p * half + i] = a.sin();
            }
        }
        Self { dim, neox, cos, sin }
    }

    /// Rotate the first `dim` elements of one head vector at `pos`.
    /// GGUF Llama weights are permuted so pairs are adjacent (2i, 2i+1);
    /// NeoX-style models (Qwen) pair element i with i + dim/2.
    #[inline]
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.dim / 2;
        let c = &self.cos[pos * half..(pos + 1) * half];
        let s = &self.sin[pos * half..(pos + 1) * half];
        if self.neox {
            let (a, b) = x[..self.dim].split_at_mut(half);
            for i in 0..half {
                let (x0, x1) = (a[i], b[i]);
                a[i] = x0 * c[i] - x1 * s[i];
                b[i] = x0 * s[i] + x1 * c[i];
            }
        } else {
            for i in 0..half {
                let (x0, x1) = (x[2 * i], x[2 * i + 1]);
                x[2 * i] = x0 * c[i] - x1 * s[i];
                x[2 * i + 1] = x0 * s[i] + x1 * c[i];
            }
        }
    }
}

/// Opt-in wall-clock profiler: set `EMBER_PROFILE=1` to print where forward
/// passes spend their time.
pub mod prof {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub const NAMES: [&str; 5] = ["matmul", "attention", "norm/rope/act", "lm head", "quantize"];
    pub const MATMUL: usize = 0;
    pub const ATTN: usize = 1;
    pub const MISC: usize = 2;
    pub const HEAD: usize = 3;
    static NS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

    pub fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| std::env::var_os("EMBER_PROFILE").is_some())
    }

    #[inline]
    pub fn time<R>(cat: usize, f: impl FnOnce() -> R) -> R {
        if !enabled() {
            return f();
        }
        let t = Instant::now();
        let r = f();
        NS[cat].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }

    pub fn report() {
        if !enabled() {
            return;
        }
        let total: u64 = NS.iter().map(|n| n.load(Ordering::Relaxed)).sum();
        eprintln!("profile ({:.1} ms total):", total as f64 / 1e6);
        for (name, n) in NAMES.iter().zip(&NS) {
            let n = n.load(Ordering::Relaxed);
            if n > 0 {
                eprintln!("  {name:<14} {:>9.1} ms  {:>5.1}%", n as f64 / 1e6, 100.0 * n as f64 / total as f64);
            }
        }
    }
}
