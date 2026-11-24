//! Quantized block formats and the dot-product kernels that run on them.
//!
//! Weights stay in their on-disk GGML block format. Activations are quantized
//! on the fly to 8-bit blocks so each weight block can be dotted against them
//! with integer arithmetic, the same scheme llama.cpp uses:
//!
//! * 32-element formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) dot against [`BlockQ8`].
//! * 256-element "k-quants" (Q2_K..Q6_K) dot against [`BlockQ8K`].
//! * Float formats (F32, F16, BF16) dot against plain f32.
//!
//! Every quantized format is handled by one *decoder* that unpacks a block into
//! small integers plus scales. The same decoder feeds both the integer dot
//! product and the f32 dequantizer, so the two can't disagree.
//!
//! Decoders emit *unsigned* integers wherever the format allows (everything
//! but Q8_0) and express the signed offset as a per-block minimum instead:
//! Q4_0's `d * (q - 8)` becomes `d * q + (-8d)`. The offset is applied once
//! per block using precomputed activation sums, and the inner loop can feed
//! the unsigned weights straight into AVX2's `maddubs` (u8 x i8) without the
//! sign-fixing instructions a signed x signed product needs.

use half::{bf16, f16};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
}

pub const QK: usize = 32;
pub const QK_K: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActKind {
    F32,
    Q8,
    Q8K,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            30 => BF16,
            _ => return None,
        })
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        use GgmlType::*;
        match self {
            F32 => (1, 4),
            F16 | BF16 => (1, 2),
            Q4_0 => (QK, 18),
            Q4_1 => (QK, 20),
            Q5_0 => (QK, 22),
            Q5_1 => (QK, 24),
            Q8_0 => (QK, 34),
            Q2_K => (QK_K, 84),
            Q3_K => (QK_K, 110),
            Q4_K => (QK_K, 144),
            Q5_K => (QK_K, 176),
            Q6_K => (QK_K, 210),
        }
    }

    pub fn act_kind(self) -> ActKind {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 => ActKind::F32,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 => ActKind::Q8,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K => ActKind::Q8K,
        }
    }

    pub fn bits_per_weight(self) -> f32 {
        let (n, b) = self.block();
        b as f32 * 8.0 / n as f32
    }
}

#[inline(always)]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

// ---------------------------------------------------------------------------
// Activation blocks
// ---------------------------------------------------------------------------

/// 32 activations quantized to i8. `s = d * sum(qs)` lets formats with a
/// per-block minimum (Q4_1, Q5_1) fold the offset in with one multiply.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct BlockQ8 {
    pub d: f32,
    pub s: f32,
    pub qs: [i8; QK],
}

/// 256 activations quantized to i8, with sums of each 16-element group so
/// k-quant minimums can be applied without touching `qs` again.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; 16],
}

pub fn quantize_q8(x: &[f32], out: &mut [BlockQ8]) {
    for (xb, ob) in x.chunks_exact(QK).zip(out.iter_mut()) {
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut sum = 0i32;
        for (q, &v) in ob.qs.iter_mut().zip(xb) {
            *q = (v * id).round() as i8;
            sum += *q as i32;
        }
        ob.d = d;
        ob.s = d * sum as f32;
    }
}

pub fn quantize_q8k(x: &[f32], out: &mut [BlockQ8K]) {
    for (xb, ob) in x.chunks_exact(QK_K).zip(out.iter_mut()) {
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (q, &v) in ob.qs.iter_mut().zip(xb) {
            *q = (v * id).round() as i8;
        }
        for (j, s) in ob.bsums.iter_mut().enumerate() {
            *s = ob.qs[16 * j..16 * j + 16].iter().map(|&q| q as i16).sum();
        }
        ob.d = d;
    }
}

// ---------------------------------------------------------------------------
// Block decoders
// ---------------------------------------------------------------------------

/// 32 unpacked weights: a SIMD register when AVX2 is on, so decoders feed the
/// dot product directly without a round trip through memory.
#[cfg(target_feature = "avx2")]
type Q32 = std::arch::x86_64::__m256i;
#[cfg(not(target_feature = "avx2"))]
type Q32 = [i8; QK];

#[inline(always)]
fn q32_array(q: Q32) -> [i8; QK] {
    #[cfg(target_feature = "avx2")]
    {
        let mut out = [0i8; QK];
        unsafe { std::arch::x86_64::_mm256_storeu_si256(out.as_mut_ptr() as *mut Q32, q) };
        out
    }
    #[cfg(not(target_feature = "avx2"))]
    q
}

/// A decoded 32-element block: value = d * q + m. `q` is unsigned (0..31)
/// for every format except Q8_0.
struct Dec32 {
    q: Q32,
    d: f32,
    m: f32,
}

/// AVX2 unpacking for the 32-element formats: 16 bytes of nibbles become 32
/// bytes (low nibbles first), and the 32 high bits of Q5 formats become a
/// byte mask (0xFF where bit i is set).
#[cfg(target_feature = "avx2")]
mod unpack {
    pub use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn nibbles(qs: &[u8]) -> __m256i {
        assert!(qs.len() >= 16);
        unsafe {
            let t = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
            let b = _mm256_set_m128i(_mm_srli_epi16(t, 4), t);
            _mm256_and_si256(b, _mm256_set1_epi8(0x0F))
        }
    }

    #[inline(always)]
    pub unsafe fn high_bits(qh: u32) -> __m256i {
        unsafe {
            let shuf =
                _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
            let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), shuf);
            let bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
            let mask = _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
            _mm256_and_si256(mask, _mm256_set1_epi8(0x10))
        }
    }

    #[inline(always)]
    pub unsafe fn hsum_ps(v: __m256) -> f32 {
        unsafe {
            let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }
}

/// Element j < 16 is the low nibble of byte j; element j + 16 the high one.
/// Q5 formats add bit j of `qh` as the fifth bit of element j.
#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn unpack32(qs: &[u8], qh: u32) -> [i8; QK] {
    let mut q = [0i8; QK];
    for j in 0..16 {
        let h0 = (((qh >> j) << 4) & 0x10) as u8;
        let h1 = ((qh >> (j + 12)) & 0x10) as u8;
        q[j] = ((qs[j] & 0xF) | h0) as i8;
        q[j + 16] = ((qs[j] >> 4) | h1) as i8;
    }
    q
}

#[inline(always)]
fn dec_q4_0(b: &[u8]) -> Dec32 {
    #[cfg(target_feature = "avx2")]
    let q = unsafe { unpack::nibbles(&b[2..18]) };
    #[cfg(not(target_feature = "avx2"))]
    let q = unpack32(&b[2..18], 0);
    let d = f16_at(b, 0);
    Dec32 { q, d, m: -8.0 * d }
}

#[inline(always)]
fn dec_q4_1(b: &[u8]) -> Dec32 {
    #[cfg(target_feature = "avx2")]
    let q = unsafe { unpack::nibbles(&b[4..20]) };
    #[cfg(not(target_feature = "avx2"))]
    let q = unpack32(&b[4..20], 0);
    Dec32 { q, d: f16_at(b, 0), m: f16_at(b, 2) }
}

#[inline(always)]
fn dec_q5_0(b: &[u8]) -> Dec32 {
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    #[cfg(target_feature = "avx2")]
    let q = unsafe { unpack::_mm256_or_si256(unpack::nibbles(&b[6..22]), unpack::high_bits(qh)) };
    #[cfg(not(target_feature = "avx2"))]
    let q = unpack32(&b[6..22], qh);
    let d = f16_at(b, 0);
    Dec32 { q, d, m: -16.0 * d }
}

#[inline(always)]
fn dec_q5_1(b: &[u8]) -> Dec32 {
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    #[cfg(target_feature = "avx2")]
    let q = unsafe { unpack::_mm256_or_si256(unpack::nibbles(&b[8..24]), unpack::high_bits(qh)) };
    #[cfg(not(target_feature = "avx2"))]
    let q = unpack32(&b[8..24], qh);
    Dec32 { q, d: f16_at(b, 0), m: f16_at(b, 2) }
}

/// The one signed 32-element format.
#[inline(always)]
fn dec_q8_0(b: &[u8]) -> Dec32 {
    let qs = &b[2..34];
    #[cfg(target_feature = "avx2")]
    let q = unsafe { unpack::_mm256_loadu_si256(qs.as_ptr() as *const Q32) };
    #[cfg(not(target_feature = "avx2"))]
    let q: [i8; QK] = std::array::from_fn(|j| qs[j] as i8);
    Dec32 { q, d: f16_at(b, 0), m: 0.0 }
}

/// A decoded 256-element k-quant block, `q` unsigned:
/// value[i] = d * sc[i/16] * q[i] - dmin * mn[i/16].
/// Formats with signed values (Q3_K, Q6_K) store their offset in `mn`.
struct DecK {
    q: [u8; QK_K],
    sc: [i16; 16],
    mn: [i16; 16],
    d: f32,
    dmin: f32,
}

#[inline(always)]
fn scale_min_k4(j: usize, s: &[u8]) -> (i16, i16) {
    if j < 4 {
        ((s[j] & 63) as i16, (s[j + 4] & 63) as i16)
    } else {
        (((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4)) as i16, ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as i16)
    }
}

#[inline(always)]
fn dec_q2_k(b: &[u8]) -> DecK {
    let scales = &b[0..16];
    let qs = &b[16..80];
    let mut q = [0u8; QK_K];
    let mut sc = [0i16; 16];
    let mut mn = [0i16; 16];
    for j in 0..16 {
        sc[j] = (scales[j] & 0xF) as i16;
        mn[j] = (scales[j] >> 4) as i16;
    }
    // Two halves of 128; each uses 32 bytes of qs, 2 bits per value at 4 shifts.
    for half in 0..2 {
        let src = &qs[32 * half..32 * half + 32];
        for shift in 0..4 {
            let base = 128 * half + 32 * shift;
            for l in 0..32 {
                q[base + l] = (src[l] >> (2 * shift)) & 3;
            }
        }
    }
    DecK { q, sc, mn, d: f16_at(b, 80), dmin: f16_at(b, 82) }
}

#[inline(always)]
fn dec_q3_k(b: &[u8]) -> DecK {
    let hmask = &b[0..32];
    let qs = &b[32..96];
    let s = &b[96..108];
    let d = f16_at(b, 108);

    let mut aux = [
        u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        u32::from_le_bytes([s[4], s[5], s[6], s[7]]),
        u32::from_le_bytes([s[8], s[9], s[10], s[11]]),
        0,
    ];
    const KM1: u32 = 0x0303_0303;
    const KM2: u32 = 0x0f0f_0f0f;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
    aux[3] = ((aux[1] >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
    aux[0] = (aux[0] & KM2) | ((tmp & KM1) << 4);
    aux[1] = (aux[1] & KM2) | (((tmp >> 2) & KM1) << 4);
    let mut sc = [0i16; 16];
    let mut mn = [0i16; 16];
    for j in 0..16 {
        sc[j] = ((aux[j / 4] >> (8 * (j % 4))) & 0xFF) as i8 as i16 - 32;
        mn[j] = 4 * sc[j];
    }

    // q = low 2 bits + 4 * high bit, in 0..8; the true value is q - 4.
    let mut q = [0u8; QK_K];
    for half in 0..2 {
        let src = &qs[32 * half..32 * half + 32];
        for shift in 0..4 {
            let base = 128 * half + 32 * shift;
            let bit = 4 * half + shift;
            for l in 0..32 {
                q[base + l] = ((src[l] >> (2 * shift)) & 3) | (((hmask[l] >> bit) & 1) << 2);
            }
        }
    }
    DecK { q, sc, mn, d, dmin: d }
}

#[inline(always)]
fn dec_q4_k(b: &[u8]) -> DecK {
    let scales = &b[4..16];
    let qs = &b[16..144];
    let mut q = [0u8; QK_K];
    let mut sc = [0i16; 16];
    let mut mn = [0i16; 16];
    for j in 0..8 {
        let (s, m) = scale_min_k4(j, scales);
        sc[2 * j] = s;
        sc[2 * j + 1] = s;
        mn[2 * j] = m;
        mn[2 * j + 1] = m;
    }
    for c in 0..4 {
        let src = &qs[32 * c..32 * c + 32];
        for l in 0..32 {
            q[64 * c + l] = src[l] & 0xF;
            q[64 * c + 32 + l] = src[l] >> 4;
        }
    }
    DecK { q, sc, mn, d: f16_at(b, 0), dmin: f16_at(b, 2) }
}

#[inline(always)]
fn dec_q5_k(b: &[u8]) -> DecK {
    let scales = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    let mut q = [0u8; QK_K];
    let mut sc = [0i16; 16];
    let mut mn = [0i16; 16];
    for j in 0..8 {
        let (s, m) = scale_min_k4(j, scales);
        sc[2 * j] = s;
        sc[2 * j + 1] = s;
        mn[2 * j] = m;
        mn[2 * j + 1] = m;
    }
    for c in 0..4 {
        let src = &qs[32 * c..32 * c + 32];
        for l in 0..32 {
            q[64 * c + l] = (src[l] & 0xF) | (((qh[l] >> (2 * c)) & 1) << 4);
            q[64 * c + 32 + l] = (src[l] >> 4) | (((qh[l] >> (2 * c + 1)) & 1) << 4);
        }
    }
    DecK { q, sc, mn, d: f16_at(b, 0), dmin: f16_at(b, 2) }
}

#[inline(always)]
fn dec_q6_k(b: &[u8]) -> DecK {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let scales = &b[192..208];
    let d = f16_at(b, 208);
    let mut q = [0u8; QK_K];
    let mut sc = [0i16; 16];
    let mut mn = [0i16; 16];
    for j in 0..16 {
        sc[j] = scales[j] as i8 as i16;
        mn[j] = 32 * sc[j]; // 6-bit values are stored offset by 32
    }
    for half in 0..2 {
        let ql = &ql[64 * half..64 * half + 64];
        let qh = &qh[32 * half..32 * half + 32];
        let y = &mut q[128 * half..128 * half + 128];
        for l in 0..32 {
            y[l] = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
            y[l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            y[l + 64] = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            y[l + 96] = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
        }
    }
    DecK { q, sc, mn, d, dmin: d }
}

// ---------------------------------------------------------------------------
// Dot products
// ---------------------------------------------------------------------------
//
// A weight row is dotted against up to 4 activation rows at once ("token
// tiling"). Each weight block is decoded once per tile and the per-token
// partial sums live in registers for the whole row.

/// Split `out` (one entry per activation row) into tiles of up to 4 and run
/// `tile::<T>` on each, so `T` is a compile-time constant in the kernel.
macro_rules! tiled {
    ($out:expr, $per:expr, $a:expr, |$t:ident, $o:ident, $at:ident| $body:expr) => {{
        let n = $out.len();
        let mut t0 = 0;
        while t0 < n {
            let tn = (n - t0).min(4);
            let $o = &mut $out[t0..t0 + tn];
            let $at = &$a[t0 * $per..];
            match tn {
                4 => {
                    const $t: usize = 4;
                    $body
                }
                3 => {
                    const $t: usize = 3;
                    $body
                }
                2 => {
                    const $t: usize = 2;
                    $body
                }
                _ => {
                    const $t: usize = 1;
                    $body
                }
            }
            t0 += tn;
        }
    }};
}

#[inline(always)]
fn row_dot32<const SIGNED: bool>(
    w: &[u8],
    bs: usize,
    a: &[BlockQ8],
    per: usize,
    out: &mut [f32],
    dec: impl Fn(&[u8]) -> Dec32 + Copy,
) {
    tiled!(out, per, a, |T, o, at| tile32::<SIGNED, T>(w, bs, at, per, o, dec));
}

#[inline(always)]
fn tile32<const SIGNED: bool, const T: usize>(
    w: &[u8],
    bs: usize,
    a: &[BlockQ8],
    per: usize,
    out: &mut [f32],
    dec: impl Fn(&[u8]) -> Dec32,
) {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use unpack::*;
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); T];
        let mut macc = [0.0f32; T];
        for (b, wb) in w.chunks_exact(bs).enumerate() {
            let blk = dec(wb);
            // maddubs multiplies unsigned x signed. Unsigned formats go in
            // as-is; Q8_0 moves the weight's sign onto the activation.
            let wq = if SIGNED { _mm256_sign_epi8(blk.q, blk.q) } else { blk.q };
            for t in 0..T {
                let ab = &a[t * per + b];
                let mut y = _mm256_loadu_si256(ab.qs.as_ptr() as *const __m256i);
                if SIGNED {
                    y = _mm256_sign_epi8(y, blk.q);
                }
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(wq, y), ones);
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(blk.d * ab.d), acc[t]);
                if !SIGNED {
                    macc[t] += blk.m * ab.s;
                }
            }
        }
        for t in 0..T {
            out[t] = hsum_ps(acc[t]) + macc[t];
        }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        let _ = SIGNED;
        out.fill(0.0);
        for (b, wb) in w.chunks_exact(bs).enumerate() {
            let blk = dec(wb);
            for t in 0..T {
                let ab = &a[t * per + b];
                let dot: i32 = blk.q.iter().zip(&ab.qs).map(|(&x, &y)| x as i32 * y as i32).sum();
                out[t] += blk.d * ab.d * dot as f32 + blk.m * ab.s;
            }
        }
    }
}

#[inline(always)]
fn row_dot_k(w: &[u8], bs: usize, a: &[BlockQ8K], per: usize, out: &mut [f32], dec: impl Fn(&[u8]) -> DecK + Copy) {
    tiled!(out, per, a, |T, o, at| tile_k::<T>(w, bs, at, per, o, dec));
}

#[inline(always)]
fn tile_k<const T: usize>(
    w: &[u8],
    bs: usize,
    a: &[BlockQ8K],
    per: usize,
    out: &mut [f32],
    dec: impl Fn(&[u8]) -> DecK,
) {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use unpack::*;
        let mut acc = [_mm256_setzero_ps(); T];
        for (b, wb) in w.chunks_exact(bs).enumerate() {
            let blk = dec(wb);
            let mut ia = [_mm256_setzero_si256(); T];
            for c in 0..8 {
                let qv = _mm256_loadu_si256(blk.q.as_ptr().add(32 * c) as *const __m256i);
                // i16 lanes 0..8 belong to 16-group 2c, lanes 8..16 to 2c+1.
                let s = _mm256_set_m128i(_mm_set1_epi16(blk.sc[2 * c + 1]), _mm_set1_epi16(blk.sc[2 * c]));
                for t in 0..T {
                    let y = _mm256_loadu_si256(a[t * per + b].qs.as_ptr().add(32 * c) as *const __m256i);
                    ia[t] = _mm256_add_epi32(ia[t], _mm256_madd_epi16(_mm256_maddubs_epi16(qv, y), s));
                }
            }
            let mn = _mm256_loadu_si256(blk.mn.as_ptr() as *const __m256i);
            for t in 0..T {
                let ab = &a[t * per + b];
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia[t]), _mm256_set1_ps(blk.d * ab.d), acc[t]);
                let bsums = _mm256_loadu_si256(ab.bsums.as_ptr() as *const __m256i);
                let m = _mm256_cvtepi32_ps(_mm256_madd_epi16(mn, bsums));
                acc[t] = _mm256_fnmadd_ps(m, _mm256_set1_ps(blk.dmin * ab.d), acc[t]);
            }
        }
        for t in 0..T {
            out[t] = hsum_ps(acc[t]);
        }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        out.fill(0.0);
        for (b, wb) in w.chunks_exact(bs).enumerate() {
            let blk = dec(wb);
            for t in 0..T {
                let ab = &a[t * per + b];
                let (mut isum, mut msum) = (0i32, 0i32);
                for g in 0..16 {
                    let r = 16 * g..16 * g + 16;
                    let d: i32 = blk.q[r.clone()].iter().zip(&ab.qs[r]).map(|(&x, &y)| x as i32 * y as i32).sum();
                    isum += blk.sc[g] as i32 * d;
                    msum += blk.mn[g] as i32 * ab.bsums[g] as i32;
                }
                out[t] += ab.d * (blk.d * isum as f32 - blk.dmin * msum as f32);
            }
        }
    }
}

#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // 8 independent accumulators so the compiler can keep a full vector
    // register of partial sums instead of a serial dependency chain.
    let mut acc = [0.0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] = x[i].mul_add(y[i], acc[i]);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Activations for `n` rows, already quantized into the format the weight
/// type wants. `per` is the number of blocks (or floats) per row.
#[derive(Clone, Copy)]
pub enum Acts<'a> {
    F32(&'a [f32], usize),
    Q8(&'a [BlockQ8], usize),
    Q8K(&'a [BlockQ8K], usize),
}

/// `out[t] = dot(w_row, acts[t])` for every activation row `t < out.len()`.
/// `scratch` holds a dequantized row for the float formats.
pub fn row_dot(ty: GgmlType, w: &[u8], acts: Acts, out: &mut [f32], scratch: &mut Vec<f32>) {
    use GgmlType::*;
    let bs = ty.block().1;
    match (ty, acts) {
        (Q4_0, Acts::Q8(a, p)) => row_dot32::<false>(w, bs, a, p, out, dec_q4_0),
        (Q4_1, Acts::Q8(a, p)) => row_dot32::<false>(w, bs, a, p, out, dec_q4_1),
        (Q5_0, Acts::Q8(a, p)) => row_dot32::<false>(w, bs, a, p, out, dec_q5_0),
        (Q5_1, Acts::Q8(a, p)) => row_dot32::<false>(w, bs, a, p, out, dec_q5_1),
        (Q8_0, Acts::Q8(a, p)) => row_dot32::<true>(w, bs, a, p, out, dec_q8_0),
        (Q2_K, Acts::Q8K(a, p)) => row_dot_k(w, bs, a, p, out, dec_q2_k),
        (Q3_K, Acts::Q8K(a, p)) => row_dot_k(w, bs, a, p, out, dec_q3_k),
        (Q4_K, Acts::Q8K(a, p)) => row_dot_k(w, bs, a, p, out, dec_q4_k),
        (Q5_K, Acts::Q8K(a, p)) => row_dot_k(w, bs, a, p, out, dec_q5_k),
        (Q6_K, Acts::Q8K(a, p)) => row_dot_k(w, bs, a, p, out, dec_q6_k),
        (F32 | F16 | BF16, Acts::F32(x, cols)) => {
            scratch.resize(cols, 0.0);
            dequantize(ty, w, scratch);
            for (t, o) in out.iter_mut().enumerate() {
                *o = dot_f32(scratch, &x[t * cols..(t + 1) * cols]);
            }
        }
        _ => unreachable!("activation format does not match weight type {ty:?}"),
    }
}

/// Dequantize a contiguous run of blocks to f32.
pub fn dequantize(ty: GgmlType, w: &[u8], out: &mut [f32]) {
    use GgmlType::*;
    let (n, bs) = ty.block();
    let blocks32 = |dec: fn(&[u8]) -> Dec32, out: &mut [f32]| {
        for (wb, ob) in w.chunks_exact(bs).zip(out.chunks_exact_mut(n)) {
            let b = dec(wb);
            for (o, &q) in ob.iter_mut().zip(&q32_array(b.q)) {
                *o = b.d * q as f32 + b.m;
            }
        }
    };
    let blocks_k = |dec: fn(&[u8]) -> DecK, out: &mut [f32]| {
        for (wb, ob) in w.chunks_exact(bs).zip(out.chunks_exact_mut(n)) {
            let b = dec(wb);
            for (i, o) in ob.iter_mut().enumerate() {
                let g = i / 16;
                *o = b.d * (b.sc[g] as i32 * b.q[i] as i32) as f32 - b.dmin * b.mn[g] as f32;
            }
        }
    };
    match ty {
        F32 => {
            for (o, b) in out.iter_mut().zip(w.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        F16 => {
            for (o, b) in out.iter_mut().zip(w.chunks_exact(2)) {
                *o = f16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        BF16 => {
            for (o, b) in out.iter_mut().zip(w.chunks_exact(2)) {
                *o = bf16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        Q4_0 => blocks32(dec_q4_0, out),
        Q4_1 => blocks32(dec_q4_1, out),
        Q5_0 => blocks32(dec_q5_0, out),
        Q5_1 => blocks32(dec_q5_1, out),
        Q8_0 => blocks32(dec_q8_0, out),
        Q2_K => blocks_k(dec_q2_k, out),
        Q3_K => blocks_k(dec_q3_k, out),
        Q4_K => blocks_k(dec_q4_k, out),
        Q5_K => blocks_k(dec_q5_k, out),
        Q6_K => blocks_k(dec_q6_k, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect()
    }

    /// Random weight bytes with sane (small, finite) f16 scale fields.
    fn rand_weights(ty: GgmlType, n_blocks: usize) -> Vec<u8> {
        let (_, bs) = ty.block();
        match ty {
            GgmlType::F32 => {
                return rand_vec(n_blocks, 3).iter().flat_map(|v| v.to_le_bytes()).collect();
            }
            GgmlType::F16 => {
                return rand_vec(n_blocks, 3).iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect();
            }
            _ => {}
        }
        let mut w = rand_bytes(n_blocks * bs, 42);
        let scale = f16::from_f32(0.01).to_le_bytes();
        let scale_offsets: &[usize] = match ty {
            GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => &[0],
            GgmlType::Q4_1 | GgmlType::Q5_1 => &[0, 2],
            GgmlType::Q2_K => &[80, 82],
            GgmlType::Q3_K => &[108],
            GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
            GgmlType::Q6_K => &[208],
            _ => &[],
        };
        for b in w.chunks_exact_mut(bs) {
            for &o in scale_offsets {
                b[o..o + 2].copy_from_slice(&scale);
            }
        }
        w
    }

    /// The integer dot path must agree with dequantize-then-f32-dot up to
    /// the activation quantization error, for every tile size (1..4 rows
    /// plus a remainder).
    #[test]
    fn int_dot_matches_dequant() {
        use GgmlType::*;
        let n = 1024;
        let rows = 7;
        let x: Vec<f32> = (0..rows).flat_map(|r| rand_vec(n, 7 + r as u64)).collect();
        for ty in [Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, F16, F32] {
            let (blk, _) = ty.block();
            let w = rand_weights(ty, n / blk);
            let mut deq = vec![0f32; n];
            dequantize(ty, &w, &mut deq);
            let mut q8 = vec![BlockQ8 { d: 0.0, s: 0.0, qs: [0; QK] }; rows * n / QK];
            let mut q8k = vec![BlockQ8K { d: 0.0, qs: [0; QK_K], bsums: [0; 16] }; rows * n / QK_K];
            let acts = match ty.act_kind() {
                ActKind::Q8 => {
                    for (xr, o) in x.chunks(n).zip(q8.chunks_mut(n / QK)) {
                        quantize_q8(xr, o);
                    }
                    Acts::Q8(&q8, n / QK)
                }
                ActKind::Q8K => {
                    for (xr, o) in x.chunks(n).zip(q8k.chunks_mut(n / QK_K)) {
                        quantize_q8k(xr, o);
                    }
                    Acts::Q8K(&q8k, n / QK_K)
                }
                ActKind::F32 => Acts::F32(&x, n),
            };
            let mut got = vec![0f32; rows];
            row_dot(ty, &w, acts, &mut got, &mut Vec::new());
            let nw = deq.iter().map(|v| v * v).sum::<f32>().sqrt();
            for (r, xr) in x.chunks(n).enumerate() {
                let reference = dot_f32(&deq, xr);
                let nx = xr.iter().map(|v| v * v).sum::<f32>().sqrt();
                let err = (got[r] - reference).abs() / (nw * nx).max(1e-6);
                assert!(err < 2e-3, "{ty:?} row {r}: got {}, want {reference}, rel err {err}", got[r]);
            }
        }
    }
}

/// Single-thread kernel throughput. Run with:
/// `cargo test --release kernel_throughput -- --ignored --nocapture`
#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn kernel_throughput() {
        use GgmlType::*;
        let cols = 4096;
        let rows = 2048;
        let x: Vec<f32> = (0..cols).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
        for ty in [Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, F16] {
            let (blk, bs) = ty.block();
            let w = vec![0x11u8; rows * cols / blk * bs];
            let mut q8 = vec![BlockQ8 { d: 0.0, s: 0.0, qs: [0; QK] }; cols / QK];
            let mut q8k = vec![BlockQ8K { d: 0.0, qs: [0; QK_K], bsums: [0; 16] }; cols / QK_K];
            quantize_q8(&x, &mut q8);
            quantize_q8k(&x, &mut q8k);
            let acts = match ty.act_kind() {
                ActKind::Q8 => Acts::Q8(&q8, cols / QK),
                ActKind::Q8K => Acts::Q8K(&q8k, cols / QK_K),
                ActKind::F32 => Acts::F32(&x, cols),
            };
            let rb = cols / blk * bs;
            let mut o = [0f32];
            let mut scratch = Vec::new();
            // Best of several passes: the machine may be busy.
            let mut secs = f64::MAX;
            for _ in 0..10 {
                let t = Instant::now();
                for r in 0..rows {
                    row_dot(ty, &w[r * rb..(r + 1) * rb], acts, &mut o, &mut scratch);
                }
                secs = secs.min(t.elapsed().as_secs_f64());
            }
            let gb = (rows * rb) as f64 / 1e9;
            println!("{ty:?}: {:.2} GB/s of weights, {:.2} Gweights/s", gb / secs, (rows * cols) as f64 / 1e9 / secs);
        }
    }
}
