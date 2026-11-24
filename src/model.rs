//! Llama-family transformer (Llama 1/2/3, Mistral, TinyLlama, SmolLM, Qwen2,
//! Qwen2.5, Qwen3) running directly on mmapped GGUF weights.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::gguf::GgufFile;
use crate::ops::{self, ActBuf, Matrix, Rope, prof};
use crate::quant::{GgmlType, dot_f32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Llama,
    Qwen2,
    Qwen3,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Arch,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub ctx_train: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    /// Linear RoPE scaling: positions are divided by this factor.
    pub rope_linear_scale: f32,
}

impl Config {
    fn from_gguf(f: &GgufFile) -> Result<Self> {
        let arch_name = f.get_str("general.architecture").context("missing general.architecture")?;
        let arch = match arch_name {
            "llama" | "mistral" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            other => bail!("unsupported architecture {other:?} (supported: llama, mistral, qwen2, qwen3)"),
        };
        let u = |k: &str| {
            f.get_u64(&format!("{arch_name}.{k}"))
                .map(|v| v as usize)
                .with_context(|| format!("missing {arch_name}.{k}"))
        };
        let dim = u("embedding_length")?;
        let n_heads = u("attention.head_count")?;
        let n_kv_heads = u("attention.head_count_kv").unwrap_or(n_heads);
        let head_dim = u("attention.key_length").unwrap_or(dim / n_heads);
        let embd = f.tensor("token_embd.weight").context("missing token_embd.weight")?;
        Ok(Config {
            arch,
            dim,
            n_layers: u("block_count")?,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim: u("feed_forward_length")?,
            vocab: embd.dims[1] as usize,
            ctx_train: u("context_length").unwrap_or(2048),
            eps: f.get_f32(&format!("{arch_name}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5),
            rope_base: f.get_f32(&format!("{arch_name}.rope.freq_base")).unwrap_or(10000.0),
            rope_dim: u("rope.dimension_count").unwrap_or(head_dim),
            rope_linear_scale: {
                let factor = f.get_f32(&format!("{arch_name}.rope.scaling.factor")).unwrap_or(1.0);
                match f.get_str(&format!("{arch_name}.rope.scaling.type")) {
                    Some("linear") if factor > 0.0 => factor,
                    Some("none") | None => 1.0,
                    Some(other) => {
                        if factor != 1.0 {
                            eprintln!("warning: {other} RoPE scaling is not implemented; long contexts may degrade");
                        }
                        1.0
                    }
                }
            },
        })
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: Matrix<'a>,
    wk: Matrix<'a>,
    wv: Matrix<'a>,
    wo: Matrix<'a>,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    w_gate: Matrix<'a>,
    w_up: Matrix<'a>,
    w_down: Matrix<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    tok_embd: Matrix<'a>,
    layers: Vec<Layer<'a>>,
    out_norm: Vec<f32>,
    output: Matrix<'a>,
    rope_freqs: Option<Vec<f32>>,
}

fn matrix<'a>(f: &'a GgufFile, name: &str, rows: usize, cols: usize) -> Result<Matrix<'a>> {
    let t = f.tensor(name).with_context(|| format!("missing tensor {name}"))?;
    if t.dims.len() != 2 || t.dims[0] as usize != cols || t.dims[1] as usize != rows {
        bail!("tensor {name}: expected [{cols}, {rows}], got {:?}", t.dims);
    }
    Ok(Matrix::new(t.ty, rows, cols, f.tensor_data(t)))
}

fn vector(f: &GgufFile, name: &str) -> Option<Vec<f32>> {
    let t = f.tensor(name)?;
    let mut v = vec![0.0; t.n_elements() as usize];
    crate::quant::dequantize(t.ty, f.tensor_data(t), &mut v);
    Some(v)
}

fn req_vector(f: &GgufFile, name: &str) -> Result<Vec<f32>> {
    vector(f, name).with_context(|| format!("missing tensor {name}"))
}

impl<'a> Model<'a> {
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        let cfg = Config::from_gguf(f)?;
        let (d, qd, kvd, ff) = (cfg.dim, cfg.q_dim(), cfg.kv_dim(), cfg.ffn_dim);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let n = |s: &str| format!("blk.{i}.{s}");
            layers.push(Layer {
                attn_norm: req_vector(f, &n("attn_norm.weight"))?,
                wq: matrix(f, &n("attn_q.weight"), qd, d)?,
                wk: matrix(f, &n("attn_k.weight"), kvd, d)?,
                wv: matrix(f, &n("attn_v.weight"), kvd, d)?,
                wo: matrix(f, &n("attn_output.weight"), d, qd)?,
                bq: vector(f, &n("attn_q.bias")),
                bk: vector(f, &n("attn_k.bias")),
                bv: vector(f, &n("attn_v.bias")),
                q_norm: vector(f, &n("attn_q_norm.weight")),
                k_norm: vector(f, &n("attn_k_norm.weight")),
                ffn_norm: req_vector(f, &n("ffn_norm.weight"))?,
                w_gate: matrix(f, &n("ffn_gate.weight"), ff, d)?,
                w_up: matrix(f, &n("ffn_up.weight"), ff, d)?,
                w_down: matrix(f, &n("ffn_down.weight"), d, ff)?,
            });
        }
        let tok_embd = matrix(f, "token_embd.weight", cfg.vocab, d)?;
        // Many small models tie the output projection to the embeddings.
        let output = match f.tensor("output.weight") {
            Some(_) => matrix(f, "output.weight", cfg.vocab, d)?,
            None => tok_embd,
        };
        Ok(Model {
            out_norm: req_vector(f, "output_norm.weight")?,
            rope_freqs: vector(f, "rope_freqs.weight"),
            cfg,
            tok_embd,
            layers,
            output,
        })
    }

    /// Count of weight tensors by quantization type, for display.
    pub fn weight_types(&self) -> Vec<(GgmlType, usize)> {
        let mut counts: Vec<(GgmlType, usize)> = Vec::new();
        let mut add = |t: GgmlType| match counts.iter_mut().find(|(k, _)| *k == t) {
            Some((_, c)) => *c += 1,
            None => counts.push((t, 1)),
        };
        for l in &self.layers {
            for m in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
                add(m.ty);
            }
        }
        add(self.tok_embd.ty);
        add(self.output.ty);
        counts
    }
}

/// KV cache plus scratch buffers for one generation stream.
pub struct Session<'m, 'a> {
    pub model: &'m Model<'a>,
    pub n_ctx: usize,
    /// Tokens currently held in the KV cache, in order.
    pub tokens: Vec<u32>,
    rope: Rope,
    k_cache: Vec<f32>, // [layer][pos][kv_dim]
    v_cache: Vec<f32>,
    max_batch: usize,
    act: ActBuf,
}

pub enum Logits {
    /// Logits for the last token only.
    Last,
    /// Logits for every token in the batch (perplexity).
    All,
}

impl<'m, 'a> Session<'m, 'a> {
    pub fn new(model: &'m Model<'a>, n_ctx: usize, max_batch: usize) -> Self {
        let c = &model.cfg;
        let kv = c.n_layers * n_ctx * c.kv_dim();
        let neox = matches!(c.arch, Arch::Qwen2 | Arch::Qwen3);
        Session {
            model,
            n_ctx,
            tokens: Vec::new(),
            rope: Rope::new(c.rope_dim, n_ctx, c.rope_base, c.rope_linear_scale, neox, model.rope_freqs.as_deref()),
            k_cache: vec![0.0; kv],
            v_cache: vec![0.0; kv],
            max_batch: max_batch.max(1),
            act: ActBuf::default(),
        }
    }

    pub fn pos(&self) -> usize {
        self.tokens.len()
    }

    /// Drop cached tokens after `n` (keeps the first `n`).
    pub fn truncate(&mut self, n: usize) {
        self.tokens.truncate(n);
    }

    /// Feed tokens through the model, appending them to the KV cache.
    /// Returns logits (`vocab` per row) for the last token or for all of them.
    pub fn forward(&mut self, tokens: &[u32], mode: Logits) -> Result<Vec<f32>> {
        if self.pos() + tokens.len() > self.n_ctx {
            bail!("context full: {} + {} tokens > n_ctx {}", self.pos(), tokens.len(), self.n_ctx);
        }
        let vocab = self.model.cfg.vocab;
        let mut out = Vec::new();
        let chunks: Vec<&[u32]> = tokens.chunks(self.max_batch).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            match mode {
                Logits::All => out.extend(self.forward_batch(chunk, true)),
                Logits::Last if last => out = self.forward_batch(chunk, false),
                Logits::Last => {
                    self.forward_batch(chunk, false);
                }
            }
        }
        debug_assert!(out.len() % vocab == 0);
        Ok(out)
    }

    fn forward_batch(&mut self, tokens: &[u32], all_logits: bool) -> Vec<f32> {
        let m = self.model;
        let c = &m.cfg;
        let n = tokens.len();
        let p0 = self.pos();
        let (d, qd, kvd, ff, hd) = (c.dim, c.q_dim(), c.kv_dim(), c.ffn_dim, c.head_dim);
        let layer_stride = self.n_ctx * kvd;

        let mut x = vec![0.0f32; n * d];
        for (t, &tok) in tokens.iter().enumerate() {
            m.tok_embd.dequant_row(tok as usize, &mut x[t * d..(t + 1) * d]);
        }
        let mut xb = vec![0.0f32; n * d];
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        let mut att = vec![0.0f32; n * qd];
        let mut hb = vec![0.0f32; n * ff];
        let mut hb2 = vec![0.0f32; n * ff];

        for (li, l) in m.layers.iter().enumerate() {
            // --- attention ---
            for (o, xr) in xb.chunks_exact_mut(d).zip(x.chunks_exact(d)) {
                ops::rmsnorm(o, xr, &l.attn_norm, c.eps);
            }
            prof::time(prof::MATMUL, || ops::matmul(&l.wq, &xb, n, &mut q, &mut self.act));
            prof::time(prof::MATMUL, || ops::matmul(&l.wk, &xb, n, &mut k, &mut self.act));
            prof::time(prof::MATMUL, || ops::matmul(&l.wv, &xb, n, &mut v, &mut self.act));
            for (buf, bias, w) in [(&mut q, &l.bq, qd), (&mut k, &l.bk, kvd), (&mut v, &l.bv, kvd)] {
                if let Some(b) = bias {
                    for row in buf.chunks_exact_mut(w) {
                        row.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                    }
                }
            }
            for t in 0..n {
                let pos = p0 + t;
                for h in q[t * qd..(t + 1) * qd].chunks_exact_mut(hd) {
                    if let Some(w) = &l.q_norm {
                        ops::rmsnorm_inplace(h, w, c.eps);
                    }
                    self.rope.apply(h, pos);
                }
                for h in k[t * kvd..(t + 1) * kvd].chunks_exact_mut(hd) {
                    if let Some(w) = &l.k_norm {
                        ops::rmsnorm_inplace(h, w, c.eps);
                    }
                    self.rope.apply(h, pos);
                }
            }
            let base = li * layer_stride + p0 * kvd;
            self.k_cache[base..base + n * kvd].copy_from_slice(&k);
            self.v_cache[base..base + n * kvd].copy_from_slice(&v);

            let kc = &self.k_cache[li * layer_stride..(li + 1) * layer_stride];
            let vc = &self.v_cache[li * layer_stride..(li + 1) * layer_stride];
            let group = c.n_heads / c.n_kv_heads;
            let scale = 1.0 / (hd as f32).sqrt();
            let q = &q;
            prof::time(prof::ATTN, || {
                att.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
                    let (t, h) = (i / c.n_heads, i % c.n_heads);
                    let len = p0 + t + 1; // causal: attend to positions 0..=pos
                    let off = (h / group) * hd;
                    let qh = &q[t * qd + h * hd..t * qd + (h + 1) * hd];
                    let mut s: Vec<f32> =
                        (0..len).map(|p| dot_f32(qh, &kc[p * kvd + off..p * kvd + off + hd]) * scale).collect();
                    ops::softmax(&mut s);
                    out.fill(0.0);
                    for (p, &w) in s.iter().enumerate() {
                        let vr = &vc[p * kvd + off..p * kvd + off + hd];
                        for (o, &vv) in out.iter_mut().zip(vr) {
                            *o = w.mul_add(vv, *o);
                        }
                    }
                })
            });
            prof::time(prof::MATMUL, || ops::matmul(&l.wo, &att, n, &mut xb, &mut self.act));
            x.iter_mut().zip(&xb).for_each(|(a, b)| *a += b);

            // --- feed-forward (SwiGLU) ---
            for (o, xr) in xb.chunks_exact_mut(d).zip(x.chunks_exact(d)) {
                ops::rmsnorm(o, xr, &l.ffn_norm, c.eps);
            }
            prof::time(prof::MATMUL, || ops::matmul(&l.w_gate, &xb, n, &mut hb, &mut self.act));
            prof::time(prof::MATMUL, || ops::matmul(&l.w_up, &xb, n, &mut hb2, &mut self.act));
            prof::time(prof::MISC, || hb.iter_mut().zip(&hb2).for_each(|(g, u)| *g = ops::silu(*g) * u));
            prof::time(prof::MATMUL, || ops::matmul(&l.w_down, &hb, n, &mut xb, &mut self.act));
            x.iter_mut().zip(&xb).for_each(|(a, b)| *a += b);
        }
        self.tokens.extend_from_slice(tokens);

        // Only project the rows we need: the vocab matmul is the largest one.
        let rows: &[f32] = if all_logits { &x } else { &x[(n - 1) * d..] };
        let nr = rows.len() / d;
        let mut normed = vec![0.0f32; nr * d];
        for (o, xr) in normed.chunks_exact_mut(d).zip(rows.chunks_exact(d)) {
            ops::rmsnorm(o, xr, &m.out_norm, c.eps);
        }
        let mut logits = vec![0.0f32; nr * c.vocab];
        prof::time(prof::HEAD, || ops::matmul(&m.output, &normed, nr, &mut logits, &mut self.act));
        logits
    }
}
