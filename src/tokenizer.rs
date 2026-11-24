//! Tokenizers built from GGUF metadata.
//!
//! Two families cover nearly every Llama-style model:
//! * `llama` (SentencePiece): greedy highest-score bigram merging over UTF-8
//!   characters, with `<0xXX>` byte fallback. Used by Llama 2, TinyLlama, Mistral.
//! * `gpt2` (byte-level BPE): regex pre-tokenization, bytes mapped to printable
//!   unicode, then rank-ordered merges. Used by Llama 3, Qwen, SmolLM.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use anyhow::{Context, Result, bail};
use fancy_regex::Regex;

use crate::gguf::GgufFile;

const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_BYTE: i32 = 6;

enum Kind {
    Spm {
        scores: Vec<f32>,
        byte_ids: Box<[Option<u32>; 256]>,
    },
    Bpe {
        /// (left id, right id) -> (merge rank, merged id)
        ranks: HashMap<(u32, u32), (usize, u32)>,
        /// Llama 3: a pre-token that is already a vocab entry is used as-is
        /// instead of being rebuilt from merges.
        ignore_merges: bool,
        pre: Vec<Regex>,
        byte_enc: Box<[char; 256]>,
        byte_dec: HashMap<char, u8>,
    },
}

pub struct Tokenizer {
    kind: Kind,
    tokens: Vec<String>,
    types: Vec<i32>,
    lookup: HashMap<String, u32>,
    /// Control / user-defined tokens matched literally in input text,
    /// longest first.
    specials: Vec<(String, u32)>,
    special_first_bytes: [bool; 256],
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    add_space_prefix: bool,
    stop: Vec<u32>,
}

const GPT2_RE: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LLAMA3_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_RE: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// GPT-2's reversible byte -> printable-unicode map.
fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable =
            (b'!' as u32..=b'~' as u32).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

fn str_array(f: &GgufFile, key: &str) -> Result<Vec<String>> {
    f.get(key)
        .and_then(|v| v.as_array())
        .with_context(|| format!("missing {key}"))?
        .iter()
        .map(|v| v.as_str().map(str::to_owned).context("non-string entry"))
        .collect()
}

impl Tokenizer {
    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        let model = f.get_str("tokenizer.ggml.model").context("missing tokenizer.ggml.model")?;
        let tokens = str_array(f, "tokenizer.ggml.tokens")?;
        let types: Vec<i32> = match f.get("tokenizer.ggml.token_type").and_then(|v| v.as_array()) {
            Some(a) => a.iter().map(|v| v.as_u64().unwrap_or(1) as i32).collect(),
            None => vec![1; tokens.len()],
        };
        let lookup: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();

        let pre_name = f.get_str("tokenizer.ggml.pre").unwrap_or("default");
        let llama3 = matches!(pre_name, "llama3" | "llama-v3" | "llama-bpe" | "smaug-bpe" | "falcon3");
        let kind = match model {
            "llama" => {
                let scores = match f.get("tokenizer.ggml.scores").and_then(|v| v.as_array()) {
                    Some(a) => a.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect(),
                    None => vec![0.0; tokens.len()],
                };
                let mut byte_ids = Box::new([None; 256]);
                for (b, slot) in byte_ids.iter_mut().enumerate() {
                    *slot = lookup.get(&format!("<0x{b:02X}>")).copied();
                }
                Kind::Spm { scores, byte_ids }
            }
            "gpt2" => {
                let merges = str_array(f, "tokenizer.ggml.merges")?;
                let ranks = merges
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, m)| {
                        let (a, b) = m.split_once(' ')?;
                        let merged = lookup.get(&format!("{a}{b}"))?;
                        Some(((*lookup.get(a)?, *lookup.get(b)?), (rank, *merged)))
                    })
                    .collect();
                let re: &[&str] = match pre_name {
                    _ if llama3 => &[LLAMA3_RE],
                    "qwen2" | "deepseek-r1-qwen" | "qwen35" | "megrez" => &[QWEN2_RE],
                    // SmolLM splits digits apart first, then applies GPT-2's rules.
                    "smollm" => &[r"\p{N}", GPT2_RE],
                    "gpt-2" | "gpt2" | "default" | "starcoder" | "refact" | "phi-2" => &[GPT2_RE],
                    other => bail!("unsupported BPE pre-tokenizer {other:?}"),
                };
                let byte_enc = Box::new(bytes_to_unicode());
                let byte_dec = byte_enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
                Kind::Bpe {
                    ranks,
                    ignore_merges: llama3,
                    pre: re.iter().map(|r| Regex::new(r)).collect::<Result<_, _>>()?,
                    byte_enc,
                    byte_dec,
                }
            }
            other => bail!("unsupported tokenizer model {other:?}"),
        };

        let mut specials: Vec<(String, u32)> = types
            .iter()
            .enumerate()
            .filter(|&(i, &t)| matches!(t, TYPE_UNKNOWN | TYPE_CONTROL | TYPE_USER_DEFINED) && !tokens[i].is_empty())
            .map(|(i, _)| (tokens[i].clone(), i as u32))
            .collect();
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let mut special_first_bytes = [false; 256];
        for (s, _) in &specials {
            special_first_bytes[s.as_bytes()[0] as usize] = true;
        }

        let id = |k: &str| f.get_u64(k).map(|v| v as u32);
        let bos = id("tokenizer.ggml.bos_token_id");
        let eos = id("tokenizer.ggml.eos_token_id");
        let mut stop: Vec<u32> =
            [eos, id("tokenizer.ggml.eot_token_id"), id("tokenizer.ggml.eom_token_id")].into_iter().flatten().collect();
        for s in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end|>", "<|eom_id|>", "<end_of_turn>", "</s>"] {
            if let Some(&i) = lookup.get(s) {
                stop.push(i);
            }
        }
        stop.sort_unstable();
        stop.dedup();

        let is_spm = matches!(kind, Kind::Spm { .. });
        Ok(Tokenizer {
            // Same defaults as llama.cpp when the file doesn't say: SPM and
            // Llama 3 models expect BOS, other BPE models don't.
            add_bos: f
                .get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.as_u64())
                .map(|v| v != 0)
                .unwrap_or(is_spm || llama3),
            add_space_prefix: f
                .get("tokenizer.ggml.add_space_prefix")
                .and_then(|v| v.as_u64())
                .map(|v| v != 0)
                .unwrap_or(is_spm),
            kind,
            tokens,
            types,
            lookup,
            specials,
            special_first_bytes,
            bos,
            eos,
            stop,
        })
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop.binary_search(&id).is_ok()
    }

    pub fn token_str(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(String::as_str).unwrap_or("")
    }

    /// Tokenize `text`. With `parse_special`, literal control-token strings
    /// such as `<|im_start|>` become their single token ids.
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos && let Some(b) = self.bos {
            out.push(b);
        }
        let mut prev_special = true;
        let mut rest = text;
        while !rest.is_empty() {
            let (plain, special) = if parse_special { self.next_special(rest) } else { (rest.len(), None) };
            if plain > 0 {
                self.encode_plain(&rest[..plain], prev_special, &mut out);
                prev_special = false;
            }
            rest = &rest[plain..];
            if let Some((len, id)) = special {
                out.push(id);
                prev_special = true;
                rest = &rest[len..];
            }
        }
        out
    }

    /// Byte offset of the next special token in `s` and its (len, id).
    fn next_special(&self, s: &str) -> (usize, Option<(usize, u32)>) {
        let bytes = s.as_bytes();
        for i in 0..bytes.len() {
            if !self.special_first_bytes[bytes[i] as usize] || !s.is_char_boundary(i) {
                continue;
            }
            for (sp, id) in &self.specials {
                if s[i..].starts_with(sp.as_str()) {
                    return (i, Some((sp.len(), *id)));
                }
            }
        }
        (s.len(), None)
    }

    fn encode_plain(&self, text: &str, at_start: bool, out: &mut Vec<u32>) {
        match &self.kind {
            Kind::Spm { scores, byte_ids } => {
                let mut s = String::with_capacity(text.len() + 3);
                if self.add_space_prefix && at_start {
                    s.push('▁');
                }
                s.push_str(&text.replace(' ', "▁"));
                self.spm_encode(&s, scores, byte_ids, out);
            }
            Kind::Bpe { ranks, ignore_merges, pre, byte_enc, .. } => {
                let mut pieces = vec![text.to_owned()];
                for re in pre {
                    pieces = pieces.iter().flat_map(|p| split_regex(re, p)).collect();
                }
                for p in pieces {
                    let mapped: String = p.bytes().map(|b| byte_enc[b as usize]).collect();
                    self.bpe_word(&mapped, ranks, *ignore_merges, out);
                }
            }
        }
    }

    fn spm_encode(&self, text: &str, scores: &[f32], byte_ids: &[Option<u32>; 256], out: &mut Vec<u32>) {
        #[derive(Clone, Copy)]
        struct Sym {
            start: usize,
            len: usize,
            prev: Option<usize>,
            next: Option<usize>,
        }
        struct Cand {
            score: f32,
            left: usize,
            right: usize,
            len: usize,
        }
        impl PartialEq for Cand {
            fn eq(&self, o: &Self) -> bool {
                self.cmp(o) == Ordering::Equal
            }
        }
        impl Eq for Cand {}
        impl PartialOrd for Cand {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Cand {
            // Highest score first; ties go to the leftmost pair.
            fn cmp(&self, o: &Self) -> Ordering {
                self.score.total_cmp(&o.score).then(o.left.cmp(&self.left))
            }
        }

        let mut syms: Vec<Sym> = Vec::new();
        for (i, (start, ch)) in text.char_indices().enumerate() {
            syms.push(Sym { start, len: ch.len_utf8(), prev: i.checked_sub(1), next: Some(i + 1) });
        }
        if let Some(last) = syms.last_mut() {
            last.next = None;
        }
        let mut heap = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<Cand>, syms: &[Sym], l: usize, r: usize| {
            let s = &text[syms[l].start..syms[r].start + syms[r].len];
            if let Some(&id) = self.lookup.get(s) {
                heap.push(Cand { score: scores[id as usize], left: l, right: r, len: s.len() });
            }
        };
        for i in 1..syms.len() {
            try_add(&mut heap, &syms, i - 1, i);
        }
        while let Some(c) = heap.pop() {
            let (l, r) = (syms[c.left], syms[c.right]);
            // Stale candidate: one side was already merged into something else.
            if l.len == 0 || r.len == 0 || l.len + r.len != c.len || l.next != Some(c.right) {
                continue;
            }
            syms[c.left].len += r.len;
            syms[c.left].next = r.next;
            syms[c.right].len = 0;
            if let Some(n) = r.next {
                syms[n].prev = Some(c.left);
            }
            if let Some(p) = l.prev {
                try_add(&mut heap, &syms, p, c.left);
            }
            if let Some(n) = r.next {
                try_add(&mut heap, &syms, c.left, n);
            }
        }
        let mut i = if syms.is_empty() { None } else { Some(0) };
        while let Some(idx) = i {
            let s = syms[idx];
            let piece = &text[s.start..s.start + s.len];
            match self.lookup.get(piece) {
                Some(&id) => out.push(id),
                None => {
                    for b in piece.bytes() {
                        if let Some(id) = byte_ids[b as usize] {
                            out.push(id);
                        }
                    }
                }
            }
            i = s.next;
        }
    }

    /// Byte-level BPE on one pre-token: repeatedly apply the lowest-ranked
    /// merge among adjacent symbols. A heap of candidate pairs over a linked
    /// list keeps this O(n log n), so a huge pasted word doesn't hang.
    fn bpe_word(&self, word: &str, ranks: &HashMap<(u32, u32), (usize, u32)>, ignore_merges: bool, out: &mut Vec<u32>) {
        use std::cmp::Reverse;
        if ignore_merges && let Some(&id) = self.lookup.get(word) {
            out.push(id);
            return;
        }
        // Every byte-mapped char is a vocab entry in GPT-2-style vocabularies.
        let mut ids: Vec<u32> =
            word.chars().filter_map(|c| self.lookup.get(c.encode_utf8(&mut [0; 4])).copied()).collect();
        let n = ids.len();
        let mut next: Vec<usize> = (1..=n).collect(); // n = end
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect(); // usize::MAX = start
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<_>, ids: &[u32], l: usize, r: usize| {
            if let Some(&(rank, _)) = ranks.get(&(ids[l], ids[r])) {
                heap.push(Reverse((rank, l, ids[l], ids[r])));
            }
        };
        for i in 1..n {
            push(&mut heap, &ids, i - 1, i);
        }
        while let Some(Reverse((_, l, lid, rid))) = heap.pop() {
            let r = next[l];
            // Stale entry: either side has changed since it was queued.
            if !alive[l] || r >= n || ids[l] != lid || ids[r] != rid {
                continue;
            }
            ids[l] = ranks[&(lid, rid)].1;
            alive[r] = false;
            next[l] = next[r];
            if next[l] < n {
                prev[next[l]] = l;
            }
            if prev[l] != usize::MAX {
                push(&mut heap, &ids, prev[l], l);
            }
            if next[l] < n {
                push(&mut heap, &ids, l, next[l]);
            }
        }
        out.extend((0..n).filter(|&i| alive[i]).map(|i| ids[i]));
    }

    /// Raw bytes a token contributes to output text. Control tokens are
    /// rendered as nothing.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let i = id as usize;
        let Some(tok) = self.tokens.get(i) else {
            return Vec::new();
        };
        let ty = self.types.get(i).copied().unwrap_or(1);
        if ty == TYPE_CONTROL {
            return Vec::new();
        }
        match &self.kind {
            Kind::Spm { .. } => {
                if ty == TYPE_BYTE
                    && let Some(hex) = tok.strip_prefix("<0x").and_then(|s| s.strip_suffix('>'))
                    && let Ok(b) = u8::from_str_radix(hex, 16)
                {
                    return vec![b];
                }
                tok.replace('▁', " ").into_bytes()
            }
            Kind::Bpe { byte_dec, .. } => {
                if ty == TYPE_USER_DEFINED {
                    return tok.as_bytes().to_vec();
                }
                tok.chars().filter_map(|c| byte_dec.get(&c).copied()).collect()
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn split_regex(re: &Regex, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text).flatten() {
        if m.start() > last {
            out.push(text[last..m.start()].to_owned());
        }
        out.push(m.as_str().to_owned());
        last = m.end();
    }
    if last < text.len() {
        out.push(text[last..].to_owned());
    }
    out
}

/// Turns a stream of token bytes into valid UTF-8 text, holding back
/// incomplete multi-byte sequences until the rest arrives.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        // Invalid bytes: replace just those and keep going.
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                        // Incomplete sequence at the end: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StreamDecoder;

    #[test]
    fn stream_decoder_holds_partial_utf8() {
        let mut d = StreamDecoder::default();
        assert_eq!(d.push(&[0xFF, 0xE2, 0x82]), "\u{FFFD}");
        assert_eq!(d.push(&[0xAC]), "€");
        assert_eq!(d.push(b"ok"), "ok");
    }
}
