//! Token sampling: repetition penalty, temperature, top-k, top-p, min-p.

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng: u64,
    cand: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = cfg.seed.max(1);
        Self { cfg, rng, cand: Vec::new() }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        let c = &self.cfg;
        if c.repeat_penalty != 1.0 {
            let start = history.len().saturating_sub(c.repeat_last_n);
            let mut seen = history[start..].to_vec();
            seen.sort_unstable();
            seen.dedup();
            for &t in &seen {
                let l = &mut logits[t as usize];
                *l = if *l > 0.0 { *l / c.repeat_penalty } else { *l * c.repeat_penalty };
            }
        }
        if c.temperature <= 0.0 {
            return argmax(logits);
        }

        let k = if c.top_k == 0 { logits.len() } else { c.top_k.min(logits.len()) };
        self.cand.clear();
        self.cand.extend(logits.iter().enumerate().map(|(i, &l)| (i as u32, l / c.temperature)));
        if k < self.cand.len() {
            self.cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            self.cand.truncate(k);
        }
        self.cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // softmax over the survivors
        let max = self.cand[0].1;
        let mut sum = 0.0;
        for c in self.cand.iter_mut() {
            c.1 = (c.1 - max).exp();
            sum += c.1;
        }
        for c in self.cand.iter_mut() {
            c.1 /= sum;
        }

        // min-p: drop tokens below min_p * p(best)
        if c.min_p > 0.0 {
            let floor = self.cand[0].1 * c.min_p;
            let keep = self.cand.iter().take_while(|x| x.1 >= floor).count().max(1);
            self.cand.truncate(keep);
        }
        // top-p: smallest prefix with cumulative mass >= top_p
        if c.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = self.cand.len();
            for (i, x) in self.cand.iter().enumerate() {
                cum += x.1;
                if cum >= c.top_p {
                    keep = i + 1;
                    break;
                }
            }
            self.cand.truncate(keep);
        }

        let total: f32 = self.cand.iter().map(|x| x.1).sum();
        let mut r = self.next_f32() * total;
        for &(id, p) in &self.cand {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        self.cand.last().unwrap().0
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}
