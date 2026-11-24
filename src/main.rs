mod chat;
mod gguf;
mod model;
mod ops;
mod quant;
mod sampler;
mod tokenizer;

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};

use chat::{ChatTemplate, Message};
use gguf::GgufFile;
use model::{Logits, Model, Session};
use sampler::{Sampler, SamplerConfig};
use tokenizer::{StreamDecoder, Tokenizer};

#[derive(Parser)]
#[command(name = "ember", version, about = "A small, fast GGUF inference engine for CPUs")]
struct Cli {
    /// Worker threads (default: number of physical cores).
    #[arg(short, long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print GGUF metadata and a tensor summary.
    Inspect {
        model: PathBuf,
        /// List every tensor.
        #[arg(long)]
        tensors: bool,
    },
    /// Print the token ids for a piece of text.
    Tokenize {
        #[arg(short, long)]
        model: PathBuf,
        text: String,
        /// Also print each token's text.
        #[arg(short, long)]
        verbose: bool,
        /// Don't prepend BOS even if the model asks for it.
        #[arg(long)]
        no_bos: bool,
    },
    /// Generate text from a prompt.
    Run {
        #[command(flatten)]
        load: LoadArgs,
        #[arg(short, long)]
        prompt: String,
        /// Max tokens to generate.
        #[arg(short = 'n', long, default_value_t = 256)]
        max_tokens: usize,
        /// Wrap the prompt in the model's chat template.
        #[arg(long)]
        chat: bool,
        /// System prompt (with --chat).
        #[arg(long)]
        system: Option<String>,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Interactive multi-turn chat. Reuses the KV cache between turns.
    Chat {
        #[command(flatten)]
        load: LoadArgs,
        #[arg(long)]
        system: Option<String>,
        #[arg(short = 'n', long, default_value_t = 1024)]
        max_tokens: usize,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Measure prompt-processing and generation throughput.
    Bench {
        #[command(flatten)]
        load: LoadArgs,
        /// Prompt tokens for the prefill test.
        #[arg(short, long, default_value_t = 512)]
        prompt_tokens: usize,
        /// Tokens for the generation test.
        #[arg(short = 'n', long, default_value_t = 128)]
        gen_tokens: usize,
        #[arg(short, long, default_value_t = 3)]
        reps: usize,
    },
    /// Perplexity over a text file (same chunking as llama-perplexity).
    Ppl {
        #[command(flatten)]
        load: LoadArgs,
        #[arg(short, long)]
        file: PathBuf,
        /// Stop after this many chunks.
        #[arg(long)]
        chunks: Option<usize>,
    },
}

#[derive(Args)]
struct LoadArgs {
    /// Path to a .gguf model.
    #[arg(short, long)]
    model: PathBuf,
    /// Context length (default: min(training context, 4096)).
    #[arg(short, long)]
    ctx: Option<usize>,
    /// Max tokens per forward pass during prompt processing.
    #[arg(short, long, default_value_t = 256)]
    batch: usize,
}

#[derive(Args)]
struct SamplingArgs {
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.7)]
    temp: f32,
    /// 0 = disabled.
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 0.05)]
    min_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
    /// RNG seed (default: time-based).
    #[arg(long)]
    seed: Option<u64>,
}

impl SamplingArgs {
    fn config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed.unwrap_or_else(|| {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64
            }),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let threads = cli.threads.unwrap_or_else(physical_cores);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    match cli.cmd {
        Cmd::Inspect { model, tensors } => inspect(&model, tensors),
        Cmd::Tokenize { model, text, verbose, no_bos } => {
            let f = GgufFile::open(&model)?;
            let tok = Tokenizer::from_gguf(&f)?;
            let ids = tok.encode(&text, tok.add_bos && !no_bos, true);
            if verbose {
                for id in &ids {
                    println!("{id:>7} {:?}", tok.token_str(*id));
                }
            } else {
                println!("{ids:?}");
            }
            Ok(())
        }
        Cmd::Run { load, prompt, max_tokens, chat, system, sampling } => {
            run(&load, &prompt, max_tokens, chat, system, &sampling)
        }
        Cmd::Chat { load, system, max_tokens, sampling } => chat_repl(&load, system, max_tokens, &sampling),
        Cmd::Bench { load, prompt_tokens, gen_tokens, reps } => bench(&load, prompt_tokens, gen_tokens, reps),
        Cmd::Ppl { load, file, chunks } => perplexity(&load, &file, chunks),
    }
}

/// Physical cores (not SMT siblings). Decoding is memory-bound, and a second
/// thread per core mostly adds synchronization cost.
fn physical_cores() -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut cores = std::collections::HashSet::new();
    for cpu in 0..logical {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_cpus_list");
        match std::fs::read_to_string(path) {
            Ok(s) => {
                cores.insert(s.trim().to_owned());
            }
            Err(_) => return logical,
        }
    }
    cores.len().max(1)
}

fn inspect(path: &Path, list_tensors: bool) -> Result<()> {
    let f = GgufFile::open(path)?;
    println!("GGUF v{}  {:.1} MiB  {} tensors", f.version, f.file_size() as f64 / (1 << 20) as f64, f.tensors.len());
    println!("\nmetadata:");
    for (k, v) in &f.metadata {
        println!("  {k} = {v}");
    }
    let mut by_type: Vec<(quant::GgmlType, usize, u64)> = Vec::new();
    for t in &f.tensors {
        match by_type.iter_mut().find(|e| e.0 == t.ty) {
            Some(e) => {
                e.1 += 1;
                e.2 += t.n_elements();
            }
            None => by_type.push((t.ty, 1, t.n_elements())),
        }
    }
    println!("\ntensor types:");
    for (ty, n, elems) in by_type {
        println!(
            "  {:<6} {n:>4} tensors  {:>8.1}M params  {:.2} bits/weight",
            format!("{ty:?}"),
            elems as f64 / 1e6,
            ty.bits_per_weight()
        );
    }
    if list_tensors {
        println!("\ntensors:");
        for t in &f.tensors {
            println!("  {:<40} {:<6} {:?}", t.name, format!("{:?}", t.ty), t.dims);
        }
    }
    Ok(())
}

struct Loaded {
    file: GgufFile,
    tok: Tokenizer,
}

fn load(args: &LoadArgs) -> Result<Loaded> {
    let file = GgufFile::open(&args.model)?;
    let tok = Tokenizer::from_gguf(&file)?;
    Ok(Loaded { file, tok })
}

fn new_session<'m, 'a>(model: &'m Model<'a>, args: &LoadArgs) -> Session<'m, 'a> {
    let n_ctx = args.ctx.unwrap_or(model.cfg.ctx_train.min(4096));
    Session::new(model, n_ctx, args.batch)
}

fn describe(model: &Model, path: &Path) {
    let c = &model.cfg;
    let types: Vec<String> = model.weight_types().iter().map(|(t, n)| format!("{t:?}x{n}")).collect();
    eprintln!(
        "ember: {} | {:?} {} layers, dim {}, heads {}/{}, vocab {} | weights {} | {} threads",
        path.file_name().unwrap_or_default().to_string_lossy(),
        c.arch,
        c.n_layers,
        c.dim,
        c.n_heads,
        c.n_kv_heads,
        c.vocab,
        types.join(" "),
        rayon::current_num_threads(),
    );
}

struct GenStats {
    prompt_tokens: usize,
    prompt_secs: f64,
    gen_tokens: usize,
    gen_secs: f64,
}

impl GenStats {
    fn print(&self) {
        eprintln!(
            "\n[prompt: {} tok, {:.1} tok/s | generated: {} tok, {:.1} tok/s]",
            self.prompt_tokens,
            self.prompt_tokens as f64 / self.prompt_secs,
            self.gen_tokens,
            self.gen_tokens as f64 / self.gen_secs,
        );
    }
}

/// Feed `new_tokens` (appended after whatever is cached), then sample until
/// a stop token or `max_tokens`. Streams text through `emit`.
fn generate(
    sess: &mut Session,
    tok: &Tokenizer,
    sampler: &mut Sampler,
    new_tokens: &[u32],
    max_tokens: usize,
    mut emit: impl FnMut(&str),
) -> Result<(Vec<u32>, GenStats)> {
    if new_tokens.is_empty() {
        bail!("empty prompt: nothing to condition on");
    }
    let t0 = Instant::now();
    let mut logits = sess.forward(new_tokens, Logits::Last)?;
    let prompt_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut out = Vec::new();
    let mut dec = StreamDecoder::default();
    for _ in 0..max_tokens {
        let next = sampler.sample(&mut logits, &sess.tokens);
        if tok.is_stop(next) {
            break;
        }
        out.push(next);
        emit(&dec.push(&tok.token_bytes(next)));
        if sess.pos() >= sess.n_ctx {
            eprintln!("\n[context full]");
            break;
        }
        logits = sess.forward(&[next], Logits::Last)?;
    }
    emit(&dec.push(&[]));
    let stats = GenStats {
        prompt_tokens: new_tokens.len(),
        prompt_secs,
        gen_tokens: out.len(),
        gen_secs: t1.elapsed().as_secs_f64(),
    };
    Ok((out, stats))
}

fn print_flush(s: &str) {
    let mut o = std::io::stdout().lock();
    let _ = o.write_all(s.as_bytes());
    let _ = o.flush();
}

fn chat_tokens(tok: &Tokenizer, tmpl: &ChatTemplate, msgs: &[Message]) -> Result<Vec<u32>> {
    let text = tmpl.render(msgs, true)?;
    Ok(tok.encode(&text, tok.add_bos && !tmpl.starts_with_bos(&text), true))
}

fn run(
    args: &LoadArgs,
    prompt: &str,
    max_tokens: usize,
    chat: bool,
    system: Option<String>,
    sampling: &SamplingArgs,
) -> Result<()> {
    let l = load(args)?;
    let model = Model::load(&l.file)?;
    describe(&model, &args.model);
    let tokens = if chat {
        let tmpl = ChatTemplate::from_gguf(&l.file, &l.tok)?;
        let mut msgs = Vec::new();
        if let Some(s) = system {
            msgs.push(Message { role: "system".into(), content: s });
        }
        msgs.push(Message { role: "user".into(), content: prompt.into() });
        chat_tokens(&l.tok, &tmpl, &msgs)?
    } else {
        l.tok.encode(prompt, l.tok.add_bos, false)
    };
    let mut sess = new_session(&model, args);
    let mut sampler = Sampler::new(sampling.config());
    if !chat {
        print_flush(prompt);
    }
    let (_, stats) = generate(&mut sess, &l.tok, &mut sampler, &tokens, max_tokens, print_flush)?;
    println!();
    stats.print();
    Ok(())
}

fn chat_repl(args: &LoadArgs, system: Option<String>, max_tokens: usize, sampling: &SamplingArgs) -> Result<()> {
    let l = load(args)?;
    let model = Model::load(&l.file)?;
    describe(&model, &args.model);
    let tmpl = ChatTemplate::from_gguf(&l.file, &l.tok)?;
    let mut sess = new_session(&model, args);
    let mut sampler = Sampler::new(sampling.config());
    let mut msgs = Vec::new();
    if let Some(s) = system {
        msgs.push(Message { role: "system".into(), content: s });
    }
    eprintln!("Type a message. Ctrl-D to quit, /reset to clear history.");
    let stdin = std::io::stdin();
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/reset" {
            msgs.retain(|m: &Message| m.role == "system");
            sess.truncate(0);
            continue;
        }
        msgs.push(Message { role: "user".into(), content: line.into() });
        let tokens = chat_tokens(&l.tok, &tmpl, &msgs)?;
        if tokens.len() >= sess.n_ctx {
            eprintln!(
                "[conversation is {} tokens, context is {}; use /reset or a larger --ctx]",
                tokens.len(),
                sess.n_ctx
            );
            msgs.pop();
            continue;
        }
        // Reuse the KV cache for the shared prefix; only run the new tail.
        let common = sess.tokens.iter().zip(&tokens).take_while(|(a, b)| a == b).count();
        let common = common.min(tokens.len() - 1);
        sess.truncate(common);
        // Generate as much as fits in what's left of the context.
        let budget = max_tokens.min(sess.n_ctx - tokens.len());
        let (out, stats) = generate(&mut sess, &l.tok, &mut sampler, &tokens[common..], budget, print_flush)?;
        println!();
        stats.print();
        msgs.push(Message { role: "assistant".into(), content: l.tok.decode(&out) });
    }
    Ok(())
}

fn bench(args: &LoadArgs, pp: usize, tg: usize, reps: usize) -> Result<()> {
    let l = load(args)?;
    let model = Model::load(&l.file)?;
    describe(&model, &args.model);
    let vocab = model.cfg.vocab as u32;
    let mut s = 0x9E37_79B9u32;
    let mut rand_tok = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s % vocab
    };
    let mut sess = Session::new(&model, args.ctx.unwrap_or(pp + tg + 1).max(pp + tg + 1), args.batch);

    // warm-up: touch every weight page once
    sess.forward(&[1, 2, 3, 4], Logits::Last)?;

    let mut pp_rates = Vec::new();
    let mut tg_rates = Vec::new();
    for _ in 0..reps {
        if pp > 0 {
            sess.truncate(0);
            let toks: Vec<u32> = (0..pp).map(|_| rand_tok()).collect();
            let t = Instant::now();
            sess.forward(&toks, Logits::Last)?;
            pp_rates.push(pp as f64 / t.elapsed().as_secs_f64());
        }
        if tg > 0 {
            sess.truncate(0);
            let t = Instant::now();
            for _ in 0..tg {
                sess.forward(&[rand_tok()], Logits::Last)?;
            }
            tg_rates.push(tg as f64 / t.elapsed().as_secs_f64());
        }
    }
    let stat = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        format!("{mean:8.2} ± {sd:.2} tok/s")
    };
    if !pp_rates.is_empty() {
        println!("pp{pp:<5} {}", stat(&pp_rates));
    }
    if !tg_rates.is_empty() {
        println!("tg{tg:<5} {}", stat(&tg_rates));
    }
    ops::prof::report();
    Ok(())
}

fn perplexity(args: &LoadArgs, path: &Path, max_chunks: Option<usize>) -> Result<()> {
    let l = load(args)?;
    let model = Model::load(&l.file)?;
    describe(&model, &args.model);
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let tokens = l.tok.encode(&text, l.tok.add_bos, false);
    let n_ctx = args.ctx.unwrap_or(512);
    let mut sess = Session::new(&model, n_ctx, args.batch);
    let vocab = model.cfg.vocab;
    let n_chunks = (tokens.len() / n_ctx).min(max_chunks.unwrap_or(usize::MAX));
    if n_chunks == 0 {
        bail!("need at least {n_ctx} tokens, file has {}", tokens.len());
    }
    eprintln!("{} tokens, {n_chunks} chunks of {n_ctx}", tokens.len());
    let first = n_ctx / 2;
    let (mut nll, mut count) = (0.0f64, 0usize);
    let t = Instant::now();
    for c in 0..n_chunks {
        let mut chunk = tokens[c * n_ctx..(c + 1) * n_ctx].to_vec();
        if l.tok.add_bos
            && let Some(b) = l.tok.bos
        {
            chunk[0] = b;
        }
        sess.truncate(0);
        let logits = sess.forward(&chunk, Logits::All)?;
        for j in first..n_ctx - 1 {
            let row = &logits[j * vocab..(j + 1) * vocab];
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse = row.iter().map(|&v| ((v - max) as f64).exp()).sum::<f64>().ln() + max as f64;
            nll += lse - row[chunk[j + 1] as usize] as f64;
            count += 1;
        }
        eprint!("[{}]{:.4} ", c + 1, (nll / count as f64).exp());
    }
    eprintln!();
    println!(
        "perplexity: {:.4} over {count} tokens ({:.1} tok/s)",
        (nll / count as f64).exp(),
        (n_chunks * n_ctx) as f64 / t.elapsed().as_secs_f64()
    );
    Ok(())
}
