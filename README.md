# ember

A small LLM inference engine for CPUs, written in Rust from scratch. It loads GGUF files (the format llama.cpp uses), runs Llama-family and Qwen models on quantized weights, and gets within reach of llama.cpp's speed in about 2,900 lines of code.

I built it to learn how inference engines work. The code is meant to be read: every file starts with a comment explaining what it does and why.

```
$ ember run -m qwen2.5-0.5b-instruct-q4_k_m.gguf --chat -p "What is 17*23? Answer briefly."
ember: qwen2.5-0.5b-q4_k_m.gguf | Qwen2 24 layers, dim 896, heads 14/2, vocab 151936 | weights Q5_0x132 Q8_0x14 Q6_Kx12 Q4_Kx12 | 10 threads
17*23 equals 391.

[prompt: 41 tok, 209.2 tok/s | generated: 11 tok, 91.5 tok/s]
```

## Features

- **GGUF loader.** The file is memory-mapped and the weights are never copied. Startup is instant.
- **Models:** Llama 1/2/3, Mistral, TinyLlama, SmolLM (anything with `general.architecture = llama`), Qwen2/2.5, and Qwen3. This covers grouped-query attention, QKV bias (Qwen2), per-head QK norm (Qwen3), tied embeddings, NeoX and standard RoPE, and Llama 3's `rope_freqs` scaling.
- **12 weight formats:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. Together these cover the common `Q4_K_M`, `Q5_K_M`, `Q8_0`, and similar files on Hugging Face.
- **Integer SIMD kernels.** Activations are quantized to 8 bits on the fly and dotted against the quantized weights with AVX2 `maddubs`. Nothing is dequantized to f32 on the hot path.
- **Batched prefill.** The prompt is processed in batches. Each weight block is decoded once per 4-token tile, with per-token partial sums kept in registers.
- **Tokenizers:** SentencePiece (Llama 2 style) and GPT-2 byte-level BPE (Llama 3, Qwen), built from the GGUF metadata. Output matches `llama-tokenize` on every case in the test set.
- **Chat templates.** Each model's own Jinja template (`tokenizer.chat_template`) is rendered with [minijinja](https://github.com/mitsuhiko/minijinja).
- **Multi-turn chat with KV-cache reuse.** Each turn only processes the new tokens.
- **Sampling:** temperature, top-k, top-p, min-p, and repetition penalty.
- **Tools:** `bench` (llama-bench-style throughput), `ppl` (perplexity, same chunking as llama-perplexity), `inspect`, `tokenize`, and an opt-in profiler (`EMBER_PROFILE=1`).

## Quick start

```sh
git clone https://github.com/amanmprojects/ember && cd ember
cargo build --release        # builds for your CPU (.cargo/config.toml sets target-cpu=native)

# any small GGUF works, e.g.
curl -LO https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

./target/release/ember chat -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

## Commands

| Command | What it does |
|---|---|
| `ember run -m M -p "text" [--chat] [--system S] [-n N]` | Generate from a prompt. `--chat` wraps it in the model's chat template. |
| `ember chat -m M [--system S]` | Interactive chat. `/reset` clears history. |
| `ember bench -m M [-p 512] [-n 128] [-r 3]` | Prompt-processing (pp) and generation (tg) speed. |
| `ember ppl -m M -f text.txt [--chunks N] [-c 512]` | Perplexity, comparable with `llama-perplexity`. |
| `ember tokenize -m M "text" [-v]` | Token ids, optionally with each token's text. |
| `ember inspect M [--tensors]` | Metadata and a tensor-type summary. |

Common flags: `-t` threads (default: physical cores), `-c` context length, `-b` prefill batch size, `--temp` (0 = greedy), `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed`.

## Correctness

Perplexity on wikitext-2 (first 10 chunks of 512 tokens), compared with llama.cpp build b12675 on the same files:

| Model | ember | llama.cpp |
|---|---|---|
RESULTS_PPL

Small differences are expected: llama.cpp stores its KV cache in f16 and rounds activations slightly differently.

Tokenization matches `llama-tokenize` exactly for SentencePiece and GPT-2 BPE models on a set of strings covering contractions, Unicode, emoji, code, digit runs, whitespace runs, and special tokens. `cargo test` checks every quant format's integer kernel against a dequantize-then-f32 reference, for every tile size.

## Performance

Intel i7-13620H (6P+4E cores, AVX2, DDR5), 10 threads for ember and llama.cpp's default (6) for llama.cpp. The laptop was running other applications, so expect ±5% noise. Numbers are tokens/second.

| Model | ember pp256 | llama.cpp pp256 | ember tg64 | llama.cpp tg64 |
|---|---|---|---|---|
RESULTS_BENCH

Generation (tg) is limited by memory bandwidth: every generated token streams the whole model from RAM. ember is close to llama.cpp there. Prompt processing (pp) is limited by compute, and that's where llama.cpp's extra work shows. On AVX2 it repacks Q4_0 weights into interleaved row groups so a single instruction works on 8 rows at once. ember doesn't do that yet.

## How it works

A forward pass for one token, from the top:

1. **Embedding.** Dequantize one row of `token_embd.weight`.
2. **For each layer:**
   - RMSNorm, then the Q, K, and V projections. These are quantized matmuls (below).
   - RoPE rotates Q and K by position. Llama GGUFs rotate adjacent pairs; Qwen uses NeoX style (element *i* paired with *i + d/2*).
   - K and V are appended to the KV cache. Attention runs over all cached positions, with each head in parallel. Grouped-query attention means several query heads share one KV head.
   - Output projection, then a residual add.
   - RMSNorm, then the SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`, then a residual add.
3. **Final RMSNorm and LM head.** Logits are computed only for the rows that need them.
4. **Sampling** picks the next token, and the loop feeds it back in.

### Quantized matmul (`src/quant.rs`, `src/ops.rs`)

Almost all of the time goes here (`EMBER_PROFILE=1` shows over 90%). Weights stay in their on-disk block format. A `Q4_K` block, for example, holds 256 weights as 4-bit values, plus 8 six-bit scales and mins and two f16 super-scales: 4.5 bits per weight.

Each matmul does this:

1. **Quantize the activations** to 8-bit blocks (`BlockQ8` for 32-wide formats, `BlockQ8K` with per-16 sums for k-quants).
2. **Split the weight rows across threads** (rayon). Each thread takes a run of rows and dots each row against up to 4 tokens at once.
3. **Decode each weight block once** into small integers plus scales, then for each token: `maddubs` (u8 × i8 → i16 pairs), `madd` with the sub-block scales (→ i32), convert to float, then FMA into a per-token accumulator.

One trick makes this cheaper. `maddubs` needs one unsigned operand, and most formats store values with an offset (Q4_0 is `d * (q - 8)`). So the decoders emit the *unsigned* `q` and treat the offset as a per-block minimum: `d*q + (-8d)`. The minimum's contribution is `min * sum(activations)`, and that sum is precomputed when the activations are quantized. The inner loop then multiplies raw nibbles directly, with no sign fix-ups. Only Q8_0, which is genuinely signed, takes the slower path.

Each format has a single decoder, and both the integer kernel and the f32 dequantizer use it. That way the two can't disagree, which is what the unit test relies on.

### Code map

| File | Lines | Contents |
|---|---|---|
LOC_TABLE

## Limitations and next steps

- **CPU only, x86-64 AVX2 fast path.** Other targets compile and use portable scalar code, which is correct but slow. NEON kernels would be the obvious addition.
- **Prefill is 0.5–0.9× llama.cpp.** The next step is repacking weights at load time into interleaved row groups (as llama.cpp does for Q4_0) so one SIMD instruction works on several rows.
- The KV cache is f32. f16 would halve its memory use and the bandwidth attention needs at long context.
- No IQ (importance-matrix) quant formats, no MoE, no multimodal models.
- One sequence at a time: no batching of independent requests and no server.

## License

MIT
