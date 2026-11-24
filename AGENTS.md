# AGENTS.md

ember is a CPU inference engine for GGUF models in about 3,000 lines of Rust. The README covers why it exists and how it compares with llama.cpp; this file covers building, running, and changing it.

## Build and run

```sh
cargo build --release      # .cargo/config.toml sets target-cpu=native, so the binary only runs on this CPU type
cargo test --release       # kernel-vs-reference tests; debug builds are very slow
cargo clippy --release
./target/release/ember --help
```

Any small GGUF works for a smoke test:

```sh
curl -LO https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
./target/release/ember run -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --chat -p "What is 17*23?"
./target/release/ember chat -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

Subcommands: `run`, `chat`, `bench`, `ppl`, `tokenize`, `inspect`. Flags are in `--help`. `-t` defaults to physical cores (SMT siblings add sync cost and no bandwidth). `EMBER_PROFILE=1` prints a time breakdown (matmul, attention, norm/rope/act, LM head) after `bench`.

`*.gguf` is gitignored. Keep model files out of commits.

## Code map

| File | Contents |
|---|---|
| `src/gguf.rs` | GGUF parser over mmap |
| `src/quant.rs` | Block formats, decoders, activation quantization, AVX2 dot kernels, tests |
| `src/ops.rs` | Threaded quantized matmul, RMSNorm, softmax, RoPE, profiler |
| `src/model.rs` | Model config, weight loading, batched forward pass, f32 KV cache |
| `src/tokenizer.rs` | SentencePiece and GPT-2 byte-level BPE, special tokens, streaming UTF-8 decode |
| `src/sampler.rs` | Repetition penalty, temperature, top-k, top-p, min-p |
| `src/chat.rs` | Chat templates via minijinja (ChatML fallback) |
| `src/main.rs` | CLI; chat REPL reuses the KV cache for the shared prefix |

## Forward pass

Per layer: RMSNorm → Q/K/V matmuls (+ bias on Qwen2, per-head QK norm on Qwen3) → RoPE → append to KV cache → attention, parallel over (token, head), with grouped-query KV sharing → output projection + residual → RMSNorm → SwiGLU `down(silu(gate(x)) * up(x))` + residual. Then final norm and LM head, computed only for the rows that need logits.

## Quantized matmul

Over 90% of time is here. Weights stay in their on-disk block format; nothing is dequantized to f32 on the hot path.

1. Activations are quantized per matmul: `BlockQ8` (32-wide formats, stores `s = d * sum(qs)`) or `BlockQ8K` (k-quants, stores per-16 `bsums`).
2. rayon splits weight rows across threads; each row is dotted against a tile of up to 4 tokens (`tiled!` macro, const `T`), with per-token accumulators in registers.
3. Each weight block is decoded once per tile, then per token: `maddubs` → `madd` (ones or per-16 scales) → `cvtepi32_ps` → `fmadd`.

**Unsigned-weight trick.** `maddubs` needs one unsigned operand, so decoders emit unsigned `q` and fold the format's offset into a per-block minimum (Q4_0: `m = -8d`, Q5_0: `m = -16d`; Q6_K: `mn = 32*sc`, Q3_K: `mn = 4*sc`, both with `dmin = d`). The minimum's contribution is `m * sum(acts)`, precomputed during activation quantization. Only Q8_0 is signed and takes the `_mm256_sign_epi8` path. A new format follows the same pattern.

**One decoder per format.** The integer kernel and `dequantize` share the decoder, which is what `int_dot_matches_dequant` relies on. Change a decoder and both paths change together; add a format by adding its decoder and a case to that test.

Performance lessons from writing the kernels:
- Decoders return `__m256i` registers. Decoding into an array and reloading it dropped Q8_0 to 2 GB/s.
- Bit-unpacking loops must use the AVX2 helpers in `quant::unpack` (`nibbles`, `high_bits`). A scalar Q5_0 loop that LLVM didn't vectorize ran at a quarter speed.
- Accumulate in vector floats across blocks; one horizontal sum per row at the end.
- Measure kernel changes with `cargo test --release kernel_throughput -- --ignored --nocapture`. Unrolls and broadcast loads that looked faster were slower.

## Model and tokenizer details that break silently

- GGUF weight dims are `[cols, rows]` (innermost first).
- RoPE: Llama GGUFs rotate adjacent pairs; Qwen uses NeoX (element `i` with `i + d/2`). Llama 3 `rope_freqs.weight` divides the frequency. Linear `rope.scaling` is applied; YaRN only warns.
- Llama 3 GGUFs omit `add_bos_token`; it defaults to true for that family, as in llama.cpp. Missing BOS cost about 2% perplexity.
- BPE pre-tokenizer is chosen by `tokenizer.ggml.pre`; an unknown value is an error, not a fallback. SmolLM splits `\p{N}` first, then applies the GPT-2 regex.
- BPE merging is heap-based over a linked list; a naive rescan was quadratic (16 s on 40k chars).
- Chat templates need `trim_blocks` and `lstrip_blocks`, set through `SyntaxConfig::builder()` (minijinja 3 has no setter).

## Verifying changes

llama.cpp is the reference. A change to kernels, the forward pass, or the tokenizer is done when it matches llama.cpp on the same file:

- **Perplexity:** `ember ppl -m M -f wiki.test.raw --chunks 10` vs `llama-perplexity -m M -f wiki.test.raw --chunks 10`. Results within about 0.5% are expected (llama.cpp uses an f16 KV cache). The README table holds the current numbers.
- **Tokenizer:** `ember tokenize -m M "text"` vs `llama-tokenize -m M -p "text" --ids`. Output must be identical. `llama-tokenize` expands `\n` escapes, so pass ember the same string through `printf %b`.
- **Speed:** `ember bench -m M -p 256 -n 64` vs `llama-bench -m M -p 256 -n 64`. Update the README table if numbers move.

## Next steps

- Repack weights at load time into interleaved row groups (llama.cpp does this for Q4_0 on AVX2) so one instruction covers several rows. This is the main prefill gap.
- f16 KV cache.
- NEON kernels; non-x86 targets currently run the portable scalar path.
