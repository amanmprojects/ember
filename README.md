# ember

A small LLM inference engine for CPUs, written in Rust from scratch. It loads GGUF files (the format llama.cpp uses), runs Llama-family and Qwen models on quantized weights, and gets within reach of llama.cpp's speed in about 3,000 lines of code. It supports Llama 1/2/3, Mistral, SmolLM, Qwen2/2.5 and Qwen3, with 13 weight formats from F32 down to Q2_K.

```
$ ember run -m qwen2.5-0.5b-instruct-q4_k_m.gguf --chat -p "What is 17*23? Answer briefly."
ember: qwen2.5-0.5b-q4_k_m.gguf | Qwen2 24 layers, dim 896, heads 14/2, vocab 151936 | weights Q5_0x132 Q8_0x14 Q6_Kx12 Q4_Kx12 | 10 threads
17*23 equals 391.

[prompt: 41 tok, 209.2 tok/s | generated: 11 tok, 91.5 tok/s]
```

## Why

There's no practical reason. If you want to run models locally, use llama.cpp: it's faster, supports far more models and hardware, and is maintained by a lot of people.

I wrote this to learn how inference works. Using llama.cpp teaches you which flags to pass. Building an engine forces you to answer the questions underneath:

- **What is actually inside a `Q4_K_M` file?** You have to unpack the bits by hand to find out. A "4-bit" weight turns out to be a nibble plus 6-bit sub-block scales plus an f16 super-scale. That's why it costs 4.5 bits per weight, not 4.
- **Why is generating text slow?** Each new token reads every weight in the model from RAM once. Profiling shows generation is limited by memory bandwidth, not arithmetic. That's why a smaller quantization runs faster, and why adding threads past the core count stops helping.
- **Why is reading the prompt fast?** Prompt tokens can be processed together, so each weight block is read once and reused for several tokens. This changes the bottleneck from memory to compute.
- **What do SIMD intrinsics buy you?** Kernel speeds swung 2x in both directions while I was writing these. A plain loop that LLVM didn't vectorize ran Q5_0 at a quarter of the speed of the other formats.
- **Why do tokenizers have so many special cases?** Many small rules turn out to matter: a leading space, a missing BOS token, a regex that splits digits differently. Each one silently changes the model's output.

The project stays small enough to read in one sitting: one file per concern, no framework, about 3,000 lines. Each file starts with a comment explaining what it does and why.

Being small doesn't mean being sloppy. llama.cpp serves as the reference: every model's perplexity and every tokenizer case is checked against it, so "it seems to work" isn't the bar.

**Why Rust?** One language covers both the safe parts (parsing untrusted files, string handling, the CLI) and the unsafe parts (AVX2 intrinsics, raw pointer writes from worker threads). The compiler marks exactly where the unsafe code is. **Why CPU only?** Nearly every machine has one, and on a CPU the performance limits are easy to see and measure.

## Correctness

Perplexity on wikitext-2 (first 10 chunks of 512 tokens), compared with llama.cpp build b12675 on the same files:

| Model | ember | llama.cpp |
|---|---|---|
| TinyLlama 1.1B Q4_K_M | 25.39 | 25.40 |
| TinyLlama 1.1B Q4_0 | 25.66 | 25.65 |
| TinyLlama 1.1B Q5_K_M | 24.72 | 24.75 |
| TinyLlama 1.1B Q2_K (Q2_K + Q3_K weights) | 29.39 | 29.29 |
| Qwen2.5 0.5B Q4_K_M | 17.08 | 17.06 |
| Qwen3 0.6B Q8_0 | 23.73 | 23.82 |
| Llama 3.2 1B Q3_K_L (Q3_K + Q5_K weights) | 17.40 | 17.41 |

Every model is within 0.4%. Small differences are expected: llama.cpp stores its KV cache in f16 and rounds activations slightly differently.

Tokenization matches `llama-tokenize` exactly for SentencePiece and GPT-2 BPE models on a set of strings covering contractions, Unicode, emoji, code, digit runs, whitespace runs, and special tokens. `cargo test` checks every quant format's integer kernel against a dequantize-then-f32 reference, for every tile size.

## Performance

Intel i7-13620H (6P+4E cores, AVX2, DDR5), 10 threads for ember and llama.cpp's default (6) for llama.cpp. The laptop was running other applications, so expect ±5% noise. Numbers are tokens/second.

| Model | ember pp256 | llama.cpp pp256 | ember tg64 | llama.cpp tg64 |
|---|---|---|---|---|
| TinyLlama 1.1B Q4_K_M | 151 | 241 | 63 | 60 |
| TinyLlama 1.1B Q4_0 | 121 | 229 | 60 | 72 |
| Qwen2.5 0.5B Q4_K_M | 250 | 306 | 91 | 91 |
| Qwen3 0.6B Q8_0 | 234 | 240 | 65 | 56 |
| Llama 3.2 1B Q3_K_L | 140 | 250 | 52 | 58 |

Generation (tg) is limited by memory bandwidth: every generated token streams the whole model from RAM. ember is at 0.8–1.15× llama.cpp there. Prompt processing (pp) is limited by compute, and ember is at 0.5–1.0× llama.cpp there. On AVX2 it repacks Q4_0 weights into interleaved row groups so a single instruction works on 8 rows at once. ember doesn't do that yet.

## Limitations

- x86-64 AVX2 is the only fast path. Other CPUs run portable scalar code: correct but slow.
- Prefill is 0.5–1.0× llama.cpp (see above). The KV cache is f32.
- No IQ quant formats, MoE, multimodal models, batching of independent requests, or server.

Build instructions, the code map, and how the kernels work are in [AGENTS.md](AGENTS.md).

## License

MIT
