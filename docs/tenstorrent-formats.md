# GGUF weights in Tenstorrent's block-float formats

Tenstorrent's Tensix cores unpack their own block-float formats, BFP8_B and BFP4_B, and not
GGUF's K-quants or IQ formats. A Tenstorrent backend would therefore convert a GGUF's weights to
block-float at load: it would decode them, then round them again. This note measures what that
second rounding costs, with each format emulated exactly on an RTX 3090; no Tenstorrent hardware
was used. The experiment was stopped partway: only Llama 3.1 8B Instruct's Q4_K_M was measured
through the formats (see [what was not done](#not-done)).

## The formats

These come from tt-metal at `628dbed`: the host packer in
`tt_metal/impl/data_format/blockfloat_common.cpp`, which `ttnn.as_tensor` runs for weights, and
Tensix's unpacker in tt-isa-documentation's `FloatBitPatterns.md`.

| Format | Bits per value | Values |
|---|---|---|
| BFP8_B | sign + 7 bits of magnitude | ±m · 2^(E−133), m in 0..127 |
| BFP4_B | sign + 3 | ±m · 2^(E−129), m in 0..7 |
| BFP2_B | sign + 1 | 0 or ±2^(E−127) |

Each 16 values share an 8-bit exponent byte, E, so BFP8_B takes 8.5 bits a weight and BFP4_B 4.5.
A block of 16 is one row of a 16×16 face of a 32×32 tile. tt_transformers stores each matrix
transposed, `[in, out]`, so a block is 16 consecutive output channels at one input index. GGUF's
blocks lie along the input instead.

Packing works as follows:
- E is the largest fp32 exponent field of the block's 16 values.
- Each value's 24-bit mantissa, with its leading 1 made explicit, is shifted down to E. The shift
  truncates.
- The result is rounded to the format's bits, to nearest with ties to even.
- On overflow the value saturates at the largest magnitude; the exponent is not raised.
- Zeros and denormals become +0.

Unpacking is exact. Every BFP8_B and BFP4_B value fits in BF16, so the emulation holds the rounded
weights as BF16 without further loss.

## Tenstorrent's default mix

tt_transformers stores each group of weights in one format, set in
`models/tt_transformers/tt/model_config.py` and overridden per model by
`model_params/<model>/*_decoder_config.json`. `simple_text_demo.py`'s README examples run
performance mode; `ModelArgs` falls back to accuracy mode.

| Weights | Performance | Accuracy, Llama 3 | Accuracy, others such as Qwen3 |
|---|---|---|---|
| q, k, v, attention output | BFP8_B | BFP8_B | BF16 |
| MLP gate and up (w1, w3) | BFP4_B, but BFP8_B in Llama-3.1-8B-Instruct's last layer | BFP8_B | BFP8_B |
| MLP down (w2) | BFP8_B | BFP8_B | BFP8_B |
| LM head | BFP8_B | BFP8_B | BFP8_B |
| Embeddings, norms | BF16 | BF16 | BF16 |

tt_transformers also permutes q's and k's rows into Meta's interleaved RoPE order before it packs
them. Llama's GGUF already holds them in that order. Qwen3's GGUF holds Hugging Face's order, so
the emulation permutes those rows to group their blocks as Tenstorrent does.

## Method

`scripts/tt_bfp.py` loads a GGUF and, as each matrix reaches the GPU, does the following:
1. Decodes it to f32.
2. Rounds it through its group's format in the mix, exactly as tt-metal's packer does.
3. Holds the result as BF16.

The model then runs on LeatEngine's reference path, in f32. The rounding matches tt-metal's packer
bit for bit. This was checked against tt-metal's own `convert_u32_to_bfp` and
`convert_bfp_to_u32`, compiled from its source, on 9.4 million values: Gaussian, outliers, ties,
saturation, zeros and denormals.

Each variant is scored on the prompt path, as `leatengine perplexity --kl-base` scores it. The
data is the first 8 chunks of 512 tokens of wikitext-2's test set, and the reference is
llama.cpp `1b70350`'s logits for the same Q4_K_M file, from its CUDA build with flash attention.
Both are the same as in [benchmarks](benchmarks.md#agreement-with-llamacpp). Divergence from that
reference measures how far a converted GGUF strays from what llama.cpp gives for the file. It
does not measure the quality lost against the original weights; that needed a BF16 reference,
which was not made (see below).

The model is bartowski's `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, SHA-256 `7b064f58…`, run with
LeatEngine at `c0904c4` on an RTX 3090.

## Results

| Llama 3.1 8B Instruct Q4_K_M | Perplexity | KL from llama.cpp | p99 KL | Same top token |
|---|---:|---:|---:|---:|
| As LeatEngine runs it today | 7.984 | 0.0011 | 0.011 | 98.3% |
| Decoded to BF16, reference path (control) | 7.988 | 0.0010 | 0.009 | 98.4% |
| Through the accuracy mix: all BFP8_B | 7.990 | 0.0014 | 0.012 | 98.1% |
| Through the performance mix: gate and up BFP4_B | 8.171 | 0.0255 | 0.310 | 92.5% |

llama.cpp's own perplexity for the file is 7.994. Two runs of the file as-is differed by 0.006 in
perplexity, so differences of that size are noise.

- **BFP8_B costs almost nothing.** A Q4_K_M rounded again to BFP8_B stays as close to llama.cpp as
  the file does in LeatEngine today.
- **BFP4_B does not.** With the MLP's gate and up in BFP4_B, as Tenstorrent's default performance
  mode stores them:
  - Divergence from llama.cpp rises 23-fold.
  - Perplexity rises 2.3%.
  - The top token changes at one position in 13.

  An earlier run also put the down projection in BFP4_B, with a draft of the rounding that
  differed only on rare ties. It scored a perplexity of 8.329, a KL of 0.050 and 89.4% top-1.

For scale, llama.cpp's own CPU and CUDA backends differ by a KL of 0.009 to 0.011 on low-bit files
([benchmarks](benchmarks.md#agreement-with-llamacpp)).

## Conclusion

On this evidence, "run any GGUF on Tenstorrent" survives conversion only at BFP8_B. That costs
8.5 bits a weight, about 1.7 times this Q4_K_M's 4.9, in memory and bandwidth. At the 4.5-bit BFP4_B of
Tenstorrent's default performance mode, a converted Q4_K_M no longer behaves like the file: it
picks a different top token than llama.cpp at one position in 13.

These runs cannot say whether that loss comes from rounding twice or from BFP4_B itself.
BFP4_B's power-of-two scale is coarser than Q4_K's fitted scales and minimums at the same 4.5
bits, so Tenstorrent's own BFP4_B weights may lose about as much. The BF16 comparison was not
run, so this stays open. Until it is, a backend that offers GGUF compatibility should convert
GGUFs to BFP8_B, or else state that its performance mode departs from the file.

## Not done

The experiment was stopped before these runs:

- **Qwen3 8B:** only bartowski's Q4_K_M as LeatEngine runs it today was scored: perplexity
  10.169, KL 0.0027, 97.1% top-1. Its control, performance and accuracy runs were not.
- **The original BF16 weights:** neither model's were run through the mixes. This is variant
  (c), Tenstorrent's own path, and the comparison the question turns on.
- **llama.cpp BF16 references:** none were made, so no row measures the loss from the original
  weights.
- **Q8_0 through the mixes:** not run.
- **Decode path:** not run.
- **Effects outside the weights:** Tenstorrent's BFP8_B activations and KV cache, and its
  matrix engine's LoFi and HiFi2 fidelities, were never in scope. They would apply to every
  variant alike.

## Reproducing

```bash
DEV=NV uv run python scripts/tt_bfp.py Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --kl-base Meta-Llama-3.1-8B-Instruct-Q4_K_M.kld --mix performance \
  --tt-model Llama-3.1-8B-Instruct
```

- `--mix` is `performance`, `accuracy`, `bf16` (the control) or `none` (the file as LeatEngine
  runs it).
- The `.kld` file is from `llama-perplexity -f wiki.test.raw -c 512 --chunks 8 -ngl 99 -fa on
  --kl-divergence-base`.
- A reference-path run takes about 3 minutes on an RTX 3090.
