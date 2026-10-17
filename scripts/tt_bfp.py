"""Scores a model with its weights rounded through Tenstorrent Tensix's block-float formats, BFP8_B
and BFP4_B, as tt_transformers stores them, against llama.cpp's logits as `leatengine perplexity
--kl-base` scores them: how much a GGUF loses converted to them at load.

    DEV=NV uv run python scripts/tt_bfp.py MODEL.gguf --kl-base BASE.kld [BASE.kld ...] \\
        [--mix performance|accuracy|bf16|none] [--tt-model Llama-3.1-8B-Instruct] [--chunks N]

Each matrix is decoded, rounded through the format tt_transformers (tt-metal's
models/tt_transformers/tt/model_config.py) gives its group in the mix, and held as BF16, which
holds every block-float value exactly; it then runs on the reference path, slowly. The rounding is
tt-metal's host packer's, which ttnn.as_tensor runs (tt_metal/impl/data_format/
blockfloat_common.cpp); Tensix unpacks what it stores exactly. Activations, the KV cache and the
matrix engine's fidelity are not emulated. --tt-model names the model as tt_transformers keys its
settings, by HF_MODEL's last part. Prints a JSON line per base, as `perplexity --json` does.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tinygrad import Tensor, dtypes

from leatengine import bench, gguf
from leatengine.engine import Engine
from leatengine.model import Config
from leatengine.quant import GGMLType, QTensor

# magnitude bits of a value of BFP8_B, BFP4_B, BFP2_B, beside its sign; or 0 for BF16
BFP8, BFP4, BFP2, BF16 = 7, 3, 1, 0
BLOCK = 16  # values that share an exponent: a row of a 16x16 face of a 32x32 tile

# tt_transformers' groups of the matrices, by GGUF's names
GROUPS = {
    "attn_q": "wqkv", "attn_k": "wqkv", "attn_v": "wqkv", "attn_output": "wo",
    "ffn_gate": "ff1_3", "ffn_up": "ff1_3", "ffn_down": "ff2",
    "output": "lm_head", "token_embd": "embedding",
}  # fmt: skip
# models whose accuracy mode keeps attention in BFP8_B rather than BF16 (model_config.py)
BFP8_ATTENTION = ("Llama-3", "Meta-Llama-3", "Mistral-7B", "Phi-3-mini", "phi-4")
# per-layer settings of model_params/<model>/performance_decoder_config.json
PERFORMANCE_LAYERS = {"Llama-3.1-8B-Instruct": {31: {"ff1_3": BFP8}}}


def mix(name: str, model: str) -> dict[int | None, dict[str, int]]:
    """The format of each group, of all layers (None) and of those set apart: the embedding stays
    BF16 and the LM head is BFP8_B in every mode, as simple_text_demo.py loads it."""
    if name == "bf16":
        return {None: dict.fromkeys(set(GROUPS.values()), BF16)}
    rest = {"lm_head": BFP8, "embedding": BF16}
    if name == "performance":
        mlp = {"ff1_3": BFP4, "ff2": BFP8}
        return {None: {"wqkv": BFP8, "wo": BFP8} | mlp | rest} | PERFORMANCE_LAYERS.get(model, {})
    attention = BFP8 if model.startswith(BFP8_ATTENTION) else BF16
    return {None: {"wqkv": attention, "wo": attention, "ff1_3": BFP8, "ff2": BFP8} | rest}


def bfp(w: Tensor, bits: int) -> Tensor:
    """w (rows, cols) through a block-float format of `bits` bits of magnitude, as tt-metal's
    convert_u32_to_bfp packs it and Tensix unpacks it, f32 in and out. tt_transformers holds a
    matrix transposed, (cols, rows), so a block is 16 rows of a column: they share the largest
    of their exponents; each value's 24-bit mantissa, its leading 1 made explicit, is shifted
    down to it, truncating, then rounded to `bits` bits, to nearest, ties to even, and saturated.
    Zeros and denormals are 0. Blocks below 2^-103 are not exact."""
    rows, cols = w.shape
    x = w.float().reshape(rows // BLOCK, BLOCK, cols)
    exp = (x.bitcast(dtypes.uint32) >> 23) & 0xFF
    shared = exp.max(1, keepdim=True).cast(dtypes.int32)

    def pow2(e: Tensor) -> Tensor:  # 2^(e - 127)
        return (e.maximum(1).cast(dtypes.uint32) << 23).bitcast(dtypes.float32)

    aligned = (exp > 0).where((x.abs() / pow2(shared - 23)).floor(), 0)  # < 2^24, so exact
    q = (aligned / 2 ** (24 - bits)).round().minimum(2**bits - 1)
    return (q * pow2(shared - (bits - 1)) * x.sign()).reshape(rows, cols)


def rounder(formats: dict[int | None, dict[str, int]], hf_order: bool, head_dim: int):
    """gguf._view, its matrices rounded through their formats and held as BF16. Of a model whose
    GGUF holds q and k in Hugging Face's order, rotate-half's, as Qwen3's, tt_transformers' rows
    are in Meta's order, which its convert_hf_to_meta permutes them to: so its blocks too."""
    view = gguf._view

    def rounded(raw: Tensor, t: gguf.TensorInfo) -> QTensor:
        w = view(raw, t)
        parts = t.name.split(".")
        group = GROUPS.get(parts[-2])
        if group is None or len(t.shape) != 2:
            return w
        layer = int(parts[1]) if parts[0] == "blk" else None
        bits = (formats[None] | formats.get(layer, {}))[group]
        x = w.dequant()
        if bits:
            meta = hf_order and parts[-2] in ("attn_q", "attn_k")
            rows, cols = t.shape
            if meta:  # row a * head_dim / 2 + j of each head to 2j + a
                x = x.reshape(-1, 2, head_dim // 2, cols).transpose(1, 2).reshape(rows, cols)
            x = bfp(x, bits)
            if meta:
                x = x.reshape(-1, head_dim // 2, 2, cols).transpose(1, 2).reshape(rows, cols)
        return QTensor(x.cast(dtypes.bfloat16).contiguous().realize(), GGMLType.BF16, t.shape)

    return rounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--kl-base", type=Path, nargs="+", required=True)
    parser.add_argument("--mix", choices=("performance", "accuracy", "bf16", "none"),
                        default="performance")  # fmt: skip
    parser.add_argument("--tt-model", default="", help="as tt_transformers names it")
    parser.add_argument("--chunks", type=int)
    args = parser.parse_args()
    if args.mix != "none":
        config = Config.from_gguf(gguf.GGUF.open(args.model).metadata)
        hf_order = config.rope_halves
        gguf._view = rounder(mix(args.mix, args.tt_model), hf_order, config.head_dims[0])
    ctx = bench.base_chunk(args.kl_base[0])
    engine = Engine(args.model, max_context=ctx, prefill_chunk=ctx)
    for base in args.kl_base:
        result = bench.kl_divergence(engine, base, args.chunks)
        record = {"model": args.model.stem, "mix": args.mix, "tt_model": args.tt_model}
        print(json.dumps(record | {"base": base.stem} | asdict(result)), flush=True)


if __name__ == "__main__":
    main()
