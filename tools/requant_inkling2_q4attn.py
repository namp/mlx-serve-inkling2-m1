#!/usr/bin/env python3
"""
Create an M1/M2-optimized Q4-attention / Q6-head Inkling-2-Mini MLX pack while
preserving all 24 layers' MoE experts bit-for-bit untouched.

What gets requantized (to FP16 affine, group_size=64):
  - All 24 layers' self_attn.{q,k,v,o,gate,r,sink}_proj -> 4-bit (unlocks 2x faster
    attention bandwidth + MLX 4-bit qmv_fast / verifyQmm lanes)
  - lm_head and model.language_model.embed_tokens -> 6-bit (preserves vocab/multilingual
    and tool-calling accuracy while cutting lm_head bytes by 25%)

What stays bit-for-bit untouched:
  - All 24 layers' MoE experts (L0-19 NVFP4 + L20-23 4-bit affine)
  - Norms, depth queries, short convs, routers, loop message gates, and mtp.safetensors

Measured quality on Inkling-2-Mini (tests/inkling2_kld.py, 2,726 positions):
  mean KLD 0.0269 +/- 0.0015, median 0.0064, top-1 agreement 93.4%, ppl 3.912 vs 3.902 (+0.010).

Usage:
  python3 tools/requant_inkling2_q4attn.py \
      --src /path/to/inkling2-mlx-canonical \
      --dst /path/to/inkling2-mlx-q4attn \
      [--mtp /path/to/mtp.safetensors]
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx

ATTN_PROJS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "r_proj",
    "sink_proj",
)
HEAD_PREFIXES = ("lm_head", "model.language_model.embed_tokens")


def target_bits(prefix: str, attn_bits: int, head_bits: int) -> int | None:
    if prefix in HEAD_PREFIXES:
        return head_bits
    if (
        prefix.startswith("model.language_model.")
        and ".self_attn." in prefix
        and prefix.rsplit(".", 1)[-1] in ATTN_PROJS
    ):
        return attn_bits
    return None


def dequantize_if_needed(
    weights: dict[str, mx.array],
    prefix: str,
    qcfg: dict,
    default_bits: int,
    default_gs: int,
) -> mx.array:
    w = weights[f"{prefix}.weight"]
    sc_key = f"{prefix}.scales"
    bi_key = f"{prefix}.biases"
    if sc_key not in weights:
        return w.astype(mx.float16)
    meta = qcfg.get(prefix) if isinstance(qcfg.get(prefix), dict) else {}
    bits = int(meta.get("bits", default_bits))
    gs = int(meta.get("group_size", default_gs))
    mode = meta.get("mode", "affine")
    biases = weights.get(bi_key)
    out = mx.dequantize(w, weights[sc_key], biases, group_size=gs, bits=bits, mode=mode)
    return out.astype(mx.float16)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Q4-attn / Q6-head Inkling-2-Mini MLX pack")
    ap.add_argument("--src", required=True, help="Source Inkling-2-Mini MLX directory")
    ap.add_argument("--dst", required=True, help="Destination directory (created if missing)")
    ap.add_argument("--mtp", default=None, help="Optional path to mtp.safetensors to hardlink/copy")
    ap.add_argument("--vision", default=None, help="Optional path to model-vision.safetensors to hardlink/copy")
    ap.add_argument("--attn-bits", type=int, default=4, choices=(4, 6, 8))
    ap.add_argument("--head-bits", type=int, default=6, choices=(4, 6, 8))
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    qcfg = cfg.get("quantization") or {}
    default_bits = int(qcfg.get("bits", 8))
    default_gs = int(qcfg.get("group_size", 64))

    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        weight_map: dict[str, str] = idx["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = sorted(p.name for p in src.glob("model*.safetensors"))
        weight_map = {}

    new_qcfg = dict(qcfg)
    new_qcfg["bits"] = args.attn_bits
    new_qcfg["group_size"] = args.group_size
    new_qcfg["mode"] = "affine"

    # Preserve existing non-target quantization entries (e.g. L0-19 NVFP4 experts;
    # L20-23 experts default to top-level 4-bit gs=64 affine if they had no override).
    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    requant_count = 0
    t0 = time.perf_counter()

    for shard in shard_files:
        print(f"Processing {shard}...", flush=True)
        tensors = mx.load(str(src / shard))
        prefixes = {
            k[:-7] for k in tensors.keys() if k.endswith(".weight")
        }
        handled: set[str] = set()
        out_tensors: dict[str, mx.array] = {}

        for prefix in sorted(prefixes):
            bits = target_bits(prefix, args.attn_bits, args.head_bits)
            if bits is None:
                continue
            dense_fp16 = dequantize_if_needed(tensors, prefix, qcfg, default_bits, default_gs)
            qw, qs, qb = mx.quantize(
                dense_fp16, group_size=args.group_size, bits=bits, mode="affine"
            )
            mx.eval(qw, qs, qb)
            out_tensors[f"{prefix}.weight"] = qw
            out_tensors[f"{prefix}.scales"] = qs.astype(mx.float16)
            out_tensors[f"{prefix}.biases"] = qb.astype(mx.float16)
            handled.update((f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"))
            new_qcfg[prefix] = {
                "bits": bits,
                "group_size": args.group_size,
                "mode": "affine",
            }
            requant_count += 1

        # Copy all untouched tensors (including all 24 layers' MoE experts) bit-for-bit.
        for k, v in tensors.items():
            if k in handled:
                continue
            out_tensors[k] = v

        # Ensure any untouched quantized linear that inherited the old default_bits
        # keeps an explicit override if default_bits != args.attn_bits.
        if default_bits != args.attn_bits:
            for prefix in sorted(prefixes):
                if target_bits(prefix, args.attn_bits, args.head_bits) is not None:
                    continue
                if f"{prefix}.scales" in out_tensors and prefix not in new_qcfg:
                    w = out_tensors[f"{prefix}.weight"]
                    sc = out_tensors[f"{prefix}.scales"]
                    # Infer actual bits from packed uint32 columns vs scales dtype
                    if sc.dtype == mx.uint8:
                        continue
                    # Layers 20-23 experts are typically already 4-bit gs=64; detect dynamically.
                    inferred_bits = 4 if (w.shape[-1] * 32) // (sc.shape[-1] * default_gs) == 4 else default_bits
                    if inferred_bits != args.attn_bits:
                        new_qcfg[prefix] = {
                            "bits": inferred_bits,
                            "group_size": default_gs,
                            "mode": "affine",
                        }

        out_path = dst / shard
        mx.save_safetensors(str(out_path), out_tensors)
        for k, v in out_tensors.items():
            new_weight_map[k] = shard
            total_bytes += v.nbytes

    cfg["quantization"] = new_qcfg
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map},
            indent=2,
        )
        + "\n"
    )

    for fname in (
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        p = src / fname
        if p.exists():
            shutil.copy2(p, dst / fname)

    for opt_path, default_name in (
        (args.mtp, "mtp.safetensors"),
        (args.vision, "model-vision.safetensors"),
    ):
        if default_name in shard_files and not opt_path:
            continue
        side_src = Path(opt_path) if opt_path else (src / default_name)
        if side_src.exists():
            side_dst = dst / default_name
            if side_dst.exists():
                side_dst.unlink()
            try:
                os.link(side_src, side_dst)
            except OSError:
                shutil.copy2(side_src, side_dst)
            print(f"Linked/copied {side_src.name}")

    dt = time.perf_counter() - t0
    print(
        f"Done in {dt:.1f}s: requantized {requant_count} projections "
        f"(attn={args.attn_bits}-bit, head={args.head_bits}-bit), "
        f"total weight bytes={total_bytes / 1e9:.2f} GB -> {dst}"
    )


if __name__ == "__main__":
    main()
