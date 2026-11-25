#!/usr/bin/env python3
"""
Graft Lucebox's Inkling-2-Mini-Fast weight tuning onto any local Inkling-2-Mini MLX pack
(e.g. an uncensored Q4-attention / Q6-head pack) with ~95 MB of extra disk space.

Background:
  Lucebox/Inkling-2-Mini-Fast-GGUF (Apache 2.0) modifies ONLY the 24 BF16
  `self_attn.o_proj` matrices (every other tensor — MoE experts, embeddings/head,
  Q/K/V/R/gate/sink, loop controllers, loop_res_query — is bit-identical to base
  Inkling-2-Mini). Because the Fast delta is >99.5% orthogonal to abliteration deltas
  (max |cos| = 0.00512 across all 24 layers), task arithmetic W_new = W_src + alpha * dW_fast
  composes both cleanly, cutting output tokens ~50-53% (~66% less <think> bloat).

Usage:
  1. Extract the exact 24-layer BF16 delta via HTTP Range requests (~678 MB total
     instead of downloading 42 GB of full GGUFs):

       python3 tools/graft_lucebox_fast.py extract \
         --out ./lucebox_fast_oproj_deltas.safetensors

  2. Build a zero-copy hardlinked mlx-serve pack (writes only model-fast-oproj.safetensors
     and relies on mlx-serve's carrier-of-record index resolution):

       python3 tools/graft_lucebox_fast.py apply \
         --deltas ./lucebox_fast_oproj_deltas.safetensors \
         --pack /path/to/inkling2-mlx-q4attn \
         --ref-q8 /path/to/inkling2-mlx-8bit \
         --out /path/to/inkling2-mlx-fast-q4attn \
         --bits 4 --alpha 1.0

     (--ref-q8 is optional: if omitted, W_src is dequantized directly from --pack.)
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import struct
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

DEFAULT_BASE_URL = (
    "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/"
    "Inkling-2-Mini-UD-Q8_K_XL.gguf"
)
DEFAULT_FAST_URL = (
    "https://huggingface.co/Lucebox/Inkling-2-Mini-Fast-GGUF/resolve/main/"
    "Inkling-2-Mini-Fast-UD-Q8_K_XL.gguf"
)
GGUF_HEADER_BYTES = 12_400_000  # 200K vocab + 424K merges header is ~12.0 MB
FAST_SHARD = "model-fast-oproj.safetensors"

COPY_FILES = [
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


def fetch_range(url: str, start: int, nbytes: int, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{start + nbytes - 1}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                buf = resp.read()
                if len(buf) != nbytes:
                    raise IOError(f"Short read: got {len(buf)}, expected {nbytes}")
                return buf
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))


def parse_gguf_header(data: bytes):
    pos = 0
    magic, _ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", data, pos)
    if magic != b"GGUF":
        raise ValueError(f"Bad GGUF magic: {magic!r}")
    pos += 24

    def read_str():
        nonlocal pos
        n = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        s = data[pos : pos + n].decode("utf-8", "replace")
        pos += n
        return s

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_val(t):
        nonlocal pos
        if t in sizes:
            pos += sizes[t]
        elif t == 8:
            read_str()
        elif t == 9:
            et, cnt = struct.unpack_from("<IQ", data, pos)
            pos += 12
            if et in sizes:
                pos += sizes[et] * cnt
            else:
                for _ in range(cnt):
                    skip_val(et)
        else:
            raise ValueError(f"Unknown GGUF value type {t}")

    alignment = 32
    for _ in range(n_kv):
        k = read_str()
        vt = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if k == "general.alignment":
            alignment = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        else:
            skip_val(vt)

    tensors = {}
    for _ in range(n_tensors):
        name = read_str()
        nd = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = struct.unpack_from(f"<{nd}Q", data, pos)
        pos += 8 * nd
        gtype, off = struct.unpack_from("<IQ", data, pos)
        pos += 12
        tensors[name] = (dims, gtype, off)

    data_off = ((pos + alignment - 1) // alignment) * alignment
    return data_off, tensors


def bf16_bytes_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32).reshape(shape)


def cmd_extract(args):
    t0 = time.time()
    print("Fetching GGUF headers (~12 MB each)...", flush=True)
    b_off, bt = parse_gguf_header(fetch_range(args.base_url, 0, GGUF_HEADER_BYTES))
    f_off, ft = parse_gguf_header(fetch_range(args.fast_url, 0, GGUF_HEADER_BYTES))

    def fetch_layer(layer: int):
        key = f"blk.{layer}.attn_output.weight"
        b_dims, b_type, b_rel = bt[key]
        f_dims, f_type, f_rel = ft[key]
        assert b_dims == f_dims == (3072, 2304), f"Unexpected dims {b_dims} vs {f_dims}"
        assert b_type == f_type == 30, f"Expected BF16 (30), got {b_type}, {f_type}"
        nbytes = b_dims[0] * b_dims[1] * 2
        w_b = bf16_bytes_to_f32(fetch_range(args.base_url, b_off + b_rel, nbytes), (2304, 3072))
        w_f = bf16_bytes_to_f32(fetch_range(args.fast_url, f_off + f_rel, nbytes), (2304, 3072))
        dw = w_f - w_b
        rel = float(np.linalg.norm(dw) / (np.linalg.norm(w_b) + 1e-12))
        return layer, dw, rel

    deltas = {}
    print("Fetching all 24 BF16 o_proj tensors (~678 MB total)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_layer, l) for l in range(24)]
        for fut in concurrent.futures.as_completed(futs):
            l, dw, rel = fut.result()
            deltas[l] = dw
            print(f"  [L{l:2d}] ||dW_fast||/||W_base|| = {rel:.5f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_map = {
        f"model.language_model.layers.{l}.self_attn.o_proj.delta": mx.array(deltas[l]).astype(mx.bfloat16)
        for l in range(24)
    }
    mx.save_safetensors(
        str(out_path),
        save_map,
        metadata={
            "base_url": args.base_url,
            "fast_url": args.fast_url,
            "license": "apache-2.0",
        },
    )
    print(f"Saved 24 BF16 o_proj deltas to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


def load_source_oproj(src_dir: Path, key_template: str, default_bits: int, default_gs: int):
    idx = json.loads((src_dir / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((src_dir / "config.json").read_text())
    qcfg = cfg.get("quantization") or cfg.get("quantization_config") or {}
    shard_to_layers = {}
    for l in range(24):
        pfx = key_template.format(l=l)
        shard = idx[pfx + ".weight"]
        shard_to_layers.setdefault(shard, []).append((l, pfx))

    weights = {}
    for shard, entries in sorted(shard_to_layers.items()):
        tensors = mx.load(str(src_dir / shard))
        for l, pfx in entries:
            w = tensors[pfx + ".weight"]
            if (pfx + ".scales") in tensors:
                layer_q = qcfg.get(pfx, {})
                bits = int(layer_q.get("bits", qcfg.get("bits", default_bits)))
                gs = int(layer_q.get("group_size", qcfg.get("group_size", default_gs)))
                w = mx.dequantize(w, tensors[pfx + ".scales"], tensors[pfx + ".biases"], group_size=gs, bits=bits)
            weights[l] = w.astype(mx.float32)
        del tensors
        mx.eval(*weights.values())
    return weights


def cmd_apply(args):
    t0 = time.time()
    pack_dir = Path(args.pack)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    deltas = mx.load(str(args.deltas))
    if args.ref_q8:
        ref_dir = Path(args.ref_q8)
        ref_idx = json.loads((ref_dir / "model.safetensors.index.json").read_text())["weight_map"]
        tmpl = (
            "model.layers.{l}.attn.o_proj"
            if "model.layers.0.attn.o_proj.weight" in ref_idx
            else "model.language_model.layers.{l}.self_attn.o_proj"
        )
        src_w = load_source_oproj(ref_dir, tmpl, default_bits=8, default_gs=64)
    else:
        src_w = load_source_oproj(
            pack_dir,
            "model.language_model.layers.{l}.self_attn.o_proj",
            default_bits=4,
            default_gs=64,
        )

    out_tensors = {}
    for l in range(24):
        dst_pfx = f"model.language_model.layers.{l}.self_attn.o_proj"
        w_new = (src_w[l] + args.alpha * deltas[dst_pfx + ".delta"].astype(mx.float32)).astype(mx.bfloat16)
        if args.bits == "bf16":
            out_tensors[dst_pfx + ".weight"] = w_new
        else:
            b = int(args.bits)
            qw, qs, qb = mx.quantize(w_new, group_size=64, bits=b)
            out_tensors[dst_pfx + ".weight"] = qw
            out_tensors[dst_pfx + ".scales"] = qs.astype(mx.bfloat16)
            out_tensors[dst_pfx + ".biases"] = qb.astype(mx.bfloat16)
    mx.eval(*out_tensors.values())

    fast_shard_path = out_dir / FAST_SHARD
    mx.save_safetensors(
        str(fast_shard_path),
        out_tensors,
        metadata={"source": "Lucebox/Inkling-2-Mini-Fast-GGUF", "alpha": str(args.alpha), "bits": args.bits},
    )

    # Hardlink all untouched safetensors shards (including vision & mtp) for zero disk duplication.
    for src_st in sorted(pack_dir.glob("*.safetensors")):
        if src_st.name == FAST_SHARD:
            continue
        dst_st = out_dir / src_st.name
        if dst_st.exists() or dst_st.is_symlink():
            dst_st.unlink()
        os.link(src_st, dst_st)

    for fname in COPY_FILES:
        src_f = pack_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, out_dir / fname)

    # Point model.safetensors.index.json o_proj keys to FAST_SHARD (mlx-serve carrier-of-record).
    idx = json.loads((pack_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    for l in range(24):
        dst_pfx = f"model.language_model.layers.{l}.self_attn.o_proj"
        wm[dst_pfx + ".weight"] = FAST_SHARD
        if args.bits == "bf16":
            wm.pop(dst_pfx + ".scales", None)
            wm.pop(dst_pfx + ".biases", None)
        else:
            wm[dst_pfx + ".scales"] = FAST_SHARD
            wm[dst_pfx + ".biases"] = FAST_SHARD
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")

    # Update config.json quantization entries and stamp a distinct SSD prefix-cache fingerprint.
    cfg = json.loads((pack_dir / "config.json").read_text())
    for qkey in ("quantization", "quantization_config"):
        if qkey in cfg and isinstance(cfg[qkey], dict):
            for l in range(24):
                dst_pfx = f"model.language_model.layers.{l}.self_attn.o_proj"
                if args.bits == "bf16":
                    cfg[qkey].pop(dst_pfx, None)
                else:
                    cfg[qkey][dst_pfx] = {"bits": int(args.bits), "group_size": 64, "mode": "affine"}
    cfg["_fast_oproj_graft"] = f"lucebox-fast-v1-b{args.bits}-a{args.alpha:g}"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print(
        f"Built {out_dir} (bits={args.bits}, alpha={args.alpha}) in {time.time() - t0:.2f}s | "
        f"{FAST_SHARD}: {fast_shard_path.stat().st_size / 1e6:.1f} MB (base shards hardlinked)"
    )


def main():
    ap = argparse.ArgumentParser(description="Extract and graft Lucebox Inkling-2-Mini-Fast o_proj deltas.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="HTTP-Range extract the 24 BF16 o_proj deltas from Hugging Face GGUFs.")
    ex.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ex.add_argument("--fast-url", default=DEFAULT_FAST_URL)
    ex.add_argument("--out", default="lucebox_fast_oproj_deltas.safetensors")
    ex.add_argument("--workers", type=int, default=6)
    ex.set_defaults(func=cmd_extract)

    ap_cmd = sub.add_parser("apply", help="Build a zero-copy hardlinked mlx-serve pack with the grafted o_proj deltas.")
    ap_cmd.add_argument("--deltas", required=True, help="Path to extracted BF16 deltas safetensors")
    ap_cmd.add_argument("--pack", required=True, help="Existing mlx-serve Inkling-2-Mini pack directory")
    ap_cmd.add_argument("--ref-q8", default=None, help="Optional higher-precision (8-bit/BF16) source pack for W_src")
    ap_cmd.add_argument("--out", required=True, help="Output directory for the grafted pack")
    ap_cmd.add_argument("--bits", default="4", choices=["4", "6", "8", "bf16"])
    ap_cmd.add_argument("--alpha", type=float, default=1.0)
    ap_cmd.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
