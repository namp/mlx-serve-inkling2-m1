# mlx-serve-inkling2-m1

Apple M1 / M2 optimizations for **Inkling 2 Mini (`inkling2_loop`)** — developed and tested on an **Apple M1 Max (64 GB)** with **Inkling 2 Mini Uncensored** (and compatible with any Inkling 2 Mini MLX/canonical checkpoint) on top of [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) (`2fe9be8`, MIT License), plus a virtually lossless Q4-attention / Q6-head requantization tool and streaming SSE reasoning fix.

> **Upstream credit & license:** Based on [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) commit `2fe9be834a601c70ef1a597563026340f0e722fe` by David Dalcu, preserved under the original [MIT License](LICENSE) (see [NOTICE](NOTICE) and [README_UPSTREAM.md](README_UPSTREAM.md) for full upstream documentation). No model weights are distributed in this repository.

---

## Highlights (Inkling 2 Mini Uncensored on Apple M1 Max, 64 GB, `--kv-quant 4`, `--ctx-size 1048576`)

Bracketed back-to-back head-to-head benchmark (`OPT_1` → `STOCK` → `OPT_2`) on the **Inkling 2 Mini Uncensored** (TokenAI-zer / OpenUncensored-derived mixed-precision) checkpoint, comparing stock `ddalcu/mlx-serve` (`2fe9be8` + SSE fix) serving the canonical 8-bit-attention / NVFP4-expert uncensored pack against this build serving the Q4-attention / Q6-head uncensored pack:

| Workload | Stock `mlx-serve` (`tok/s` / TTFT) | This Repo Mean (`tok/s` / TTFT) | Peak Observed | Speedup |
|---|---:|---:|---:|---:|
| **1K raw** (`1,214` prompt tok) | `33.87` / `4.04s` | **`45.34`** / `3.21s` | `46.83 tok/s` | **+33.9%** |
| **Python code** (`reasoning_effort=low`) | `35.80` / `0.80s` | **`44.96`** / `0.67s` | `48.48 tok/s` | **+25.6%** |
| **English prose** (`reasoning_effort=low`) | `37.01` / `0.94s` | **`43.61`** / `0.89s` | `47.40 tok/s` | **+17.8%** |
| **Greek prose** (`reasoning_effort=low`) | `35.77` / `0.78s` | **`42.10`** / `0.64s` | `46.18 tok/s` | **+17.7%** |
| **`xhigh` (`temp=1.0, top_p=1.0`)** | `34.88` / `0.63s` | **`41.52`** / `0.58s` | `45.02 tok/s` | **+19.1%** |
| **8K context** (`7,935` prompt tok) | `30.26` / `25.11s` | **`42.38`** / `25.05s` | `43.55 tok/s` | **+40.1%** |
| **18K needle (`18,035` tok, cold / SSD-warm / post-restart)** | `60.8s` cold / `3.7s` repeat wall | **`52.2s` cold (`41.7 tok/s`) / `0.60s` TTFT (`43.9 tok/s`) / `0.55s` TTFT post-restart (`44.1 tok/s`)** | `44.05 tok/s` | **Flat decode speed 1K→18K** |

*(A full 1,048,576-token prompt occupancy test was not benchmarked; 1M context is configured and exposed with 4-bit KV caching, and live-verified up to 18,035 tokens with SSD prefix cache persistence across restarts.)*

---

## What Changed

All engine optimizations live cleanly inside `src/plugins/inkling2/`, plus one streaming reasoning parser fix in `src/host/chat/reasoning/think_tags.zig`:

1. **Native FP16 activations & weights on Apple7/8 (M1/M2, `gen < 15`)** (`plugin.zig`, `layers.zig`, `ups.metal`, `downs.metal`):
   - M1/M2 GPUs lack hardware BF16 ALUs. Automatically runs Inkling-2-Mini in FP16 while applying NVFP4 per-expert `global_scale` in `float32` before narrowing to FP16 so intermediate expert activations never overflow (`INKLING_FORCE_BF16=1` disables).
2. **Fused Q4/Q8 KV Attention with Relative Bias, Sinks & Output Gate** (`qkv_ink2.metal`, `qkv_ink2_merge.metal`, `qkv_ink2_verify.metal`, `qkv_ink2_verify_merge.metal`):
   - Stock `inkling2` ignored quantized KV triples and forced full 48-layer KV dequantization every token (~5 ms penalty by 8K).
   - Reads packed Q4/Q8 KV cache directly with `simdgroup_float8x8` MMA, fuses per-head relative bias + sink + sigmoid output gate, tunes `nsplit = clamp((t_k + 255) / 256, 1, 16)`, and lets 4-bit `o_proj` run on MLX's ~228 GB/s `qmv_fast` instead of ~77 GB/s `oproj.metal`.
3. **Barrier-Free MoE `downs.metal` & Zero-Waste NVFP4 `ups.metal`**:
   - Rewrote `downs.metal` so each SIMD group sweeps all `KSEL * (I / 16)` (`8 × 72 = 576 = 18 × 32`) chunks across `ROWS=4` output columns at `BG=8` (`256` threads/TG) with zero `threadgroup_barrier` and zero idle SIMD lanes (~255 GB/s vs ~93 GB/s in cold-DRAM microbenchmarks).
   - Added an exact 8-value (`uint`) NVFP4 sweep (`288 / 32 = 9` full iterations, `ROWS=2, BG=8`) to `ups.metal` (~257 GB/s vs ~144 GB/s) and fast `uint` 4-bit/8-bit affine paths in `bank_header.metal`.
4. **Coalesced FP16 Loop Message Gates & Hoisted Constants** (`gate.metal`, `kernels.zig`, `plugin.zig`):
   - Converts `gates.{down,forget,message}` from F32 to FP16, transposes `forget`/`message` to `[4, RANK, D]` with threadgroup SRAM staging (`TG=128`), and hoists ~336 per-token constant array allocations onto `layers.Body`.
5. **Opt-In M1 DSpark MTP (`INKLING_FORCE_DRAFT=1`)** (`draft.zig`, `markov_fold.metal`, `qkv_ink2_verify.metal`, `plugin.zig`):
   - Adds FP16 + 8-bit quantized drafter layers (`INKLING_DRAFT_BITS=4|6|8|16`), symmetric per-row INT8 `d.fold` (`103 MB → 51.5 MB`, cutting draft latency from `9.8 ms` to `6.15 ms`), fused Q4 KV verify up to `S=8`, and expected-tokens-per-ms draft count selection.
   - **Why it stays opt-in on M1/M2:** Across Inkling-2-Mini's 48 visits of 128-expert (`top_k=8`) MoE without SME/NAX tiles, verifying `S=4..5` rows (`53–67 ms`) + drafting (`6.2 ms`) needs `~3.0` tokens/round just to break even against our `~20.6 ms` (`45–48.5 tok/s`) pipelined serial path. DSpark reached `51.3 tok/s` on repetitive `1K_raw` copy tasks, but trailed serial on real Python code (`45.6` vs `48.5`), English (`43.1` vs `47.4`), Greek (`41.2` vs `46.2`), `xhigh` (`40.9` vs `45.0`), and 8K (`40.0` vs `42.4`). Keeping M1/M2 serial by default gives the highest real-world speed and saves ~400 MB RAM while leaving `INKLING_FORCE_DRAFT=1` available for copy-heavy workloads.
6. **Inkling SSE `<|content_thinking|>` Separation Fix** (`src/host/chat/reasoning/think_tags.zig`):
   - Latches `<|content_thinking|>` on model-opened reasoning blocks and guards split boundaries on `<|end_message|>` so streamed thinking stays strictly inside `reasoning_content` without leaking into visible `content`.

---

## Virtually Lossless Q4-Attention / Q6-Head Pack (`tools/requant_inkling2_q4attn.py`)

Because Inkling-2-Mini runs 48 visits per token, stock 8-bit `self_attn` projections (`~1,374 MB/tok`) and 8-bit `lm_head` (`~492 MB/tok`) dominate non-MoE bandwidth on M1 Max. Requantizing **only** `self_attn.*_proj` to 4-bit affine (`gs=64`) and `embed_tokens`/`lm_head` to 6-bit affine (`gs=64`) — while keeping **all 24 layers' MoE experts 100% bit-for-bit untouched** from the source checkpoint (including the uncensored weights, with zero safety/refusal wrappers added) — cuts model size from `15.6 GB` to `15.0 GB` and unlocks MLX's 4-bit `qmv_fast` / `verifyQmm` lanes with virtually zero quality loss:

- **KLD on Inkling 2 Mini Uncensored (`tests/inkling2_kld.py`, 2,726 positions across Zig/Python/Markdown):**
  - Mean KLD: `0.0269 ± 0.0015` (median `0.0064`)
  - Top-1 agreement: `93.4%`
  - Perplexity: `3.912` vs `3.902` (`+0.010` ppl)

```bash
python3 tools/requant_inkling2_q4attn.py \
  --src /path/to/inkling2-uncensored-mlx-canonical \
  --dst /path/to/inkling2-uncensored-mlx-q4attn
```

---

## Building & Running

Requires Zig 0.17.0 and macOS Apple Silicon (see [README_UPSTREAM.md](README_UPSTREAM.md) for submodule/MLX build scripts):

```bash
git clone --recursive https://github.com/namp/mlx-serve-inkling2-m1.git
cd mlx-serve-inkling2-m1
./scripts/build-mlx.sh
zig build -Doptimize=ReleaseFast -Dunits=true -j4
```

Serve with 1M context, Q4 KV caching, and tiered RAM + SSD prefix caching (sized for 64 GB Macs running long multi-turn agentic sessions):

```bash
./zig-out/bin/mlx-serve \
  --model /path/to/inkling2-mlx-q4attn \
  --serve --host 127.0.0.1 --port 8085 \
  --ctx-size 1048576 --max-tokens 65536 \
  --os-reserve-gib 16 --max-concurrent 1 \
  --kv-quant 4 --no-vision \
  --prefix-cache-entries 8 --prefix-cache-mem 6GB --prefix-cache-disk 8GB \
  --temp 1.0 --top-p 1.0 --top-k 0
```

> **Prefix-cache sizing tip:** Under `--kv-quant 4`, a `50K–120K`-token session uses `~0.6–1.3 GB` of KV/checkpoint state. Giving the hot RAM tier `6GB` alongside an `8GB` SSD tier avoids LRU eviction races during rapid tool-call loops or multi-thread switching (fall back to `--no-prefix-cache-ram --prefix-cache-disk 2GB` only on tighter RAM/disk budgets; `mlx-serve` keeps a `min(64 GiB, 10% volume)` free-disk reserve before arming the SSD tier).

### Runtime Environment Knobs

| Env Var | Effect |
|---|---|
| `INKLING_FORCE_BF16=1` | Disable M1/M2 native FP16 mode (fall back to BF16) |
| `INKLING_NO_FUSED_QKV=1` | Disable fused Q4/Q8 KV decode attention (`qkv_ink2`) |
| `INKLING_NO_FUSED_QKV_VERIFY=1` | Disable fused Q4/Q8 KV verify attention (`qkv_ink2_verify`) |
| `INKLING_FORCE_DRAFT=1` | Opt in to M1/M2 DSpark MTP (useful for copy-heavy/repetitive workloads) |
| `INKLING_NO_DRAFT=1` | Force serial decode even on Apple9+ |
| `INKLING_DRAFT_BITS=4\|6\|8\|16` | Drafter transformer layer quantization width on M1/M2 (default `8`) |
| `INKLING_DRAFT_MAX=1..7` | Cap drafted tokens per round on M1/M2 (default `7`) |
| `INKLING_NO_FOLD_Q8=1` | Keep Markov `d.fold` in FP16 instead of per-row symmetric INT8 |

### Benchmarking

```bash
python3 tools/bench_inkling2.py \
  --bin ./zig-out/bin/mlx-serve \
  --model /path/to/inkling2-mlx-q4attn
```
