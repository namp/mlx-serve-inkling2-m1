# mlx-serve-inkling2-m1

Apple M1 / M2 optimizations for **Inkling 2 Mini (`inkling2_loop`)** — developed and tested on an **Apple M1 Max (64 GB)** with **Inkling 2 Mini Uncensored** (and compatible with any Inkling 2 Mini MLX/canonical checkpoint) on top of [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) (`2fe9be8`, MIT License), plus a virtually lossless Q4-attention / Q6-head requantization tool and streaming SSE reasoning fix.

> **Upstream credit & license:** Based on [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) commit `2fe9be834a601c70ef1a597563026340f0e722fe` by David Dalcu, preserved under the original [MIT License](LICENSE) (see [NOTICE](NOTICE) and [README_UPSTREAM.md](README_UPSTREAM.md) for full upstream documentation). Grouped expert deduplication and shared Gumbel-Max speculative coupling ideas were adapted for Apple M1/M2 from [dangxingyu/megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo) (`engines/inkling-2-mini_m5max`). No model weights are distributed in this repository.

---

## Highlights (Inkling 2 Mini Uncensored on Apple M1 Max, 64 GB, `--kv-quant 4`, `--ctx-size 1048576`)

### 1. Latest Build: Grouped Expert Dedup + `SUB16`/`SUB8` NVFP4 + Shared Gumbel-Max Coupling + N-Gram Copy

Clean 4-arm bracketed A/B/A (`Prev M1-Opt A1` → `New Serial` → `New Default (DSpark + Gumbel + Copy)` → `Prev M1-Opt A2`) serving the Q4-attention / Q6-head uncensored pack (with BF16 vision enabled, `--temp 1.0 --top-p 1.0 --top-k 0`):

| Workload | Prev M1-Opt Avg (`A1`/`A2`) | New Serial (`INKLING_NO_DRAFT=1`) | **New Default (DSpark+Gumbel+Copy)** | Default vs Prev M1-Opt |
|---|---:|---:|---:|---:|
| **Short code (greedy, 256t)** | `41.1` (`46.4` / `35.8`) | `43.5 tok/s` (`1.06×`) | **`44.0 tok/s`** | **1.07×** |
| **Short code (`T=1.0`, 256t)** | `41.1` (`46.2` / `36.0`) | `43.0 tok/s` (`1.05×`) | **`41.7 tok/s`** | **1.01×** |
| **Long code (`T=1.0`, 512t)** | `40.2` (`43.3` / `37.2`) | `42.7 tok/s` (`1.06×`) | **`49.6 tok/s`** (`3.34` acc/rnd; `80–89 tok/s` code bursts) | **1.23×** |
| **Greek prose (`T=1.0`, 256t)** | `39.0` (`41.5` / `36.4`) | `42.1 tok/s` (`1.08×`) | **`44.2 tok/s`** | **1.13×** |
| **`xhigh` math (`T=1.0`, 384t)** | `38.5` (`41.3` / `35.6`) | `42.7 tok/s` (`1.11×`) | **`43.3 tok/s`** | **1.13×** |
| **Code edit / verbatim copy (512t)** | `37.0` (`40.6` / `33.5`) | `43.1 tok/s` (`1.16×`) | **`73.7 tok/s`** (`310/314` = 98.7% copy acc) | **1.99×** |
| **13.7K needle (`T=1.0`, Q4 KV)** | — | `33.3 tok/s` concurrent | **`34.3 tok/s`** (needle exact) | **+3.0% vs serial** |
| **36K–37K live agentic tool-call turns (`T=1.0`, SSD cache)** | — | `33.0–39.7 tok/s` concurrent | **`35.0–40.7 tok/s`** (`97–245 ms` SSD restore) | **0% to +13.6% vs serial** |

*(Also live-verified with BF16 vision `model-vision.safetensors` enabled, passing exact image OCR readout in both serial and default DSpark modes, and zero garbled tokens across all `T=1.0` Gumbel-coupled generations.)*

### 2. Baseline vs Stock `ddalcu/mlx-serve` (Previous M1-Opt Serial Pass)

Bracketed back-to-back head-to-head benchmark (`OPT_1` → `STOCK` → `OPT_2`) on the **Inkling 2 Mini Uncensored** (TokenAI-zer / OpenUncensored-derived mixed-precision) checkpoint, comparing stock `ddalcu/mlx-serve` (`2fe9be8` + SSE fix) serving the canonical 8-bit-attention / NVFP4-expert uncensored pack against the prior M1-opt serial build serving the Q4-attention / Q6-head uncensored pack:

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

All engine optimizations live cleanly inside `src/plugins/inkling2/`, plus minimal DSpark native-draft / Gumbel / copy hooks in `src/abi/plugin.zig`, `src/engines/mlx/plugin/driver.zig`, `src/engines/mlx/spec/dspark.zig`, and the streaming reasoning parser fix in `src/host/chat/reasoning/think_tags.zig`:

1. **Native FP16 activations & weights on Apple7/8 (M1/M2, `gen < 15`)** (`plugin.zig`, `layers.zig`, `ups.metal`, `downs.metal`):
   - M1/M2 GPUs lack hardware BF16 ALUs. Automatically runs Inkling-2-Mini in FP16 while applying NVFP4 per-expert `global_scale` in `float32` before narrowing to FP16 so intermediate expert activations never overflow (`INKLING_FORCE_BF16=1` disables).
2. **Fused Q4/Q8 KV Attention with Relative Bias, Sinks & Output Gate** (`qkv_ink2.metal`, `qkv_ink2_merge.metal`, `qkv_ink2_verify.metal`, `qkv_ink2_verify_merge.metal`):
   - Stock `inkling2` ignored quantized KV triples and forced full 48-layer KV dequantization every token (~5 ms penalty by 8K).
   - Reads packed Q4/Q8 KV cache directly with `simdgroup_float8x8` MMA, fuses per-head relative bias + sink + sigmoid output gate, tunes `nsplit = clamp((t_k + 255) / 256, 1, 16)`, and lets 4-bit `o_proj` run on MLX's ~228 GB/s `qmv_fast` instead of ~77 GB/s `oproj.metal`.
3. **265–273 GB/s M1 Max NVFP4 MoE Microkernels & Grouped 4-Pair Expert Deduplication** (`ups.metal`, `downs.metal`, `plan_experts.metal`, `ups_dedup.metal`, `downs_dedup.metal`, `sum.metal`, `router.metal`, `gate.metal`):
   - **`S=1` serial decode:** Upgraded `ups.metal` to 16-lane sub-warps (`SUB16`) with `uint2` 128B cache-line loads, hoisted `2^22`, `fma()`, and `ROWS=4` (**`5.89 → 4.20 ms` / 273 GB/s** across 48 visits on M1 Max, ~80% of physical 341 GB/s DRAM peak); hoisted `2^22` into per-expert `sc_cache` with inline `uint2`/`fma` in `downs.metal` (**`2.64 → 2.42 ms` / 237 GB/s**). *(Constant LUT decode from `megakernel-zoo` microbenched 33% slower on M1 Max than bit-placement + `fma`, so bit-placement was kept.)*
   - **`S=2..16` multi-row verify:** Single-TG GPU planner (`plan_experts.metal`) groups `(row, pick)` pairs into slots of up to 4 pairs sharing one expert weight read across `ups_dedup.metal` (`SUB16`, **265 GB/s**) and `downs_dedup.metal` (`SUB8`, **267 GB/s**) — cutting `S=8` MoE verify **2.18×** (`~69 → 31.6 ms` across 48 visits) and dropping short-context per-verify-row cost to **0.21×** a plain step (`MLX_SERVE_INKLING2_NO_DEDUP=1` disables).
   - Added register-rolling `ROLL=1` to `sum.metal` (reads `DN` 1× instead of 4×), and multi-row `SROWS ≤ 16` weight reuse to `router.metal` and `gate.metal`.
4. **Coalesced FP16 Loop Message Gates & Hoisted Constants** (`gate.metal`, `kernels.zig`, `plugin.zig`):
   - Converts `gates.{down,forget,message}` from F32 to FP16, transposes `forget`/`message` to `[4, RANK, D]` with threadgroup SRAM staging (`TG=128`), and hoists ~336 per-token constant array allocations onto `layers.Body`.
5. **Default-On M1/M2 DSpark + Exact Shared Gumbel-Max Coupling + Zero-Rollback N-Gram Copy** (`draft.zig`, `markov_header/fold/pick.metal`, `gumbel_picks.metal`, `plugin.zig`, `driver.zig`, `dspark.zig`):
   - **M1/M2 `verifyQmm` width cap (`max_steps=6`, `rows ≤ 7`):** Avoids M1 Max's **+16 ms** `simd_qmm` cliff at `rows=8` (`64.5 ms` at `rows=7` vs `81 ms` at `rows=8`) and cuts `draft()` latency to **~6.0 ms** (with 8-bit drafter layers and symmetric per-row INT8 `d.fold`).
   - **Exact Single-Sync Shared Gumbel-Max Coupling (`T > 0, top_p = 1.0, top_k = 0`):** Both `markov_fold.metal` and target `gumbel_picks.metal` share a deterministic per-position Gumbel(0, 1) noise hash (`msv_ink2_gumbel`), preserving the exact `softmax(logits / T)` distribution while eliminating `probsAllPositions` + 2nd GPU sync and boosting `T=1.0` code acceptance to **3.3–3.5 tokens/round** (vs ~1.2 with one-hot proposals). Uses exact 23-bit uniforms (`(float(x >> 9) + 0.5f) * 2^-23`, strictly `< 1.0f`) to avoid the 24-bit IEEE-754 round-to-`1.0f → +Inf` bug.
   - **Zero-Rollback N-Gram Copy Proposals (`dspark.zig`):** Proposes context-matching spans directly into the deduped verify path with `0 ms` draft forward (`6` drafts normally, up to `15` drafts on `≥16`-token matches at short context) — reaching **73.7 tok/s** (`98.7%` copy acceptance) on code edits while `dspark_pol` backs off cleanly on unpredictable prose (with JIT-spike filtering in `RowCost` and fast streak wakeup).
   - Enabled by default on M1/M2 (`gen >= 13`); pass `INKLING_NO_DRAFT=1` for pure optimized serial decode (~350 MB less RAM).
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

Serve with 1M context, Q4 KV caching, and an 8 GB SSD prefix cache (sized for 64 GB Macs running long multi-turn agentic sessions):

```bash
./zig-out/bin/mlx-serve \
  --model /path/to/inkling2-mlx-q4attn \
  --serve --host 127.0.0.1 --port 8085 \
  --ctx-size 1048576 --max-tokens 65536 \
  --os-reserve-gib 16 --max-concurrent 1 \
  --kv-quant 4 \
  --prefix-cache-entries 8 --no-prefix-cache-ram --prefix-cache-disk 8GB \
  --temp 1.0 --top-p 1.0 --top-k 0
```

*(Add `--no-vision` if you don't need multimodal image input; with `model-vision.safetensors` present, BF16 vision works out of the box alongside DSpark.)*

> **Prefix-cache sizing tip:** Under `--kv-quant 4`, a `50K–135K`-token session uses `~0.6–1.5 GB` of KV/checkpoint state, so `2GB` on disk can trigger LRU eviction races between rapid tool-call turns or multiple agent threads — `8GB` keeps 5–12 large sessions warm (`~0.10–0.25s` restore from SSD). Keep `--no-prefix-cache-ram` when using `--ctx-size 1048576` on 64 GB: `mlx-serve` reserves the full manual context at load time, clamping `--prefix-cache-mem` to `0 MB` while downgrading the SSD tier (`2048 -> 512 MB/flush`, `16 -> 8` SSM checkpoints/entry). Only remove `--no-prefix-cache-ram` if you also lower `--ctx-size` (e.g. `262144`) so RAM headroom remains. Note that `mlx-serve` reserves `min(64 GiB, 10% volume)` of free disk before arming the SSD tier.

### Runtime Environment Knobs

| Env Var | Effect |
|---|---|
| `INKLING_FORCE_BF16=1` | Disable M1/M2 native FP16 mode (fall back to BF16) |
| `INKLING_NO_FUSED_QKV=1` | Disable fused Q4/Q8 KV decode attention (`qkv_ink2`) |
| `INKLING_NO_FUSED_QKV_VERIFY=1` | Disable fused Q4/Q8 KV verify attention (`qkv_ink2_verify`) |
| `MLX_SERVE_INKLING2_NO_DEDUP=1` | Disable grouped 4-pair expert deduplication during multi-row verify |
| `INKLING_NO_DRAFT=1` | Force pure optimized serial decode (disables DSpark drafter, saves ~350 MB RAM) |
| `INKLING_NO_GUMBEL=1` | Disable Shared Gumbel-Max coupling on `T > 0` (fall back to one-hot Leviathan verify) |
| `INKLING_DRAFT_BITS=4\|6\|8\|16` | Drafter transformer layer quantization width on M1/M2 (default `8`) |
| `INKLING_DRAFT_MAX=1..7` | Cap drafted tokens per round on M1/M2 (default `6`, keeping verify `rows ≤ 7` in `verifyQmm`) |
| `INKLING_NO_FOLD_Q8=1` | Keep Markov `d.fold` in FP16 instead of per-row symmetric INT8 |
| `INKLING_PROFILE_DRAFT=1` | Log per-round `[ink2-prof]` draft/verify/commit timings |

### Benchmarking

```bash
python3 tools/bench_inkling2.py \
  --bin ./zig-out/bin/mlx-serve \
  --model /path/to/inkling2-mlx-q4attn
```
