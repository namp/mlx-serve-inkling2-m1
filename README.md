# mlx-serve-inkling2-m1

Apple M1 / M2 optimizations for **Inkling 2 Mini (`inkling2_loop`)** — developed and tested on an **Apple M1 Max (64 GB)** with **Inkling 2 Mini Uncensored** (and compatible with any Inkling 2 Mini MLX/canonical checkpoint) on top of [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) **`v26.11.4`** (`45df50e`, MIT License), plus a virtually lossless Q4-attention / Q6-head requantization tool, Lucebox Fast `o_proj` graft tool, and streaming SSE reasoning fix.

> **Upstream credit & license:** Rebased onto [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) release **`v26.11.4`** (`45df50e`), preserved under the original [MIT License](LICENSE) (see [NOTICE](NOTICE) and [README_UPSTREAM.md](README_UPSTREAM.md) for full upstream documentation). Grouped expert deduplication ideas were adapted for Apple M1/M2 from [dangxingyu/megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo) (`engines/inkling-2-mini_m5max`); native Inkling-2-Mini DFlash 2 contract details follow [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) and [ARahim3/mlx-dspark](https://github.com/ARahim3/mlx-dspark); the anti-overthinking `o_proj` graft tool extracts Apache-2.0 deltas from [Lucebox/Inkling-2-Mini-Fast-GGUF](https://huggingface.co/Lucebox/Inkling-2-Mini-Fast-GGUF); attention loop unrolling, native FP16 NVFP4 prefill GEMM design, and LoopCD contrastive decoding (`arXiv:2610.02185`) with thinking-block exclusion were inspired by [jundot/omlx](https://github.com/jundot/omlx) v0.8.4 (`#5623`, `#5632`, `#5617`, `#5640`). No model weights are distributed in this repository.

---

## Highlights (Inkling 2 Mini Uncensored on Apple M1 Max, 64 GB, `--kv-quant 4`, `--ctx-size 1048576`)

### 0. Rebase onto Upstream `v26.11.4` (`45df50e`) + M1/M2 Kernel Integration

Combines 100% of our M1/M2 Inkling 2 Mini stack with upstream **`mlx-serve v26.11.4`**:
- **Upstream `v26.11.4` wins preserved:** cached `HostGpu.launch` Metal dispatches, zero-copy donated `Spec` slack ring buffer, pre-evaluated `Body.consts`, `topk_group.metal` fused top-K + grouping (`-48` launches/verify round), `join_qkv` 6-projection fusion on sliding pass 0 (`-40` projections/step), `downs.metal` 1-row `NST` conv-state fusion (`-48` `state_kernel` launches/step) + `few_tail.metal` `STATE=1`, `relattn.metal` separate `WINDOW==0` plain loop + fused `GATED` output epilogue (`-48` `gated_kernel` launches in prefill), `oQ` pack support, Q8+SLC `markov_fold_q8` / `markov_pick_rnn`, sliding-window prefix-cache resume (`#1291`), and dynamic hot-cache memory-pressure clamping (`#1185`).
- **M1/M2 (`gen < 15`) protections added on `v26.11.4`:**
  - `relattn.metal` stays on `BK=16, BQ=32, TG=128` for M1/M2 (avoids upstream's `BK=32` threadgroup memory spill that regressed 8K prefill by `-11.3%` on M1 Max, while keeping upstream's `WINDOW==0` plain loop and fused `GATED` epilogue).
  - `groupTileWidth` stays on `BM=4` for M1/M2 (keeps our `SUB8=1` 8-lane NVFP4 `downs_grp` fast path active on `s=17..64` prompts instead of falling back to `BM=8, SUB8=0`).
  - `act.metal` / `sum.metal` `VEC4` gated on matching input/output dtypes so mixed-dtype BF16→FP16 fallback paths remain exact; `qkv_ink2` reuses 1-D `w.consts.rel_params` for `attn_scale`.
- **Thermally-bracketed M1 Max A/B** (`Prev 7f41957+M1 A1` → **`v26.11.4 + M1`** → `Prev 7f41957+M1 A2`, 25s cooldowns, Q4 KV + DFlash 2 `W=8`, `--prefix-cache-entries 0`, isolated `:18086`):

| Workload | Prev `7f41957+M1` Mean (`A1`/`A2`) | **`v26.11.4 + M1`** | Delta vs Mean |
|---|---:|---:|---:|
| **`code_t0` (`T=0`, 256t)** | `35.3 tok/s` (`41.7` / `28.9`) | **`39.8 tok/s`** | **+12.8%** |
| **`code_t1` (`T=1.0`, 256t)** | `31.9 tok/s` (`35.7` / `28.1`) | **`37.0 tok/s`** | **+16.0%** |
| **`math_t0` (`T=0`, 256t)** | `33.7 tok/s` (`37.6` / `29.8`) | **`36.3 tok/s`** | **+7.7%** |
| **`math_t1` (`T=1.0`, 256t)** | `34.4 tok/s` (`38.7` / `30.1`) | **`35.1 tok/s`** | **+2.1%** |
| **`edit_t0` (`T=0`, copy, 256t)** | `33.0 tok/s` (`37.2` / `28.7`) | **`38.7 tok/s`** (`34/34` copy accepts) | **+17.3%** |
| **`pp_2k` (`2,179` tok)** | `234.5 tok/s` (`262.5` / `206.5`) | **`238.4 tok/s`** | **+1.7%** |
| **`pp_8k` (`8,278` tok)** | `212.5 tok/s` (`239.5` / `185.6`) | **`214.6 tok/s`** | **+1.0%** |
| **Pure serial (`INKLING_NO_DRAFT=1 --no-pld`, cool)** | `54.8 tok/s` (`54.0` / `55.6`) | **`53.2 tok/s`** (`54.7` / `51.6`) | ~parity |

*(All `143` `inkling2` unit/proof/contract tests pass with `0` failures; `--loopcd-skip-thinking` live-verified on 54 visits.)*

### 1. Native FP16 NVFP4 Prefill GEMM (`S >= 64`) + Unrolled Attention + LoopCD (`--loopcd-skip-thinking`)

Building on top of the PR #1331 M1-Opt release:
- **Flash Attention unrolling + plain-tile fast path** (`relattn.metal`, `relsplit.metal`, `relsplit_lanes.metal`): **+28% to +52%** faster isolated M1 Max FP16 attention microbenchmarks.
- **Native FP16 NVFP4 prefill expert GEMM + fused soft-capped SwiGLU** (`nvfp4_plan.metal`, `nvfp4_prefill.metal`): eliminates M1 Max's BF16 `gather_qmm` fallback across all 20 NVFP4 expert layers (44/48 visits) on `S >= 64` prefill chunks with exact `e2m1 * fp8_e4m3` in normal `half`, swapped grid for L2 weight reuse, and fused SwiGLU (~56.5 MB/visit DRAM saved).
- **LoopCD contrastive decoding (`arXiv:2610.02185`) with Exclude Thinking Blocks** (`--loopcd` / `--loopcd-skip-thinking`, `loopcd_thinking.metal`, `loopcd_guide.metal`): saves pass-0 `(h, blocks)` after visit 18, runs 6 reference coda visits (54 total visits/caches), keeps thinking blocks 100% unguided while preserving control-token softmax probabilities (`>= 199998`), and works with DFlash 2 `W=8` tree verify, Q4 KV, and RAM/SSD prefix caches.
- **Thermally-bracketed M1 Max A/B** (`Prev PR1331 M1-Opt A1` → `New Default` → `New + --loopcd-skip-thinking` → `Prev PR1331 M1-Opt A2`, 25s cooldowns, Q4 KV + DFlash 2 `W=8` + `--warmup-full` + `--prefix-cache-entries 0`):

| Workload | Prev PR1331 M1-Opt Avg (`A1`/`A2`) | **New Default (Unroll + FP16 NVFP4 Prefill)** | Delta vs Mean (vs Cold `A1`) | New + `--loopcd-skip-thinking` (54 visits) |
|---|---:|---:|---:|---:|
| **`pp_2k` (`2,348` tok)** | `249.0 tok/s` (`273.1` / `224.9`) | **`338.6 tok/s`** (`6.94s`) | **+36.0%** (**+24.0%**) | `257.8 tok/s` (`9.11s`) |
| **`pp_8k` (`8,105` tok)** | `242.1 tok/s` (`259.1` / `225.2`) | **`308.6 tok/s`** (`26.26s`) | **+27.5%** (**+19.1%**) | **`260.8 tok/s`** (`31.08s`) |
| **Plain decode step** | `23.1–23.2 ms` | **`18.8 ms`** | **−18.8% latency** | `26.1 ms` (+6 coda visits) |
| **`code_t0` (DFlash `W=8`)** | `56.53 tok/s` (`53.52` / `59.54`) | `56.01 tok/s` | ~parity | `45.68 tok/s` (−18%) |
| **`reason_t0` (DFlash `W=8`)** | `71.50 tok/s` (`A1`) | `66.38 tok/s` | ~parity | `61.05 tok/s` (−8% vs New) |

*(Outputs on `pp_2k` and `pp_8k` matched baseline character-for-character. Under deterministic FP16-KV serial decode, `--loopcd-skip-thinking` produced a 100% character-identical thinking block (`232/232` chars) while cleanly guiding the final answer.)*

### 2. Rebase onto Upstream PR #1331 + M1 Max GPU DFlash 2 Best-First Tree (`W=8`)

All proven M1/M2 optimizations (native FP16 trunk, coalesced transposed message gates, `SUB16`/`SUB8` NVFP4 MoE kernels, fused Q4/Q8 KV decode + tree-aware `S=2..8` verify, 4-bit `L.o` bypass, hoisted constants) have been rebased onto upstream **PR #1331** (`mlx-serve 26.11.4`), tuned specifically for M1 Max:

- **Solo `W=8` GPU best-first tree on M1/M2 (`plugin.zig`):** `simd_qmm` covers `M=8` cleanly on M1 Max (`70.7 ms` vs `70.6 ms` at `W=7`), giving a free 8th tree node that boosts accepted tokens/round by **+12% to +26%**.
- **Greedy strong-copy bypass (`dspark.zig`):** Skips the 5-layer DFlash 2 forward on greedy n-gram copy matches and verifies up to 15 copied tokens in one pass (`avg=5.00`, 16-row copy rounds).
- **Tree-mode stochastic prior (`generate/init.zig`):** Uses a `0.80` initial prior (instead of the 1D chain `0.70`) when GPU best-first tree mode (`nativeStepGroupAvail`) is active so our ~20 ms M1 Max serial step doesn't lock `T=1.0` requests out of tree drafting on token 0.
- **Thermally-bracketed M1 Max A/B (`Prev M1-Opt DFlash2 A1` → `PR #1331 + M1-Opt` → `Prev M1-Opt DFlash2 A2`, 25s cooldown, Fast-Uncensored Q4-attn pack + BF16 vision):**

| Benchmark | Prev M1-Opt Avg (`A1`/`A2`) | **PR #1331 + M1-Opt (`W=8` Tree)** | Delta |
|---|---:|---:|---:|
| **Core-6 Effective tok/s** | `52.52 eff tok/s` (`54.57` / `50.47`) | **`54.39 eff tok/s`** | **+3.6%** |
| **`code_t0` (`T=0`)** | `64.2 tok/s` (`68.4` / `60.0`) | **`73.7 tok/s`** (`avg=5.00`) | **+14.8%** |
| **`code_t1` (`T=1.0`)** | `56.1 tok/s` (`59.6` / `52.6`) | **`57.4 tok/s`** (`avg=3.77`) | **+2.3%** |
| **`math_t0` (`T=0`)** | `58.5 tok/s` (`62.6` / `54.4`) | **`62.3 tok/s`** (`avg=4.01`) | **+6.5%** |
| **`math_t1` (`T=1.0`) / `edit_t0` (`T=0`)** | `55.5` / `65.9 tok/s` | `52.2` / `63.6 tok/s` (`avg=5.00`) | `-5.9%` / `-3.5%` |
| **`greek_t1` (`T=1.0`)** | `43.5 tok/s` (`45.4` / `41.5`) | `38.0 tok/s` (`avg=1.71`) | `-12.6%` |
| **BF16 Vision (`7391`, steady wall)** | `2.36s` (`2.33s` / `2.39s`) | **`1.53s`** (`1.23s` live on `:8085`) | **-35.2%** |
| **8K Q4 KV (`8,059` pt, `T=1.0`)** | `31.80s` wall (`pp 319.9`, `dec 52.4`) | **`28.53s` wall** (`pp 324.5`, `dec 51.4`) | **-10.3% wall** |

### 3. Earlier M1-Opt Native DFlash 2 + Lucebox Fast `o_proj` Graft Baseline

Stacking native **DFlash 2** parallel block-diffusion drafting (`src/plugins/inkling2/dflash2.zig`) with Lucebox's 24-layer `o_proj` anti-overthinking graft (`tools/graft_lucebox_fast.py`) delivers **~2.3–2.5× faster end-to-end wall-clock answers** on M1 Max while preserving uncensored behavior, 1M Q4 KV context, 8 GB SSD prefix cache, and BF16 vision:

- **DFlash 2 vs. bundled `mtp.safetensors` (`dspark`) on Fast-Uncensored Q4-attn pack (M1 Max, `T=0` & `T=1.0` Shared Gumbel-Max):**

| Benchmark | Bundled `mtp.safetensors` | **Native `DFlash 2` (8-bit BF16)** | Delta |
|---|---:|---:|---:|
| **6-prompt suite total (cold start)** | `41.4 eff tok/s` (`90.3s`) | **`49.0 eff tok/s` (`70.3s`)** | **+18.4%** |
| **`code_t1` (`T=1.0`) / `edit_t0` (`T=0`)** | `48.2` / `52.2 tok/s` | **`60.5` / `61.5 tok/s`** | **+25.5% / +17.8%** |
| **`math_t1` (`T=1.0`) / `greek_t1` (`T=1.0`)** | `40.7` / `36.1 tok/s` | **`53.1` / `42.8 tok/s`** (`1.72` vs `0.88` Greek acc/rnd) | **+30.5% / +18.6%** |
| **Thermally-bracketed A/B/C/A (4 prompts, heat-soaked)** | `34.1 eff tok/s` (`0.980×` serial) | **`39.7 eff tok/s` (`1.114×` serial)** | **+16.4% (+13.6% norm.)** |
| **35.6K live Codex turn (`T=1.0`, SSD cache)** | — | **`37.9 tok/s`** (`264 ms` SSD restore of 35.6K tokens) | `47` reasoning + `17` reply tok |

- **Lucebox Fast `o_proj` graft (`tools/graft_lucebox_fast.py`) on Inkling-2-Mini-Uncensored Q4-attn (95.6 MB disk via hardlinked shards):**
  - **8-prompt suite (math, code, diagnosis, Greek, SQLi):** `12,257` → **`5,783` tokens (`-52.8%`)**, `<think>` chars **`-65.6%`**, exact BF16 vision (`7391`) preserved.
  - **6-prompt cold-start & abliteration suite:** `3,862` → **`1,926` tokens (`-50.1%`)**, **`86.1s → 46.8s` (`1.84×` faster; `3.0×` on `T=1.0` Python code)**, zero refusal regressions (`max |cos(ΔW_fast, ΔW_unc)| = 0.00512` across all 24 layers; `99.65%` directional survival under 4-bit `gs=64`).

### 4. Grouped Expert Dedup + `SUB16`/`SUB8` NVFP4 + Shared Gumbel-Max Coupling + N-Gram Copy

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

### 5. Baseline vs Stock `ddalcu/mlx-serve` (Previous M1-Opt Serial Pass)

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
6. **Native Metal DFlash 2 Drafter (`incoai/Inkling-2-Mini-DFlash2`)** (`src/plugins/inkling2/dflash2.zig`, `plugin.zig`, `draft.zig`, `config.zig`):
   - Wires `incoai/Inkling-2-Mini-DFlash2` directly into `inkling2`'s zero-rollback `kernels.Open` / `kernels.commit` verify loop (reusing the exact `[5, 23, 41, 48]` taps, `dflash_dyn_conv.metal`, and `dflash_topk.metal`).
   - Keeps the **48-visit trunk in native FP16** on M1/M2 while running the **5-layer DFlash 2 drafter in 8-bit affine BF16** (preventing FP16 dynamic-range overflow on Inkling's un-normed taps and dynamic conv kernels, casting only at the 8-row boundary).
   - Implements `selector_softcap_mode: "after_correction"` (`30.0`), full `embed_norm ∘ embed_tokens` for anchor & `mask_token_id` (`200056`), valid vocab truncation (`200,058`), mask-free `keep = sliding_window - block_size = 2040` incremental context cache, and exact **23-bit Shared Gumbel-Max coupling** at `T > 0`.
   - Auto-loads from `<model_dir>/dflash2` (symlink `incoai/Inkling-2-Mini-DFlash2` there) or `INKLING_DFLASH2_DIR`; falls back cleanly to `mtp.safetensors` when absent or with `INKLING_NO_DFLASH2=1`.
7. **Inkling SSE `<|content_thinking|>` Separation Fix** (`src/host/chat/reasoning/think_tags.zig`):
   - Latches `<|content_thinking|>` on model-opened reasoning blocks and guards split boundaries on `<|end_message|>` so streamed thinking stays strictly inside `reasoning_content` without leaking into visible `content`.
8. **Flash Attention Unrolling & Plain-Tile Fast Path** (`relattn.metal`, `relsplit.metal`, `relsplit_lanes.metal`):
   - Adds `#pragma clang loop unroll(full)` to all fragment/load/store loops (eliminating register spills on Apple7/8 GPUs) and a mask/bias-free `plain` interior-tile branch for full-attention prefill & split-K tiles (+28% to +52% faster isolated FP16 attention on M1 Max).
9. **Native FP16 NVFP4 Prefill Expert GEMM + Fused Soft-Capped SwiGLU (`S >= 64`)** (`nvfp4_plan.metal`, `nvfp4_prefill.metal`, `kernels.zig`):
   - Replaces M1/M2's BF16 `gather_qmm` fallback during prompt prefill across all 20 NVFP4 expert layers (44/48 visits) with a zero-host-sync GPU planner and `BM=16, BN=64, BK=32` FP16 MMA kernel using exact `e2m1 * fp8_e4m3` dequantization in normal `half`, swapped grid for L2 weight reuse, FP32 accumulation + inline tensor scale, and fused soft-capped SwiGLU (`MLX_SERVE_INKLING2_PREFILL_FP4=0` disables).
10. **LoopCD Contrastive Decoding (`arXiv:2610.02185`) + Exclude Thinking Blocks** (`--loopcd`, `--loopcd-skip-thinking`, `loopcd_thinking.metal`, `loopcd_guide.metal`, `config.zig`, `layers.zig`, `plugin.zig`):
    - Captures pass-0 `(h, blocks)` after visit 18, runs 6 reference coda visits (54 total visits/caches), and fuses adaptive guidance (`weight = 0.5 * (1 - (p_top1 - p_top2))`) on GPU.
    - `--loopcd-skip-thinking` tracks per-slot thinking state in layer 0's `.ssm` slot (prefix-cache and batch safe, including `TREE=1` DFlash 2 `W=8` scans), leaves thinking blocks 100% unguided, and applies a `logsumexp` shift to ordinary tokens (`< 199998`) so control-token softmax probabilities remain exact.

---

## Zero-Copy Lucebox Fast `o_proj` Graft (`tools/graft_lucebox_fast.py`)

`Lucebox/Inkling-2-Mini-Fast-GGUF` changes **only** the 24 BF16 `self_attn.o_proj` matrices (`0.0` diff on all other weights) and is >99.5% orthogonal to uncensored abliteration deltas (`max |cos| = 0.00512`). Using `mlx-serve`'s carrier-of-record index loader (`src/sdk/weights/load.zig`), `tools/graft_lucebox_fast.py` downloads only those 24 tensors (~678 MB via HTTP Range requests), merges `W_src + ΔW_fast`, writes a single ~95 MB `model-fast-oproj.safetensors` shard, hardlinks all untouched shards/vision/MTP files (0 GB duplicated), and stamps `_fast_oproj_graft` in `config.json` to isolate the SSD prefix cache:

```bash
# 1. Extract the 24 BF16 o_proj deltas (~678 MB download, ~340 MB saved)
python3 tools/graft_lucebox_fast.py extract \
  --out ./lucebox_fast_oproj_deltas.safetensors

# 2. Build a zero-copy hardlinked pack (~95 MB extra disk)
python3 tools/graft_lucebox_fast.py apply \
  --deltas ./lucebox_fast_oproj_deltas.safetensors \
  --pack /path/to/inkling2-mlx-q4attn \
  --ref-q8 /path/to/inkling2-mlx-8bit \
  --out /path/to/inkling2-mlx-fast-q4attn \
  --bits 4 --alpha 1.0

# 3. Optional: enable DFlash 2 by symlinking incoai/Inkling-2-Mini-DFlash2
ln -s /path/to/Inkling-2-Mini-DFlash2 /path/to/inkling2-mlx-fast-q4attn/dflash2
```

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
  --temp 1.0 --top-p 1.0 --top-k 0 --warmup-full
```

*(Pass `--warmup-full` so PR #1331's load-time `[rounds]` ladder completes before `:8085` opens and arms DFlash 2 GPU best-first tree verification from prompt #1; add `--loopcd-skip-thinking` for LoopCD contrastive decoding on final answers with unguided thinking blocks, or `--loopcd` for full LoopCD; add `--no-vision` if you don't need multimodal image input.)*

> **Prefix-cache sizing tip:** Under `--kv-quant 4`, a `50K–135K`-token session uses `~0.6–1.5 GB` of KV/checkpoint state, so `2GB` on disk can trigger LRU eviction races between rapid tool-call turns or multiple agent threads — `8GB` keeps 5–12 large sessions warm (`~0.10–0.25s` restore from SSD). Keep `--no-prefix-cache-ram` when using `--ctx-size 1048576` on 64 GB: `mlx-serve` reserves the full manual context at load time, clamping `--prefix-cache-mem` to `0 MB` while downgrading the SSD tier (`2048 -> 512 MB/flush`, `16 -> 8` SSM checkpoints/entry). Only remove `--no-prefix-cache-ram` if you also lower `--ctx-size` (e.g. `262144`) so RAM headroom remains. Note that `mlx-serve` reserves `min(64 GiB, 10% volume)` of free disk before arming the SSD tier.

### Runtime Environment Knobs

| Env Var | Effect |
|---|---|
| `INKLING_FORCE_BF16=1` | Disable M1/M2 native FP16 trunk mode (fall back to BF16) |
| `INKLING_NO_FUSED_QKV=1` | Disable fused Q4/Q8 KV decode attention (`qkv_ink2`) |
| `INKLING_NO_FUSED_QKV_VERIFY=1` | Disable fused Q4/Q8 KV verify attention (`qkv_ink2_verify`) |
| `INKLING_TREE_WIDTH=2..16` | Override solo GPU best-first draft tree width (default `8` on M1/M2) |
| `MLX_SERVE_INKLING2_TREE=0` | Disable GPU best-first draft tree (`stepGroup`) and fall back to chain verify |
| `INKLING_DRAFT_MAX=1..15` | Cap chain verify drafts per round on M1/M2 (default `7`, `rows ≤ 8`) |
| `INKLING_NO_DFLASH2=1` | Disable DFlash 2 (`drafter/` and `dflash2/`) and fall back to bundled `mtp.safetensors` |
| `INKLING_NO_DRAFT=1` | Force pure optimized serial decode (disables both DFlash 2 and bundled drafters) |
| `MLX_SERVE_INKLING2_PREFILL_FP4=0` | Disable M1/M2 native FP16 NVFP4 prefill expert GEMM (`S >= 64`) |
| `MLX_SERVE_INKLING2_LOOPCD=1` | Enable LoopCD contrastive decoding (same as `--loopcd`) |
| `MLX_SERVE_INKLING2_LOOPCD_SKIP_THINKING=1` | Enable LoopCD while skipping guidance inside thinking blocks (same as `--loopcd-skip-thinking`) |

### Benchmarking

```bash
python3 tools/bench_inkling2.py \
  --bin ./zig-out/bin/mlx-serve \
  --model /path/to/inkling2-mlx-q4attn
```
