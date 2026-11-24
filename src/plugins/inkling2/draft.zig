//! Inkling 2's bundled block drafter (`mtp.safetensors`, DSpark), as oMLX's `omlx/patches/inkling2/dspark.py` runs
//! it in its DFlash drafter (Apache-2.0; no other open reference ships it): a few pre-norm layers over a block of one
//! anchor (the committed token's normed embedding) and `block - 1` copies of a trained mask vector. Each layer mixes
//! the trunk's taps (the stream entering a few visits and the final-normed output) with its own softmax weights,
//! norms the mix and projects it to context keys and values; the block attends that context (the newest `window - 1`
//! committed positions) and itself in both directions, under a rotary over absolute positions. The mask rows go
//! through the trunk's LM head; an RNN Markov head adds each row a bias from the token drafted before it, so the
//! block chains left to right off one forward. Drafts only: the trunk verifies every one.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const reader = @import("../../sdk/bases/reader.zig");
const head_block = @import("../../sdk/bases/blocks/head.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const config = @import("config.zig");
const kernels = @import("kernels.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const gpa = std.heap.c_allocator;
const project = head_block.project;

pub const Layer = struct {
    in_norm: Array,
    post_norm: Array,
    /// The tap mix's norm (`dflare_norm`), f32.
    mix_norm: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    k_ctx: Linear,
    v_ctx: Linear,
    q_norm: Array,
    k_norm: Array,
    k_norm_ctx: Array,
    gate: Linear,
    up: Linear,
    down: Linear,
};

/// The drafter's layers a context cache holds.
pub const max_layers = 4;

/// The drafter's context keys and values a layer (`[1, kv_heads, rows, hd]`, normed and rotated at their positions)
/// for the positions before `end`, kept across one slot's rounds: a round projects only the rows committed since. It
/// is valid while the taps ring it was projected from continues, and is released when the ring restarts.
pub const ContextKv = struct {
    k: [max_layers]Array = @splat(.{}),
    v: [max_layers]Array = @splat(.{}),
    end: u64 = 0,

    pub fn release(c: *ContextKv) void {
        for (c.k ++ c.v) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        c.* = .{};
    }
};

pub const Drafter = struct {
    cfg: config.Draft,
    hidden: u32,
    s: mlx.mlx_stream,
    owned: std.ArrayList(Array) = .empty,
    layers: []Layer = &.{},
    norm: Array = .{},
    /// `[1, 1, hidden]`.
    mask: Array = .{},
    /// Each layer's softmax weights over the taps, then the Markov head's: `[layers + 1, taps, 1]` f32.
    alpha: Array = .{},
    /// The Markov head: the previous token's row `[vocab, rank]`, the RNN's projection of [state | that row | the
    /// block row | the newest context mix] to its gate, candidate and latent, and the latent's bias over the vocab
    /// (the trunk's head over `hidden_out`, folded once: `fold` `[rank, vocab]`, `fold_bias` `[1, 1, vocab]`).
    prev_rows: Array = .{},
    rnn: Linear = .{},
    rnn_bias: Array = .{},
    rnn_norm: Array = .{},
    fold: Array = .{},
    /// Per-row f32 scales `[rank]` when `fold` is symmetric INT8; null when `fold` is half.
    fold_scale: ?Array = null,
    fold_bias: Array = .{},
    /// Cap on drafted tokens per round (0 = full cfg.block - 1).
    max_steps: u32 = 0,

    pub fn load(ctx: *const abi.LoadCtx, cfg: config.Draft, hidden: u32, head: *const Linear, s: mlx.mlx_stream) !?*Drafter {
        return loadOpt(ctx, cfg, hidden, head, s, false);
    }

    /// The drafter beside the pack, its Markov head folded over the trunk's `head`; null when the pack ships none.
    /// When `use_f16` (M1/M2), casts weights to FP16 and optionally quantizes the 3 drafter transformer layers' dense
    /// projections to 4-bit (or 8-bit via INKLING_DRAFT_BITS) affine for 3.5x less drafter memory traffic.
    pub fn loadOpt(ctx: *const abi.LoadCtx, cfg: config.Draft, hidden: u32, head: *const Linear, s: mlx.mlx_stream, use_f16: bool) !?*Drafter {
        if (!ctx.dir.exists(ctx.dir.ctx, abi.Str.of(config.draft_file))) return null;
        var w = try reader.headWeights(ctx, &.{config.draft_file}, "fuse_weight", "\x00");
        defer w.deinit();
        if (w.get("fuse_weight") == null) return null;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .cfg = cfg, .hidden = hidden, .s = s };
        errdefer d.release();
        var g = Graph.init(gpa, s);
        defer g.deinit();
        const qbits: u32 = blk: {
            if (!use_f16) break :blk 16;
            if (std.c.getenv("INKLING_DRAFT_BITS")) |raw| {
                const sp = std.mem.span(raw);
                if (std.mem.eql(u8, sp, "4")) break :blk 4;
                if (std.mem.eql(u8, sp, "6")) break :blk 6;
                if (std.mem.eql(u8, sp, "8")) break :blk 8;
                if (std.mem.eql(u8, sp, "16")) break :blk 16;
            }
            break :blk 8;
        };
        if (use_f16) {
            if (std.c.getenv("INKLING_DRAFT_MAX")) |raw| {
                const v = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch @min(@as(u32, 6), cfg.block - 1);
                d.max_steps = @min(v, cfg.block - 1);
            } else {
                // On M1/M2 (no NAX tile), verifyQmm serves up to 7 rows cleanly while 8 rows hits the +16 ms simd_qmm cliff;
                // capping at 6 drafts (7 verify rows) also saves 1 Markov step per round.
                d.max_steps = @min(@as(u32, 6), cfg.block - 1);
            }
            @import("../../sdk/util/log.zig").info("[inkling2-draft] M1/M2 FP16 drafter loaded: layer_bits={d} max_steps={d} fold_q8={}\n", .{ qbits, d.max_steps, std.c.getenv("INKLING_NO_FOLD_Q8") == null });
        }
        var at: At = .{ .d = d, .w = &w, .g = &g, .use_f16 = use_f16, .layer_bits = qbits };

        if (cfg.layers > max_layers) return error.Inkling2DraftGeometry;
        d.layers = try gpa.alloc(Layer, cfg.layers);
        for (d.layers, 0..) |*L, i| {
            at.layer = @intCast(i);
            L.* = .{
                .in_norm = try at.need("input_layernorm.weight"),
                .post_norm = try at.need("post_attention_layernorm.weight"),
                .mix_norm = try at.float("dflare_norm.weight"),
                .q = try at.linear("self_attn.q_proj"),
                .k = try at.linear("self_attn.k_proj"),
                .v = try at.linear("self_attn.v_proj"),
                .o = try at.linear("self_attn.o_proj"),
                .k_ctx = try at.linear("self_attn.k_proj_ctx"),
                .v_ctx = try at.linear("self_attn.v_proj_ctx"),
                .q_norm = try at.need("self_attn.q_norm.weight"),
                .k_norm = try at.need("self_attn.k_norm.weight"),
                .k_norm_ctx = try at.need("self_attn.k_norm_ctx.weight"),
                .gate = try at.linear("mlp.gate_proj"),
                .up = try at.linear("mlp.up_proj"),
                .down = try at.linear("mlp.down_proj"),
            };
        }
        at.layer = null;
        d.norm = try at.need("norm.weight");
        d.mask = try at.keep(try g.reshape(try at.raw("mask_embed"), &.{ 1, 1, @intCast(hidden) }));
        const fuse = try at.raw("fuse_weight");
        const fsh = ops.shape(fuse);
        if (fsh.len != 2 or fsh[0] != @as(c_int, @intCast(cfg.layers + 1)) or fsh[1] != @as(c_int, @intCast(cfg.tap_count))) return error.Inkling2DraftGeometry;
        // As the reference: the weights rounded to the stream's dtype before they scale the f32 taps.
        d.alpha = try at.keep(try g.reshape(try g.astype(try g.astype(try g.softmax(try g.astype(fuse, .float32), -1, true), ops.dtype(d.norm)), .float32), &.{ fsh[0], fsh[1], 1 }));
        d.prev_rows = try at.need("markov_head.markov_w1.weight");
        d.rnn = try at.linear("markov_head.rnn_proj");
        d.rnn_bias = try at.need("markov_head.rnn_proj.bias");
        d.rnn_norm = try at.float("markov_head.dflare_norm.weight");
        const rank: c_int = @intCast(cfg.rank);
        if (ops.shape(d.prev_rows)[1] != rank or ops.shape(d.rnn.w)[0] != 3 * rank) return error.Inkling2DraftGeometry;
        // `head(hidden_out(latent))` = latent over `head(hidden_out's columns)` plus `head(hidden_out's bias)`.
        var out_w = try at.raw("markov_head.hidden_out.weight");
        var out_b = try at.raw("markov_head.hidden_out.bias");
        if (use_f16) {
            out_w = try g.astype(out_w, .float16);
            out_b = try g.astype(out_b, .float16);
        }
        const fold_raw = try g.reshape(try project(&g, head, try g.reshape(try g.transpose(out_w, &.{ 1, 0 }), &.{ 1, rank, @intCast(hidden) })), &.{ rank, -1 });
        if (use_f16 and std.c.getenv("INKLING_NO_FOLD_Q8") == null) {
            const f32_fold = try g.astype(fold_raw, .float32);
            const amax = try g.maximum(try g.max(try g.abs(f32_fold), 1, true), try g.scalar(1e-8, .float32));
            const scale = try g.mulScalar(amax, 1.0 / 127.0);
            const scaled = try g.div(f32_fold, scale);
            const rounded = try g.call(mlx.mlx_round, .{ scaled, @as(c_int, 0) });
            const clipped = try g.call(mlx.mlx_clip, .{ rounded, try g.scalar(-127.0, .float32), try g.scalar(127.0, .float32) });
            d.fold = try at.keep(try g.astype(clipped, .int8));
            d.fold_scale = try at.keep(try g.reshape(scale, &.{rank}));
        } else {
            d.fold = try at.keep(fold_raw);
        }
        d.fold_bias = try at.keep(try project(&g, head, try g.reshape(out_b, &.{ 1, 1, @intCast(hidden) })));
        return d;
    }

    fn release(d: *Drafter) void {
        for (d.owned.items) |a| _ = mlx.mlx_array_free(a);
        d.owned.deinit(gpa);
        gpa.free(d.layers);
    }

    pub fn deinit(d: *Drafter) void {
        d.release();
        gpa.destroy(d);
    }

    /// The taps `[1, n, taps * hidden]` mixed with weight row `row` and normed: `[1, n, hidden]` in the taps' dtype.
    fn mix(d: *const Drafter, g: *Graph, taps32: Array, row: c_int, weight: Array) !Array {
        const mixed = try g.sum(try g.mul(taps32, try g.sliceAxis(d.alpha, 0, row, row + 1)), 2, false);
        return g.astype(try g.rmsNorm(mixed, weight, d.cfg.eps), ops.dtype(d.norm));
    }

    /// `x` `[1, n, heads * hd]` as heads `[1, heads, n, hd]`, normed per head (null = not) and rotated from position
    /// `at` (null = not).
    fn heads(d: *const Drafter, g: *Graph, x: Array, count: u32, norm_weight: ?Array, at: ?c_int) !Array {
        const n = ops.shape(x)[1];
        var h = try g.reshape(x, &.{ 1, n, @intCast(count), @intCast(d.cfg.head_dim) });
        if (norm_weight) |w| h = try g.rmsNorm(h, w, d.cfg.eps);
        h = try g.transpose(h, &.{ 0, 2, 1, 3 });
        return if (at) |p| g.rope(h, @intCast(d.cfg.head_dim), d.cfg.theta, p) else h;
    }

    /// A drafted block: its tokens `[block - 1]` int32 and the drafter's probability of each `[block - 1]` f32, lazy.
    pub const Block = struct { tokens: Array, confidence: Array };

    /// The `block - 1` tokens drafted after the committed token `t1` (`[1]` int32) at position `position`. `anchor` `[1, 1, hidden]` is `t1`'s normed embedding, `taps` `[1, n, taps * hidden]` the trunk's
    /// taps at the `n >= 1` positions before `position`, `head` the trunk's LM head, `vocab` its real rows and `softcap` its
/// logit soft cap. `cache` carries the context's keys and values over the slot's rounds (null: projected afresh).
    pub fn draft(d: *const Drafter, g: *Graph, t1: Array, anchor: Array, taps: Array, position: c_int, head: *const Linear, vocab: c_int, softcap: f32, cache: ?*ContextKv) !Block {
        return d.draftWithGumbel(g, t1, anchor, taps, position, head, vocab, softcap, cache, null);
    }

    pub fn draftWithGumbel(d: *const Drafter, g: *Graph, t1: Array, anchor: Array, taps: Array, position: c_int, head: *const Linear, vocab: c_int, softcap: f32, cache: ?*ContextKv, gumbel: ?kernels.GumbelSpec) !Block {
        const cfg = &d.cfg;
        const hidden: c_int = @intCast(d.hidden);
        const block: c_int = @intCast(cfg.block);
        const rank: c_int = @intCast(cfg.rank);
        const n = ops.shape(taps)[1];
        const dt = ops.dtype(d.norm);
        const at: u64 = @intCast(position);
        // The rows the cache lacks: those committed since its end, when it still reaches back to the ring's first.
        var fresh: c_int = n;
        if (cache) |cc| if (cc.k[0].ctx != null and cc.end <= at and at - cc.end <= @as(u64, @intCast(n))) {
            const since: c_int = @intCast(at - cc.end);
            if (since + ops.shape(cc.k[0])[2] >= n) fresh = since;
        };
        const tapsOf = struct {
            fn f(gr: *Graph, x: Array, from: c_int, to: c_int, count: u32, width: c_int) !Array {
                return gr.astype(try gr.reshape(try gr.sliceAxis(x, 1, from, to), &.{ 1, to - from, @intCast(count), width }), .float32);
            }
        }.f;
        const new32 = if (fresh > 0) try tapsOf(g, taps, n - fresh, n, cfg.tap_count, hidden) else null;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const back = [_]c_int{ 0, 2, 1, 3 };

        var h = try g.concat(&.{ try g.astype(anchor, dt), try g.broadcast(try g.astype(d.mask, dt), &.{ 1, block - 1, hidden }) }, 1);
        for (d.layers, 0..) |*L, i| {
            var lg = g.scope();
            defer lg.deinit();
            var kc: Array = .{};
            var vc: Array = .{};
            if (new32) |rows32| {
                const context = try d.mix(&lg, rows32, @intCast(i), L.mix_norm);
                kc = try d.heads(&lg, try project(&lg, &L.k_ctx, context), cfg.kv_heads, L.k_norm_ctx, position - fresh);
                vc = try d.heads(&lg, try project(&lg, &L.v_ctx, context), cfg.kv_heads, null, null);
            }
            if (fresh < n) {
                const cc = cache.?;
                if (fresh > 0) {
                    kc = try lg.concat(&.{ cc.k[i], kc }, 2);
                    vc = try lg.concat(&.{ cc.v[i], vc }, 2);
                } else {
                    kc = cc.k[i];
                    vc = cc.v[i];
                }
                const total = ops.shape(kc)[2];
                kc = try lg.sliceAxis(kc, 2, total - n, total);
                vc = try lg.sliceAxis(vc, 2, total - n, total);
            }
            if (cache) |cc| {
                const k_held = lg.escape(kc);
                const v_held = lg.escape(vc);
                if (cc.k[i].ctx != null) _ = mlx.mlx_array_free(cc.k[i]);
                if (cc.v[i].ctx != null) _ = mlx.mlx_array_free(cc.v[i]);
                cc.k[i] = k_held;
                cc.v[i] = v_held;
            }
            const x = try lg.rmsNorm(h, L.in_norm, cfg.eps);
            const q = try d.heads(&lg, try project(&lg, &L.q, x), cfg.heads, L.q_norm, position);
            const keys = try lg.concat(&.{ kc, try d.heads(&lg, try project(&lg, &L.k, x), cfg.kv_heads, L.k_norm, position) }, 2);
            const values = try lg.concat(&.{ vc, try d.heads(&lg, try project(&lg, &L.v, x), cfg.kv_heads, null, null) }, 2);
            const read = try lg.reshape(try lg.transpose(try lg.sdpa(q, keys, values, scale, .none), &back), &.{ 1, block, -1 });
            const mid = try lg.add(h, try project(&lg, &L.o, read));
            const y = try lg.rmsNorm(mid, L.post_norm, cfg.eps);
            const mlp = try project(&lg, &L.down, try lg.mul(try lg.silu(try project(&lg, &L.gate, y)), try project(&lg, &L.up, y)));
            h = try g.keep(try lg.add(mid, mlp));
        }
        const want_steps: c_int = if (d.max_steps > 0 and d.max_steps < cfg.block - 1) @intCast(d.max_steps) else block - 1;
        const rows = try g.sliceAxis(try g.rmsNorm(h, d.norm, cfg.eps), 1, 1, 1 + want_steps);
        const base = try g.astype(try project(g, head, rows), .float32);

        if (cache) |cc| cc.end = at;
        // The RNN restarts each block from the newest committed position's mix.
        const newest = try d.mix(g, try tapsOf(g, taps, n - 1, n, cfg.tap_count, hidden), @intCast(d.layers.len), d.rnn_norm);
        return d.markovWithGumbel(g, t1, rows, newest, base, vocab, softcap, kernels.markovRuns(g, d.rnn.w, d.fold, rank), gumbel);
    }

    /// The Markov head's chain over the block rows `rows` `[1, block - 1, hidden]` and their head logits `base` f32,
    /// from the committed token `t1` and the newest mix `newest` `[1, 1, hidden]`: each step's token and probability.
    /// `fused` runs a step as `kernels.markovStep`'s three launches, else as the op chain.
    pub fn markov(d: *const Drafter, g: *Graph, t1: Array, rows: Array, newest: Array, base: Array, vocab: c_int, softcap: f32, fused: bool) !Block {
        return d.markovWithGumbel(g, t1, rows, newest, base, vocab, softcap, fused, null);
    }

    pub fn markovWithGumbel(d: *const Drafter, g: *Graph, t1: Array, rows: Array, newest: Array, base: Array, vocab: c_int, softcap: f32, fused: bool, gumbel: ?kernels.GumbelSpec) !Block {
        const rank: c_int = @intCast(d.cfg.rank);
        const steps: usize = @intCast(ops.shape(rows)[1]);
        const dt = ops.dtype(d.norm);
        var state = try g.zeros(&.{ 1, 1, rank }, dt);
        var prev = t1;
        var drafted: [abi.max_round_tokens]Array = undefined;
        var sure: [abi.max_round_tokens]Array = undefined;
        const dense_rows = try g.contiguous(rows);
        const fold_dense = if (!fused and d.fold_scale != null) try g.astype(try g.mul(try g.astype(d.fold, .float32), try g.reshape(d.fold_scale.?, &.{ rank, 1 })), dt) else d.fold;
        for (0..steps) |i| {
            const at: c_int = @intCast(i);
            if (fused) {
                const step_gumbel: ?kernels.GumbelSpec = if (gumbel) |gs| .{ .inv_t = gs.inv_t, .seed = gs.seed, .q0 = gs.q0 + @as(u32, @intCast(i)) } else null;
                const st = try kernels.markovStepGumbel(g, d.rnn.w, d.rnn_bias, state, d.prev_rows, dense_rows, newest, prev, d.fold, d.fold_scale, d.fold_bias, base, at, vocab, softcap, step_gumbel);
                state = st.state;
                prev = st.token;
                drafted[i] = st.token;
                sure[i] = st.confidence;
                continue;
            }
            const last = try g.astype(try g.reshape(try g.take(d.prev_rows, prev, 0), &.{ 1, 1, rank }), dt);
            const fed = try g.concat(&.{ state, last, try g.sliceAxis(rows, 1, at, at + 1), newest }, 2);
            const p = try g.astype(try g.add(try project(g, &d.rnn, fed), d.rnn_bias), .float32);
            const keep = try g.astype(try g.sigmoid(try g.sliceAxis(p, 2, 0, rank)), dt);
            const blend = try g.mul(try g.sub(try g.scalar(1, dt), keep), try g.astype(try g.tanh(try g.sliceAxis(p, 2, rank, 2 * rank)), dt));
            state = try g.add(try g.mul(keep, state), blend);
            const latent = try g.astype(try g.tanh(try g.sliceAxis(p, 2, 2 * rank, 3 * rank)), dt);
            const bias = try g.astype(try g.add(try g.matmul(latent, fold_dense), d.fold_bias), .float32);
            // The soft cap is monotonic: the pick over the real rows is the capped logits'; the probability is not.
            const logits = try g.sliceAxis(try g.add(try g.sliceAxis(base, 1, at, at + 1), bias), 2, 0, vocab);
            prev = try g.astype(try g.reshape(try g.argmax(logits, -1, false), &.{1}), .int32);
            drafted[i] = prev;
            const capped = if (softcap > 0) try g.mulScalar(try g.tanh(try g.mulScalar(logits, 1.0 / softcap)), softcap) else logits;
            sure[i] = try g.reshape(try g.exp(try g.sub(try g.max(capped, -1, false), try g.call(mlx.mlx_logsumexp_axis, .{ capped, -1, false }))), &.{1});
        }
        return .{ .tokens = try g.concat(drafted[0..steps], 0), .confidence = try g.concat(sure[0..steps], 0) };
    }
};

/// Reads the sidecar's tensors (a layer's under `layers.N.`), owned by the drafter.
const At = struct {
    d: *Drafter,
    w: *const Weights,
    g: *Graph,
    layer: ?u32 = null,
    use_f16: bool = false,
    layer_bits: u32 = 16,

    /// A new handle the graph owns.
    fn raw(a: At, leaf: []const u8) !Array {
        var buf: [160]u8 = undefined;
        const name = if (a.layer) |i| try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, leaf }) else leaf;
        const x = a.w.get(name) orelse return error.MissingWeight;
        return a.g.keep(x);
    }

    fn keep(a: At, x: Array) !Array {
        try ops.eval(x);
        const out = a.g.escape(x);
        try a.d.owned.append(gpa, out);
        return out;
    }

    fn need(a: At, leaf: []const u8) !Array {
        var x = try a.raw(leaf);
        if (a.use_f16 and ops.dtype(x) == .bfloat16) x = try a.g.astype(x, .float16);
        return a.keep(x);
    }

    fn float(a: At, leaf: []const u8) !Array {
        return a.keep(try a.g.astype(try a.raw(leaf), .float32));
    }

    /// A projection in the checkpoint's layout, `[out, in]`. For drafter transformer layers on M1/M2,
    /// optionally quantizes to affine FP16 (`layer_bits` in {4, 6, 8}) to slash drafter bandwidth.
    fn linear(a: At, base: []const u8) !Linear {
        var buf: [160]u8 = undefined;
        const wname = try std.fmt.bufPrint(&buf, "{s}.weight", .{base});
        if (a.layer != null and a.layer_bits < 16) {
            var wf = try a.raw(wname);
            if (a.use_f16 and ops.dtype(wf) == .bfloat16) wf = try a.g.astype(wf, .float16);
            var q = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(q);
            try mlx.check(mlx.mlx_quantize(&q, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(a.layer_bits)), "affine", .{ .ctx = null }, a.g.s));
            var parts: [3]Array = undefined;
            for (&parts, 0..) |*p, i| {
                var t = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&t, q, i));
                p.* = try a.keep(try a.g.adopt(t));
            }
            return .{ .w = parts[0], .s = parts[1], .b = parts[2], .bits = a.layer_bits, .group_size = 64, .mode = .affine };
        }
        return .{ .w = try a.need(wname) };
    }
};
