//! DFlash 2 parallel block-diffusion drafter for Inkling 2 (`incoai/Inkling-2-Mini-DFlash2`),
//! wired directly into the inkling2 plugin's zero-rollback open/commit verify loop.
//!
//! Protocol (matching `incoai/Inkling-2-Mini-DFlash2` SGLang reference & `ARahim3/mlx-dspark`):
//!   - Context taps: `[5, 23, 41, 48]` concatenated `[1, n, 4 * hidden]` -> `fc` -> `hidden_norm`
//!     -> per-layer `k_proj` (+ `k_norm` + RoPE) / `v_proj`, incrementally cached in `draft.ContextKv`
//!     up to `keep = sliding_window - block_size` (2040), so the 8-row block always fits inside the
//!     2048 sliding window with ZERO array mask.
//!   - Block input `[1, 8, hidden]`: `[anchor, mask_embed x 7]`, where BOTH anchor and mask_token_id
//!     (200056) pass through the trunk's full `embed_norm(embed_tokens(id))`.
//!   - 5 Qwen3 layers with bidirectional block attention and 2-tap grouped dynamic causal conv
//!     (`k=2, g=16`) wrapping attention and MLP (`kernels/spec/dflash_dyn_conv.metal`).
//!   - Post-norm draft rows `1..1+m` -> trunk `lm_head` sliced to valid vocab (`200058`) -> GPU
//!     `topKRows` (`kernels/spec/dflash_topk.metal`, `k=16`) -> bilinear `candidate_selector`
//!     (`rank=256`, `after_correction` softcap 30.0) -> exact greedy or 23-bit Shared Gumbel-Max walk.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const head_block = @import("../../sdk/bases/blocks/head.zig");
const embed_block = @import("../../sdk/bases/blocks/embed.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const load_weights = @import("../../sdk/weights/load.zig");
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const log = @import("../../sdk/util/log.zig");
const config = @import("config.zig");
const draft = @import("draft.zig");
const kernels = @import("kernels.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const gpa = std.heap.c_allocator;
const project = head_block.project;

pub const DFlash2Config = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    window: u32,
    block: u32,
    mask_token_id: u32,
    conv_kernel_size: u32,
    conv_group_size: u32,
    selector_rank: u32,
    selector_top_k: u32,
    eps: f32 = 1e-6,
    theta: f32 = 10000.0,
    softcap: f32 = 30.0,
    softcap_after: bool = true,
    taps: [config.max_taps]u32 = @splat(0),
    tap_count: u32 = 0,
};

pub const DynConv = struct {
    /// `[ksize, hidden]` (side 0 = prepare, side 1 = finish), contiguous in drafter dtype.
    base0: Array,
    base1: Array,
    /// `[2 * ksize * groups, hidden]`
    proj: Linear,
};

pub const Layer = struct {
    in_norm: Array,
    post_norm: Array,
    attn_conv: DynConv,
    mlp_conv: DynConv,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Array,
    k_norm: Array,
    gate: Linear,
    up: Linear,
    down: Linear,
};

pub const DraftResult = struct {
    tokens: [abi.max_round_tokens]i32 = @splat(0),
    confidence: [abi.max_round_tokens]f32 = @splat(0),
    len: u32 = 0,
};

const DYN_CONV_SOURCE = kernelSource("kernels/spec/dflash_dyn_conv.metal");
var dyn_conv_kernel: ?mlx.mlx_fast_metal_kernel = null;
const DynConvKey = struct { l: c_int, h: c_int, ks: c_int, gs: c_int, dt: mlx.mlx_dtype };
var dyn_conv_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var dyn_conv_key: ?DynConvKey = null;

const TOPK_SOURCE = kernelSource("kernels/spec/dflash_topk.metal");
var topk_kernel: ?mlx.mlx_fast_metal_kernel = null;
const TopKKey = struct { m: c_int, v: c_int, k: c_int, dt: mlx.mlx_dtype };
var topk_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var topk_key: ?TopKKey = null;

fn jsonField(o: std.json.ObjectMap, k: []const u8) ?std.json.Value {
    return o.get(k);
}

fn jsonU32(v: ?std.json.Value) ?u32 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |n| if (n >= 0 and n <= std.math.maxInt(u32)) @intCast(n) else null,
        else => null,
    };
}

fn jsonF32(v: ?std.json.Value) ?f32 {
    const x = v orelse return null;
    return switch (x) {
        .float => |f| @floatCast(f),
        .integer => |n| @floatFromInt(n),
        else => null,
    };
}

pub fn parseConfig(bytes: []const u8, trunk_hidden: u32, trunk_visits: u32) !DFlash2Config {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{}) catch return error.InvalidDflash2Config;
    if (doc != .object) return error.InvalidDflash2Config;
    const root = doc.object;
    const dfv = jsonField(root, "dflash_config") orelse return error.InvalidDflash2Config;
    if (dfv != .object) return error.InvalidDflash2Config;
    const df = dfv.object;

    var theta: f32 = 10000.0;
    if (jsonField(root, "rope_parameters")) |rp| {
        if (rp == .object) if (jsonF32(jsonField(rp.object, "rope_theta"))) |t| {
            theta = t;
        };
    } else if (jsonF32(jsonField(root, "rope_theta"))) |t| {
        theta = t;
    }

    var softcap_after = false;
    if (jsonField(df, "selector_softcap_mode")) |mv| {
        if (mv == .string) {
            if (std.mem.eql(u8, mv.string, "after_correction")) {
                softcap_after = true;
            } else if (!std.mem.eql(u8, mv.string, "before_correction")) {
                return error.InvalidDflash2Config;
            }
        }
    }

    var c: DFlash2Config = .{
        .hidden = jsonU32(jsonField(root, "hidden_size")) orelse return error.InvalidDflash2Config,
        .layers = jsonU32(jsonField(root, "num_hidden_layers")) orelse return error.InvalidDflash2Config,
        .heads = jsonU32(jsonField(root, "num_attention_heads")) orelse return error.InvalidDflash2Config,
        .kv_heads = jsonU32(jsonField(root, "num_key_value_heads")) orelse return error.InvalidDflash2Config,
        .head_dim = jsonU32(jsonField(root, "head_dim")) orelse return error.InvalidDflash2Config,
        .intermediate = jsonU32(jsonField(root, "intermediate_size")) orelse return error.InvalidDflash2Config,
        .window = jsonU32(jsonField(root, "sliding_window")) orelse 2048,
        .block = jsonU32(jsonField(df, "block_size")) orelse return error.InvalidDflash2Config,
        .mask_token_id = jsonU32(jsonField(df, "mask_token_id")) orelse return error.InvalidDflash2Config,
        .conv_kernel_size = jsonU32(jsonField(df, "conv_kernel_size")) orelse return error.InvalidDflash2Config,
        .conv_group_size = jsonU32(jsonField(df, "conv_group_size")) orelse return error.InvalidDflash2Config,
        .selector_rank = jsonU32(jsonField(df, "selector_rank")) orelse return error.InvalidDflash2Config,
        .selector_top_k = jsonU32(jsonField(df, "selector_top_k")) orelse return error.InvalidDflash2Config,
        .eps = jsonF32(jsonField(root, "rms_norm_eps")) orelse 1e-6,
        .theta = theta,
        .softcap = jsonF32(jsonField(df, "final_logit_softcapping")) orelse jsonF32(jsonField(root, "final_logit_softcapping")) orelse 0.0,
        .softcap_after = softcap_after,
    };

    if (c.hidden != trunk_hidden) return error.InvalidDflash2Config;
    const ids = jsonField(df, "target_layer_ids") orelse return error.InvalidDflash2Config;
    if (ids != .array or ids.array.items.len < 2 or ids.array.items.len > config.max_taps) return error.InvalidDflash2Config;
    for (ids.array.items, 0..) |item, i| {
        c.taps[i] = jsonU32(item) orelse return error.InvalidDflash2Config;
        const last = i + 1 == ids.array.items.len;
        if (if (last) c.taps[i] != trunk_visits else c.taps[i] >= trunk_visits) return error.InvalidDflash2Config;
    }
    c.tap_count = @intCast(ids.array.items.len);

    if (c.layers == 0 or c.layers > draft.max_layers or c.heads == 0 or c.kv_heads == 0 or
        c.heads % c.kv_heads != 0 or c.head_dim == 0 or c.head_dim % 2 != 0 or
        c.block < 2 or c.block > abi.max_round_tokens or c.window <= c.block or
        c.conv_kernel_size == 0 or c.conv_group_size == 0 or c.hidden % c.conv_group_size != 0 or
        c.selector_rank == 0 or c.selector_top_k == 0 or c.selector_top_k > 32)
        return error.InvalidDflash2Config;
    return c;
}

/// Resolve optional DFlash2 checkpoint directory:
/// 1. `INKLING_DFLASH2_DIR` env var if set;
/// 2. `<model_dir>/dflash2` if `config.json` and `model.safetensors` exist there.
pub fn resolveDir(allocator: std.mem.Allocator, model_dir: []const u8) !?[]u8 {
    if (std.c.getenv("INKLING_NO_DFLASH2") != null) return null;
    const io = std.Io.Threaded.global_single_threaded.io();
    const candidates: [2]?[]const u8 = .{
        if (std.c.getenv("INKLING_DFLASH2_DIR")) |p| std.mem.span(p) else null,
        if (model_dir.len > 0) try std.fmt.allocPrint(allocator, "{s}/dflash2", .{model_dir}) else null,
    };
    defer if (candidates[1]) |p| allocator.free(p);
    for (candidates) |maybe_dir| {
        const dir_path = maybe_dir orelse continue;
        if (dir_path.len == 0) continue;
        const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir_path});
        defer allocator.free(cfg_path);
        const st_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{dir_path});
        defer allocator.free(st_path);
        const f1 = std.Io.Dir.openFileAbsolute(io, cfg_path, .{}) catch continue;
        f1.close(io);
        const f2 = std.Io.Dir.openFileAbsolute(io, st_path, .{}) catch continue;
        f2.close(io);
        return try allocator.dupe(u8, dir_path);
    }
    return null;
}

/// Exact 23-bit deterministic Gumbel(0, 1) noise matching `kernels/markov_header.metal::msv_ink2_gumbel`.
pub fn gumbel23(seed: u32, q0: u32, v: u32) f32 {
    const key: u32 = ((seed ^ (q0 *% 0x85EBCA6B)) *% 0xC2B2AE35) +% 0x27D4EB2F;
    var x: u32 = (v *% 0x9E3779B1) +% key;
    x ^= x >> 16;
    x *%= 0x85EBCA6B;
    x ^= x >> 13;
    x *%= 0xC2B2AE35;
    x ^= x >> 16;
    const u: f32 = (@as(f32, @floatFromInt(x >> 9)) + 0.5) * 1.1920928955078125e-07;
    return -@log(-@log(u));
}

pub fn groupedDynConv(g: *Graph, hidden: Array, dynamic: Array, base: Array, group_size: u32) !Array {
    if (try dynConvFused(g, hidden, dynamic, base, group_size)) |y| return y;
    return groupedDynConvOps(g, hidden, dynamic, base, group_size);
}

fn dynConvFused(g: *Graph, hidden: Array, dynamic: Array, base: Array, group_size: u32) !?Array {
    if (!mlx.streamIsGpu(g.s)) return null;
    const dt = ops.dtype(hidden);
    if (ops.dtype(dynamic) != dt or ops.dtype(base) != dt) return null;
    const hsh = ops.shape(hidden);
    const bsh = ops.shape(base);
    const dsh = ops.shape(dynamic);
    if (hsh.len != 3 or hsh[0] != 1 or bsh.len != 2 or dsh.len != 4) return null;
    const gs: c_int = @intCast(group_size);
    const key = DynConvKey{ .l = hsh[1], .h = hsh[2], .ks = bsh[0], .gs = gs, .dt = dt };
    if (@rem(key.h, gs) != 0 or bsh[1] != key.h or dsh[0] != 1 or dsh[1] != key.l or dsh[2] != key.ks or dsh[3] != @divExact(key.h, gs)) return null;
    if (dyn_conv_kernel == null) {
        const ins = [_][*:0]const u8{ "x", "dyn", "base" };
        const outs = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("msv_ink2_dflash_dyn_conv", in_vec, out_vec, DYN_CONV_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        dyn_conv_kernel = k;
    }
    if (dyn_conv_key == null or !std.meta.eql(dyn_conv_key.?, key)) {
        if (dyn_conv_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        dyn_conv_cfg = null;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, key.l, key.h }, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, key.l * key.h, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        inline for (.{ .{ "L", key.l }, .{ "H", key.h }, .{ "KS", key.ks }, .{ "GS", key.gs } }) |kv|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
        dyn_conv_cfg = cfg;
        dyn_conv_key = key;
    }
    const ins = [_]Array{ hidden, dynamic, base };
    const vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, dyn_conv_kernel.?, vec, dyn_conv_cfg.?, g.s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return try g.adopt(y);
}

pub fn groupedDynConvOps(g: *Graph, hidden: Array, dynamic: Array, base: Array, group_size: u32) !Array {
    const hsh = ops.shape(hidden);
    const len = hsh[1];
    const h = hsh[2];
    const gs: c_int = @intCast(group_size);
    const groups = @divExact(h, gs);
    const ksize: usize = @intCast(ops.shape(base)[0]);
    const dt = ops.dtype(hidden);
    const blocks = try g.reshape(hidden, &.{ 1, len, groups, gs });
    var out = try g.zeros(&.{ 1, len, groups, gs }, dt);
    for (0..ksize) |tap| {
        const values = if (tap == 0)
            blocks
        else blk: {
            const pad = try g.zeros(&.{ 1, @intCast(tap), groups, gs }, dt);
            const head = try g.sliceAxis(blocks, 1, 0, len - @as(c_int, @intCast(tap)));
            break :blk try g.concat(&.{ pad, head }, 1);
        };
        const kernel = try g.reshape(try g.sliceAxis(base, 0, @intCast(tap), @as(c_int, @intCast(tap)) + 1), &.{ 1, 1, groups, gs });
        const dslice = try g.reshape(try g.sliceAxis(dynamic, 2, @intCast(tap), @as(c_int, @intCast(tap)) + 1), &.{ 1, len, groups, 1 });
        out = try g.add(out, try g.mul(kernel, values));
        out = try g.add(out, try g.mul(dslice, values));
    }
    return g.reshape(out, &.{ 1, len, h });
}

fn topKRows(g: *Graph, logits: Array, k: usize) ![2]Array {
    const sh = ops.shape(logits);
    const dt = ops.dtype(logits);
    if (mlx.streamIsGpu(g.s) and k > 0 and k <= 32 and sh.len == 3 and sh[0] == 1 and
        sh[2] >= 256 * @as(c_int, @intCast(k)) and (dt == .bfloat16 or dt == .float16 or dt == .float32))
    {
        const key = TopKKey{ .m = sh[1], .v = sh[2], .k = @intCast(k), .dt = dt };
        if (topk_kernel == null) {
            const ins = [_][*:0]const u8{"logits"};
            const outs = [_][*:0]const u8{ "idx", "val" };
            const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
            defer _ = mlx.mlx_vector_string_free(in_vec);
            const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
            defer _ = mlx.mlx_vector_string_free(out_vec);
            const kern = mlx.mlx_fast_metal_kernel_new("msv_ink2_dflash_topk_rows", in_vec, out_vec, TOPK_SOURCE, "", true, false);
            if (kern.ctx == null) return error.MetalKernelCompileFailed;
            topk_kernel = kern;
        }
        if (topk_key == null or !std.meta.eql(topk_key.?, key)) {
            if (topk_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
            topk_cfg = null;
            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, key.m, key.k }, 3, .int32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, key.m, key.k }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256, key.m, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", key.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
            topk_cfg = cfg;
            topk_key = key;
        }
        const ins = [_]Array{logits};
        const vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, topk_kernel.?, vec, topk_cfg.?, g.s));
        var idx = mlx.mlx_array_new();
        var val = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&idx, outs, 0));
        errdefer _ = mlx.mlx_array_free(idx);
        try mlx.check(mlx.mlx_vector_array_get(&val, outs, 1));
        return .{ try g.adopt(idx), try g.adopt(val) };
    }
    // Fallback for small test vocabularies or CPU stream.
    const vocab = sh[2];
    const kk: c_int = @intCast(k);
    var part = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_argpartition_axis(&part, logits, vocab - kk, 2, g.s));
    const part_g = try g.adopt(part);
    const cands_i32 = try g.astype(try g.sliceAxis(part_g, 2, vocab - kk, vocab), .int32);
    var unary = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_along_axis(&unary, logits, cands_i32, 2, g.s));
    return .{ cands_i32, try g.astype(try g.adopt(unary), .float32) };
}

pub const Drafter = struct {
    cfg: DFlash2Config,
    s: mlx.mlx_stream,
    owned: std.ArrayList(Array) = .empty,
    fc: Linear = .{},
    hidden_norm: Array = .{},
    layers: []Layer = &.{},
    norm: Array = .{},
    /// `[1, 1, hidden]` precomputed `embed_norm(embed_tokens[mask_token_id])`.
    mask: Array = .{},
    sel_hidden: Linear = .{},
    pred_cb: Array = .{},
    succ_cb: Array = .{},
    max_steps: u32 = 0,

    pub fn loadOpt(
        model_dir: []const u8,
        trunk_hidden: u32,
        trunk_visits: u32,
        trunk_eps: f32,
        embed: *const embed_block.TableEmbed.Weights,
        embed_norm: Array,
        s: mlx.mlx_stream,
        use_f16: bool,
    ) !?*Drafter {
        const dir_path = (try resolveDir(gpa, model_dir)) orelse return null;
        defer gpa.free(dir_path);

        const io = std.Io.Threaded.global_single_threaded.io();
        const cfg_path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir_path});
        defer gpa.free(cfg_path);
        const cfg_bytes = std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20)) catch return null;
        defer gpa.free(cfg_bytes);
        const cfg = parseConfig(cfg_bytes, trunk_hidden, trunk_visits) catch |e| {
            log.warn("[inkling2-dflash2] skipping {s}: config parse failed ({s})\n", .{ dir_path, @errorName(e) });
            return null;
        };

        const st_path = try std.fmt.allocPrint(gpa, "{s}/model.safetensors", .{dir_path});
        defer gpa.free(st_path);
        var w = load_weights.loadWeightsSingleFile(gpa, st_path) catch |e| {
            log.warn("[inkling2-dflash2] skipping {s}: safetensors load failed ({s})\n", .{ dir_path, @errorName(e) });
            return null;
        };
        defer w.deinit();

        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .cfg = cfg, .s = s };
        errdefer d.release();

        var g = Graph.init(gpa, s);
        defer g.deinit();

        const qbits: u32 = blk: {
            if (std.c.getenv("INKLING_DFLASH2_BITS")) |raw| {
                const sp = std.mem.span(raw);
                if (std.mem.eql(u8, sp, "4")) break :blk 4;
                if (std.mem.eql(u8, sp, "6")) break :blk 6;
                if (std.mem.eql(u8, sp, "8")) break :blk 8;
                if (std.mem.eql(u8, sp, "16")) break :blk 16;
            }
            // Keep DFlash2 activations & scales in native BF16 (Inkling taps + base_kernel exceed FP16 range);
            // 8-bit affine BF16 cuts drafter memory traffic ~2x with zero dynamic-range overflow.
            break :blk if (use_f16) 8 else 16;
        };
        if (std.c.getenv("INKLING_DRAFT_MAX")) |raw| {
            const v = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch @min(@as(u32, 6), cfg.block - 1);
            d.max_steps = @min(v, cfg.block - 1);
        } else if (use_f16) {
            // M1/M2 verifyQmm serves up to 7 rows (1 anchor + 6 drafts) without hitting the +16ms simd_qmm cliff.
            d.max_steps = @min(@as(u32, 6), cfg.block - 1);
        } else {
            d.max_steps = cfg.block - 1;
        }

        // Always keep DFlash2 weights & activations in BF16 (never FP16, which overflows Inkling's tap/conv range).
        var at: At = .{ .d = d, .w = &w, .g = &g, .use_f16 = false, .layer_bits = qbits };
        d.fc = try at.linear("fc", true);
        d.hidden_norm = try at.need("hidden_norm.weight");
        d.norm = try at.need("norm.weight");
        d.sel_hidden = try at.linear("candidate_selector.hidden_projection", false);
        d.pred_cb = try at.need("candidate_selector.predecessor_codebook");
        d.succ_cb = try at.need("candidate_selector.successor_codebook");

        // Precompute normed mask embedding: embed_norm(embed_tokens[mask_token_id]).
        const mid = try g.fromSlice(i32, &.{@intCast(cfg.mask_token_id)}, &.{ 1, 1 });
        const raw_mask = try embedRow(&g, embed, mid);
        const normed_mask = try g.rmsNorm(raw_mask, embed_norm, trunk_eps);
        d.mask = try at.keep(try g.astype(normed_mask, ops.dtype(d.norm)));

        d.layers = try gpa.alloc(Layer, cfg.layers);
        for (d.layers, 0..) |*L, i| {
            at.layer = @intCast(i);
            L.* = .{
                .in_norm = try at.need("input_layernorm.weight"),
                .post_norm = try at.need("post_attention_layernorm.weight"),
                .attn_conv = try at.dynConv("attention_conv"),
                .mlp_conv = try at.dynConv("mlp_conv"),
                .q = try at.linear("self_attn.q_proj", true),
                .k = try at.linear("self_attn.k_proj", true),
                .v = try at.linear("self_attn.v_proj", true),
                .o = try at.linear("self_attn.o_proj", true),
                .q_norm = try at.need("self_attn.q_norm.weight"),
                .k_norm = try at.need("self_attn.k_norm.weight"),
                .gate = try at.linear("mlp.gate_proj", true),
                .up = try at.linear("mlp.up_proj", true),
                .down = try at.linear("mlp.down_proj", true),
            };
        }
        log.info("[inkling2-dflash2] loaded {s}: layers={d} block={d} max_steps={d} window={d} layer_bits={d} selector=rank{d}/top{d} softcap={d:.1}({s})\n", .{
            dir_path, cfg.layers, cfg.block, d.max_steps, cfg.window, qbits, cfg.selector_rank, cfg.selector_top_k, cfg.softcap,
            if (cfg.softcap_after) "after" else "before",
        });
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

    fn heads(d: *const Drafter, g: *Graph, x: Array, count: u32, norm_weight: ?Array, at: ?c_int) !Array {
        const n = ops.shape(x)[1];
        var h = try g.reshape(x, &.{ 1, n, @intCast(count), @intCast(d.cfg.head_dim) });
        if (norm_weight) |w| h = try g.rmsNorm(h, w, d.cfg.eps);
        h = try g.transpose(h, &.{ 0, 2, 1, 3 });
        return if (at) |p| g.rope(h, @intCast(d.cfg.head_dim), d.cfg.theta, p) else h;
    }

    fn convPrepare(d: *const Drafter, g: *Graph, conv: *const DynConv, normed: Array) !struct { hidden: Array, finish_dyn: Array } {
        const len = ops.shape(normed)[1];
        const ks: c_int = @intCast(d.cfg.conv_kernel_size);
        const groups: c_int = @intCast(d.cfg.hidden / d.cfg.conv_group_size);
        const dyn = try g.reshape(try project(g, &conv.proj, normed), &.{ 1, len, 2, ks, groups });
        const prep_dyn = try g.contiguous(try g.reshape(try g.sliceAxis(dyn, 2, 0, 1), &.{ 1, len, ks, groups }));
        const finish_dyn = try g.contiguous(try g.reshape(try g.sliceAxis(dyn, 2, 1, 2), &.{ 1, len, ks, groups }));
        const hidden_out = try groupedDynConv(g, normed, prep_dyn, conv.base0, d.cfg.conv_group_size);
        return .{ .hidden = hidden_out, .finish_dyn = finish_dyn };
    }

    /// Draft up to `max_steps` tokens after committed anchor token `t1` (`[1]` int32, value `anchor_id`)
    /// at `position`, conditioned on trunk taps `[1, n, 4 * hidden]` and incremental `cache`.
    pub fn draftWithGumbel(
        d: *const Drafter,
        g: *Graph,
        t1: Array,
        anchor: Array,
        taps: Array,
        position: c_int,
        head: *const Linear,
        vocab: c_int,
        cache: ?*draft.ContextKv,
        gumbel: ?kernels.GumbelSpec,
    ) !DraftResult {
        const cfg = &d.cfg;
        const hidden: c_int = @intCast(cfg.hidden);
        const block: c_int = @intCast(cfg.block);
        const n = ops.shape(taps)[1];
        const dt = ops.dtype(d.norm);
        const at: u64 = @intCast(position);

        var fresh: c_int = n;
        if (cache) |cc| if (cc.k[0].ctx != null and cc.end <= at and at - cc.end <= @as(u64, @intCast(n))) {
            const since: c_int = @intCast(at - cc.end);
            if (since + ops.shape(cc.k[0])[2] >= n) fresh = since;
        };

        const enc: ?Array = if (fresh > 0) blk: {
            const slice = try g.astype(try g.sliceAxis(taps, 1, n - fresh, n), dt);
            break :blk try g.rmsNorm(try project(g, &d.fc, slice), d.hidden_norm, cfg.eps);
        } else null;

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const back = [_]c_int{ 0, 2, 1, 3 };
        var h = try g.concat(&.{ try g.astype(anchor, dt), try g.broadcast(d.mask, &.{ 1, block - 1, hidden }) }, 1);

        for (d.layers, 0..) |*L, i| {
            var lg = g.scope();
            defer lg.deinit();

            var kc: Array = .{};
            var vc: Array = .{};
            if (enc) |e_rows| {
                kc = try d.heads(&lg, try project(&lg, &L.k, e_rows), cfg.kv_heads, L.k_norm, position - fresh);
                vc = try d.heads(&lg, try project(&lg, &L.v, e_rows), cfg.kv_heads, null, null);
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
                if (total > n) {
                    kc = try lg.sliceAxis(kc, 2, total - n, total);
                    vc = try lg.sliceAxis(vc, 2, total - n, total);
                }
            }
            if (cache) |cc| {
                const k_held = lg.escape(kc);
                const v_held = lg.escape(vc);
                if (cc.k[i].ctx != null) _ = mlx.mlx_array_free(cc.k[i]);
                if (cc.v[i].ctx != null) _ = mlx.mlx_array_free(cc.v[i]);
                cc.k[i] = k_held;
                cc.v[i] = v_held;
            }

            const normed = try lg.rmsNorm(h, L.in_norm, cfg.eps);
            const attn_prep = try d.convPrepare(&lg, &L.attn_conv, normed);
            const q = try d.heads(&lg, try project(&lg, &L.q, attn_prep.hidden), cfg.heads, L.q_norm, position);
            const bk = try d.heads(&lg, try project(&lg, &L.k, attn_prep.hidden), cfg.kv_heads, L.k_norm, position);
            const bv = try d.heads(&lg, try project(&lg, &L.v, attn_prep.hidden), cfg.kv_heads, null, null);
            const keys = try lg.concat(&.{ kc, bk }, 2);
            const values = try lg.concat(&.{ vc, bv }, 2);
            // `n <= window - block` (enforced by Spec.keep), so every block row attends all keys with zero mask.
            const read = try lg.reshape(try lg.transpose(try lg.sdpa(q, keys, values, scale, .none), &back), &.{ 1, block, -1 });
            const o_out = try project(&lg, &L.o, read);
            const attn_fin = try groupedDynConv(&lg, o_out, attn_prep.finish_dyn, L.attn_conv.base1, cfg.conv_group_size);
            const mid = try lg.add(h, attn_fin);

            const ff_normed = try lg.rmsNorm(mid, L.post_norm, cfg.eps);
            const mlp_prep = try d.convPrepare(&lg, &L.mlp_conv, ff_normed);
            const mlp_out = try project(&lg, &L.down, try lg.mul(try lg.silu(try project(&lg, &L.gate, mlp_prep.hidden)), try project(&lg, &L.up, mlp_prep.hidden)));
            const mlp_fin = try groupedDynConv(&lg, mlp_out, mlp_prep.finish_dyn, L.mlp_conv.base1, cfg.conv_group_size);
            h = try g.keep(try lg.add(mid, mlp_fin));
        }

        if (cache) |cc| cc.end = at;

        const want_steps: u32 = if (d.max_steps > 0 and d.max_steps < cfg.block - 1) d.max_steps else cfg.block - 1;
        const m: usize = want_steps;
        const k: usize = @min(@as(usize, cfg.selector_top_k), @as(usize, @intCast(vocab)));
        const rank: c_int = @intCast(cfg.selector_rank);

        const rows = try g.sliceAxis(try g.rmsNorm(h, d.norm, cfg.eps), 1, 1, 1 + @as(c_int, @intCast(m)));
        // Match trunk lm_head scale dtype (FP16 on M1/M2) so head projection stays on its fast native lane.
        const head_dt = if (head.s.ctx != null) ops.dtype(head.s) else ops.dtype(head.w);
        const rows_head = if (ops.dtype(rows) != head_dt and (head_dt == .float16 or head_dt == .bfloat16))
            try g.astype(rows, head_dt)
        else
            rows;
        const raw_full = try project(g, head, rows_head);
        const raw_v = try g.contiguous(try g.sliceAxis(raw_full, 2, 0, vocab));
        const top = try topKRows(g, raw_v, k);
        const cands_i32 = top[0];
        const unary_f32 = top[1];

        const hp = try project(g, &d.sel_hidden, rows); // [1, m, rank]
        const cands_flat = try g.reshape(cands_i32, &.{@intCast(m * k)});
        const succ_rows = try g.reshape(try g.take(d.succ_cb, cands_flat, 0), &.{ @intCast(m), @intCast(k), rank });
        const pred_rows = try g.reshape(try g.take(d.pred_cb, cands_flat, 0), &.{ @intCast(m), @intCast(k), rank });

        // e0 [1, k]: anchor_row * hp[0] @ succ[0]^T
        const a_row = try g.take(d.pred_cb, t1, 0); // [1, rank]
        const h0 = try g.reshape(try g.sliceAxis(hp, 1, 0, 1), &.{ 1, rank });
        const s0 = try g.reshape(try g.sliceAxis(succ_rows, 0, 0, 1), &.{ @intCast(k), rank });
        const e0_f32 = try g.astype(try g.matmul(try g.mul(a_row, h0), try g.transpose(s0, &.{ 1, 0 })), .float32);

        if (m > 1) {
            const pred_head = try g.sliceAxis(pred_rows, 0, 0, @intCast(m - 1));
            const h_tail = try g.reshape(try g.sliceAxis(hp, 1, 1, @intCast(m)), &.{ @intCast(m - 1), 1, rank });
            const succ_tail_t = try g.transpose(try g.sliceAxis(succ_rows, 0, 1, @intCast(m)), &.{ 0, 2, 1 });
            const e_f32 = try g.astype(try g.matmul(try g.mul(pred_head, h_tail), succ_tail_t), .float32);
            try ops.evalMany(&.{ cands_i32, unary_f32, e0_f32, e_f32 });
            const cd = mlx.mlx_array_data_int32(cands_i32) orelse return error.Inkling2DraftRound;
            const ud = mlx.mlx_array_data_float32(unary_f32) orelse return error.Inkling2DraftRound;
            const e0d = mlx.mlx_array_data_float32(e0_f32) orelse return error.Inkling2DraftRound;
            const ed = mlx.mlx_array_data_float32(e_f32) orelse return error.Inkling2DraftRound;
            return walkSelector(cd[0 .. m * k], ud[0 .. m * k], e0d[0..k], ed[0 .. (m - 1) * k * k], m, k, cfg.softcap, cfg.softcap_after, gumbel);
        } else {
            try ops.evalMany(&.{ cands_i32, unary_f32, e0_f32 });
            const cd = mlx.mlx_array_data_int32(cands_i32) orelse return error.Inkling2DraftRound;
            const ud = mlx.mlx_array_data_float32(unary_f32) orelse return error.Inkling2DraftRound;
            const e0d = mlx.mlx_array_data_float32(e0_f32) orelse return error.Inkling2DraftRound;
            return walkSelector(cd[0..k], ud[0..k], e0d[0..k], &.{}, 1, k, cfg.softcap, cfg.softcap_after, gumbel);
        }
    }
};

fn softcapVal(x: f32, cap: f32) f32 {
    return if (cap > 0) cap * std.math.tanh(x / cap) else x;
}

/// Host selector walk over `[m, k]` candidates with exact `after_correction` / `before_correction`
/// softcapping and optional 23-bit Shared Gumbel-Max coupling.
pub fn walkSelector(
    cands: []const i32,
    unary: []const f32,
    e0: []const f32,
    edges: []const f32,
    m: usize,
    k: usize,
    softcap: f32,
    softcap_after: bool,
    gumbel: ?kernels.GumbelSpec,
) DraftResult {
    var out: DraftResult = .{ .len = @intCast(m) };
    var prev: usize = 0;
    var sc: [32]f32 = undefined;
    for (0..m) |step| {
        var best_j: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (0..k) |j| {
            const u_raw = unary[step * k + j];
            const u = if (!softcap_after and softcap > 0) softcapVal(u_raw, softcap) else u_raw;
            const edge = if (step == 0) e0[j] else edges[(step - 1) * k * k + prev * k + j];
            const sum = u + edge;
            const capped = if (softcap_after and softcap > 0) softcapVal(sum, softcap) else sum;
            const score = if (gumbel) |gs|
                capped * gs.inv_t + gumbel23(gs.seed, gs.q0 + @as(u32, @intCast(step)), @intCast(cands[step * k + j]))
            else
                capped;
            sc[j] = score;
            if (j == 0 or score > best_v) {
                best_v = score;
                best_j = j;
            }
        }
        var tot: f32 = 0.0;
        const slope: f32 = if (gumbel != null) 1.15 else 1.0;
        for (0..k) |j| tot += @exp((sc[j] - best_v) * slope);
        out.tokens[step] = cands[step * k + best_j];
        out.confidence[step] = if (tot > 0 and std.math.isFinite(tot)) 1.0 / tot else 0.0;
        prev = best_j;
    }
    return out;
}

/// Embed `[1, 1]` token id through the trunk's (possibly quantized) embedding table without needing a Run ctx.
fn embedRow(g: *Graph, embed: *const embed_block.TableEmbed.Weights, id: Array) !Array {
    const t = &embed.table;
    if (t.s.ctx == null) return g.astype(try g.take(t.w, id, 0), embed.act);
    const w_row = try g.take(t.w, id, 0);
    const s_row = try g.take(t.s, id, 0);
    const b_row = if (t.b.ctx != null) try g.take(t.b, id, 0) else Array{};
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(
        &out,
        w_row,
        s_row,
        b_row,
        mlx.mlx_optional_int.some(@intCast(t.group_size)),
        mlx.mlx_optional_int.some(@intCast(t.bits)),
        t.mode.cstr(),
        .{ .ctx = null },
        .{ .value = embed.act, .has_value = true },
        g.s,
    ));
    return g.adopt(out);
}

const At = struct {
    d: *Drafter,
    w: *const Weights,
    g: *Graph,
    layer: ?u32 = null,
    use_f16: bool = false,
    layer_bits: u32 = 16,

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

    fn linear(a: At, base: []const u8, allow_quant: bool) !Linear {
        var buf: [160]u8 = undefined;
        const wname = try std.fmt.bufPrint(&buf, "{s}.weight", .{base});
        if (allow_quant and a.layer_bits < 16) {
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

    fn dynConv(a: At, prefix: []const u8) !DynConv {
        var b0: [160]u8 = undefined;
        var b1: [160]u8 = undefined;
        var bk = try a.raw(try std.fmt.bufPrint(&b0, "{s}.base_kernel", .{prefix}));
        if (a.use_f16 and ops.dtype(bk) == .bfloat16) bk = try a.g.astype(bk, .float16);
        const sh = ops.shape(bk);
        if (sh.len != 3 or sh[0] != 2) return error.InvalidDflash2Config;
        const base0 = try a.keep(try a.g.contiguous(try a.g.reshape(try a.g.sliceAxis(bk, 0, 0, 1), &.{ sh[1], sh[2] })));
        const base1 = try a.keep(try a.g.contiguous(try a.g.reshape(try a.g.sliceAxis(bk, 0, 1, 2), &.{ sh[1], sh[2] })));
        const proj = try a.linear(try std.fmt.bufPrint(&b1, "{s}.kernel_projection", .{prefix}), false);
        return .{ .base0 = base0, .base1 = base1, .proj = proj };
    }
};

// ── Tests ──

const testing = std.testing;

test "inkling2 dflash2: parseConfig reads real incoai/Inkling-2-Mini-DFlash2 contract" {
    const json_bytes =
        \\{
        \\  "architectures": ["DFlash2DraftModel"],
        \\  "dflash_config": {
        \\    "block_size": 8,
        \\    "conv_group_size": 16,
        \\    "conv_kernel_size": 2,
        \\    "final_logit_softcapping": 30.0,
        \\    "mask_token_id": 200056,
        \\    "selector_rank": 256,
        \\    "selector_softcap_mode": "after_correction",
        \\    "selector_top_k": 16,
        \\    "target_layer_ids": [5, 23, 41, 48]
        \\  },
        \\  "head_dim": 128,
        \\  "hidden_size": 2304,
        \\  "intermediate_size": 6912,
        \\  "num_attention_heads": 32,
        \\  "num_hidden_layers": 5,
        \\  "num_key_value_heads": 8,
        \\  "rms_norm_eps": 1e-06,
        \\  "rope_parameters": {"rope_theta": 10000.0},
        \\  "sliding_window": 2048
        \\}
    ;
    const c = try parseConfig(json_bytes, 2304, 48);
    try testing.expectEqual(@as(u32, 5), c.layers);
    try testing.expectEqual(@as(u32, 8), c.block);
    try testing.expectEqual(@as(u32, 200056), c.mask_token_id);
    try testing.expectEqual(@as(u32, 2048), c.window);
    try testing.expectEqual(@as(u32, 4), c.tap_count);
    try testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, c.taps[0..c.tap_count]);
    try testing.expect(c.softcap_after);
    try testing.expectEqual(@as(f32, 30.0), c.softcap);
}

test "inkling2 dflash2: walkSelector applies after_correction softcap and deterministic gumbel23" {
    const cands = [_]i32{ 10, 20, 30, 40, 11, 21, 31, 41 };
    const unary = [_]f32{ 10.0, 5.0, 1.0, 0.0, 2.0, 9.0, 1.0, 0.0 };
    const e0 = [_]f32{ -8.0, 4.0, 0.0, 0.0 }; // makes cand[1]=20 win (5+4=9 vs 10-8=2)
    // edges[prev=1, :] prefers cand[2]=31 (1 + 12 = 13 vs 9 + 0 = 9)
    var edges: [16]f32 = @splat(0.0);
    edges[1 * 4 + 2] = 12.0;

    const r = walkSelector(&cands, &unary, &e0, &edges, 2, 4, 30.0, true, null);
    try testing.expectEqual(@as(u32, 2), r.len);
    try testing.expectEqual(@as(i32, 20), r.tokens[0]);
    try testing.expectEqual(@as(i32, 31), r.tokens[1]);
    try testing.expect(r.confidence[0] > 0.9 and r.confidence[0] <= 1.0);
    try testing.expect(r.confidence[1] > 0.9 and r.confidence[1] <= 1.0);

    // Gumbel noise must be finite for arbitrary seeds/ids.
    const g1 = gumbel23(42, 100, 200055);
    const g2 = gumbel23(42, 100, 200055);
    try testing.expect(std.math.isFinite(g1) and g1 == g2);
}

test "inkling2 dflash2: GPU groupedDynConv matches op chain in FP16" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    var g = Graph.init(a, mlx.gpuStream());
    defer g.deinit();
    const L: c_int = 8;
    const H: c_int = 64;
    const GS: u32 = 16;
    const KS: c_int = 2;
    var x_buf: [L * H]f32 = undefined;
    var d_buf: [L * KS * (H / GS)]f32 = undefined;
    var b_buf: [KS * H]f32 = undefined;
    for (&x_buf, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13);
    for (&d_buf, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.17) * 0.25;
    for (&b_buf, 0..) |*v, i| v.* = if (i < H) 1.0 else -0.15;

    const x = try g.astype(try g.fromSlice(f32, &x_buf, &.{ 1, L, H }), .float16);
    const dyn = try g.astype(try g.fromSlice(f32, &d_buf, &.{ 1, L, KS, H / GS }), .float16);
    const base = try g.astype(try g.fromSlice(f32, &b_buf, &.{ KS, H }), .float16);
    const got = try g.astype(try groupedDynConv(&g, x, dyn, base, GS), .float32);
    const want = try g.astype(try groupedDynConvOps(&g, x, dyn, base, GS), .float32);
    try ops.evalMany(&.{ got, want });
    const gd = mlx.mlx_array_data_float32(got).?;
    const wd = mlx.mlx_array_data_float32(want).?;
    for (0..@intCast(L * H)) |i| {
        try testing.expectApproxEqAbs(wd[i], gd[i], 1e-3);
    }
}
