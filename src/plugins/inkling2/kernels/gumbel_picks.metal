// Inkling 2: exact Gumbel-Max target pick per verify row r in 0..R-1.
// LG [R, V] (soft-capped target logits), SUPP [V] bool (when HAS_SUPP == 1), SP [3] uint32 = (bits(1/T), seed, q0).
// Computes argmax_{v: !SUPP[v]} (float(LG[r, v]) * inv_t + msv_ink2_gumbel(seed, q0 + r, v)) -> OUT [R] int32.
// Grid: (256, R, 1), threadgroup: (256, 1, 1).
threadgroup float rm[8];
threadgroup uint ri[8];
constexpr uint NONE = 0xffffffffu;
const uint r = threadgroup_position_in_grid.y;
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
const uint V = LG_shape[1];
const float inv_t = as_type<float>(SP[0]);
const uint seed = SP[1];
const uint q0 = SP[2] + r;
const device T* row = LG + size_t(r) * V;
float m = -INFINITY;
uint mi = NONE;
for (uint v = t; v < V; v += 256) {
    if (HAS_SUPP && SUPP[v]) continue;
    const float sc = float(row[v]) * inv_t + msv_ink2_gumbel(seed, q0, v);
    if (mi == NONE || msv_ink2_better(sc, v, m, mi)) {
        m = sc;
        mi = v;
    }
}
for (ushort off = 16; off > 0; off >>= 1) {
    const float om = simd_shuffle_xor(m, off);
    const uint oi = simd_shuffle_xor(mi, off);
    if (oi != NONE && (mi == NONE || msv_ink2_better(om, oi, m, mi))) {
        m = om;
        mi = oi;
    }
}
if (lane == 0) {
    rm[sg] = m;
    ri[sg] = mi;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0) {
    float top = rm[0];
    uint at_top = ri[0];
    for (uint g = 1; g < 8; g++) {
        if (ri[g] != NONE && (at_top == NONE || msv_ink2_better(rm[g], ri[g], top, at_top))) {
            top = rm[g];
            at_top = ri[g];
        }
    }
    OUT[r] = int(at_top);
}
