// Inkling 2 prefill NVFP4 expert worklist planner: one 256-thread threadgroup histograms unsorted pair expert ids
// SEL[0..N_PAIRS), computes per-expert segment offsets SEG_START[0..NE] and BM-row tile offsets ITEM_START[0..NE],
// and scatters unsorted pair indices into expert-contiguous ORDER[0..N_PAIRS). Zero host sync, shared by gate_up & down.
// grid (256, 1, 1), threadgroup (256, 1, 1). Template args: NE (num experts <= 512), BM (rows per tile, 16).
const uint tid = thread_position_in_threadgroup.x;
const uint n_pairs = uint(SEL_shape[0]);

threadgroup atomic_uint counts[NE];
threadgroup atomic_uint cursor[NE];
threadgroup uint seg[NE + 1];
threadgroup uint items[NE + 1];

for (uint e = tid; e < NE; e += 256u) {
    atomic_store_explicit(&counts[e], 0u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint p = tid; p < n_pairs; p += 256u) {
    const uint e = SEL[p];
    if (e < NE) {
        atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
    uint s_acc = 0u;
    uint i_acc = 0u;
    seg[0] = 0u;
    items[0] = 0u;
    SEG_START[0] = 0u;
    ITEM_START[0] = 0u;
    for (uint e = 0u; e < NE; ++e) {
        const uint c = atomic_load_explicit(&counts[e], memory_order_relaxed);
        atomic_store_explicit(&cursor[e], s_acc, memory_order_relaxed);
        s_acc += c;
        i_acc += (c + BM - 1u) / BM;
        seg[e + 1u] = s_acc;
        items[e + 1u] = i_acc;
        SEG_START[e + 1u] = s_acc;
        ITEM_START[e + 1u] = i_acc;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint p = tid; p < n_pairs; p += 256u) {
    const uint e = SEL[p];
    if (e < NE) {
        const uint dst = atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed);
        ORDER[dst] = p;
    }
}
