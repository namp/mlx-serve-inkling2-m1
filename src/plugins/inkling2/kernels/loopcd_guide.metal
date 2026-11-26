// Inkling 2 LoopCD-Logits contrastive guidance kernel (arXiv:2610.02185, matching oMLX PR #5640).
// FINAL, REF: [N, V] f32 soft-capped logits (trimmed to unpadded_vocab_size); THINK: [N] f32 (0/1, used when SKIP_THINK);
// PR: [1] f32 { max_strength (0.5) }; CTRL: [1] i32 { first_control_token (199998) }.
// weight = max_strength * (1 - (p_top1 - p_top2)) where p = softmax(FINAL).
// SKIP_THINK == 0: OUT = FINAL + weight * (FINAL - REF).
// SKIP_THINK == 1: if THINK[row] > 0.5, OUT = FINAL; else ordinary tokens (< first_ctrl) are contrasted and shifted by
//   (logsumexp_ord(FINAL) - logsumexp_ord(guided)) so every control token's softmax probability is preserved identically.
// grid (256, N, 1), threadgroup (256, 1, 1).
const uint row = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int V = FINAL_shape[1];
const auto frow = FINAL + size_t(row) * V;
const auto rrow = REF + size_t(row) * V;
device float* orow = OUT + size_t(row) * V;

if constexpr (SKIP_THINK != 0) {
    if (THINK[row] > 0.5f) {
        for (int v = int(tid); v < V; v += 256) orow[v] = frow[v];
        return;
    }
}

const int first_ctrl = SKIP_THINK != 0 ? min(CTRL[0], V) : V;

// Pass 1: top-2 of frow[0..V) (exact tie handling) and max of ordinary tokens [0..first_ctrl).
float t1 = -INFINITY, t2 = -INFINITY;
float max_ord_f = -INFINITY;
for (int v = int(tid); v < V; v += 256) {
    const float x = frow[v];
    if (x > t1) { t2 = t1; t1 = x; }
    else if (x > t2) { t2 = x; }
    if constexpr (SKIP_THINK != 0) {
        if (v < first_ctrl) max_ord_f = max(max_ord_f, x);
    }
}

#pragma clang loop unroll(full)
for (ushort off = 16; off > 0; off >>= 1) {
    const float o1 = simd_shuffle_xor(t1, off);
    const float o2 = simd_shuffle_xor(t2, off);
    if (o1 > t1) { t2 = max(t1, o2); t1 = o1; }
    else { t2 = max(t2, o1); }
}
if constexpr (SKIP_THINK != 0) max_ord_f = simd_max(max_ord_f);

threadgroup float tg1[8], tg2[8], tg_aux[8];
if (lane == 0u) {
    tg1[sg] = t1;
    tg2[sg] = t2;
    tg_aux[sg] = max_ord_f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
    float g1 = tg1[0], g2 = tg2[0], mo = tg_aux[0];
    for (uint s = 1u; s < 8u; ++s) {
        const float o1 = tg1[s], o2 = tg2[s];
        if (o1 > g1) { g2 = max(g1, o2); g1 = o1; }
        else { g2 = max(g2, o1); }
        mo = max(mo, tg_aux[s]);
    }
    tg1[0] = g1;
    tg2[0] = g2;
    tg_aux[0] = mo;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const float m1 = tg1[0];
const float m2 = tg2[0];
const float mo_f = tg_aux[0];

// Pass 2: softmax partition sum over all tokens, plus ordinary-token partition sum when SKIP_THINK.
float sum_all = 0.0f;
float sum_ord_f = 0.0f;
for (int v = int(tid); v < V; v += 256) {
    const float x = frow[v];
    sum_all += precise::exp(x - m1);
    if constexpr (SKIP_THINK != 0) {
        if (v < first_ctrl) sum_ord_f += precise::exp(x - mo_f);
    }
}
sum_all = simd_sum(sum_all);
if constexpr (SKIP_THINK != 0) sum_ord_f = simd_sum(sum_ord_f);
if (lane == 0u) {
    tg1[sg] = sum_all;
    tg_aux[sg] = sum_ord_f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0u) {
    float sa = 0.0f, so = 0.0f;
    for (uint s = 0u; s < 8u; ++s) { sa += tg1[s]; so += tg_aux[s]; }
    tg1[0] = sa;
    tg_aux[0] = so;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const float total_sum = tg1[0];
const float margin = (total_sum > 0.0f) ? ((1.0f - precise::exp(m2 - m1)) / total_sum) : 1.0f;
const float weight = PR[0] * (1.0f - margin);

if constexpr (SKIP_THINK == 0) {
    for (int v = int(tid); v < V; v += 256) {
        const float f = frow[v];
        orow[v] = f + weight * (f - rrow[v]);
    }
} else {
    if (first_ctrl <= 0) {
        for (int v = int(tid); v < V; v += 256) orow[v] = frow[v];
        return;
    }
    const float final_mass = mo_f + precise::log(tg_aux[0]);

    // Pass 3a: max over ordinary guided logits.
    float max_ord_g = -INFINITY;
    for (int v = int(tid); v < first_ctrl; v += 256) {
        const float f = frow[v];
        max_ord_g = max(max_ord_g, f + weight * (f - rrow[v]));
    }
    max_ord_g = simd_max(max_ord_g);
    if (lane == 0u) tg_aux[sg] = max_ord_g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float mg = tg_aux[0];
        for (uint s = 1u; s < 8u; ++s) mg = max(mg, tg_aux[s]);
        tg_aux[0] = mg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mo_g = tg_aux[0];

    // Pass 3b: sum exp over ordinary guided logits.
    float sum_ord_g = 0.0f;
    for (int v = int(tid); v < first_ctrl; v += 256) {
        const float f = frow[v];
        const float g = f + weight * (f - rrow[v]);
        sum_ord_g += precise::exp(g - mo_g);
    }
    sum_ord_g = simd_sum(sum_ord_g);
    if (lane == 0u) tg_aux[sg] = sum_ord_g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sg_tot = 0.0f;
        for (uint s = 0u; s < 8u; ++s) sg_tot += tg_aux[s];
        tg_aux[0] = sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float shift = final_mass - (mo_g + precise::log(tg_aux[0]));
    for (int v = int(tid); v < V; v += 256) {
        const float f = frow[v];
        orow[v] = (v < first_ctrl) ? (f + weight * (f - rrow[v]) + shift) : f;
    }
}
