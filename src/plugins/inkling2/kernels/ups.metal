// Inkling 2, one row (or small non-grouped fallback): router picks (when ROUTE), picked experts' gate/up projections
// and soft-capped SwiGLU in one launch, the packed bank read in place.
// SUB16 == 1 (FMT == 0 NVFP4, K%16==0, I%4==0): 16-lane sub-warps with uint2 (128B exact cache-line loads, 9 exact
// full iters on K=2304, hoisted 2^22, fma(), f32 global_scale before narrowing) -> 273 GB/s on M1 Max.
// SUB16 == 0: general fallback supporting MXFP4, MXFP8, affine (2..8 bits) and arbitrary I.
const uint lane = thread_index_in_simdgroup;
const uint e = thread_position_in_grid.z;
const uint tid = thread_position_in_threadgroup.x + 32 * thread_position_in_threadgroup.y;
constexpr uint TGN = 32 * SGS;
threadgroup float key[E];
threadgroup float raw[KSEL];
threadgroup uint pick[KSEL];
uint ex = 0;
if (ROUTE) {
    const uint srow0 = (e / KSEL) * E;
    float sig[(E + TGN - 1) / TGN];
    for (uint f = tid, j = 0; f < E; f += TGN, j++) {
        sig[j] = 1.0f / (1.0f + exp(-SCR[srow0 + f]));
        key[f] = sig[j] + RB[f];
    }
    if (tid < KSEL) { pick[tid] = tid; raw[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint f = tid, j = 0; f < E; f += TGN, j++) {
        const float mine = key[f];
        uint rank = 0;
        for (uint h = 0; h < E; h++) rank += (key[h] > mine || (key[h] == mine && h < f)) ? 1u : 0u;
        if (rank < KSEL && mine == mine) { pick[rank] = f; raw[rank] = sig[j]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ex = pick[e % KSEL];
    if (thread_position_in_grid.y == 0 && tid == 0) {
        float tot = 0.0f;
        for (uint k = 0; k < KSEL; k++) tot += raw[k];
        PIDX[e] = ex;
        PWTS[e] = raw[e % KSEL] / (tot + 1e-20f) * RS[0];
    }
} else ex = IDX[e];
const device T* x = X + size_t(e / KSEL) * K;
constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;

if (SUB16) {
    const uint hid = lane >> 4;
    const uint slane = lane & 15u;
    const uint row0 = (thread_position_in_grid.y * 2 + hid) * 2;
    const size_t base = size_t(ex) * (2 * I) + 2 * row0;
    const device uint2* wrow = (const device uint2*)(WQ + base * WPR);
    const device uchar* srow = (const device uchar*)WS + base * SPR;
    constexpr uint NG = K / 16;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint g = slane; g < NG; g += 16) {
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + g * 16);
        const vec<T, 4> v0 = xp[0], v1 = xp[1], v2 = xp[2], v3 = xp[3];
        for (uint r = 0; r < 4; r++) {
            const uint2 q = wrow[r * NG + g];
            const float2 p0 = mlxserve_e2m1x2(q.x, 0), p1 = mlxserve_e2m1x2(q.x, 1);
            const float2 p2 = mlxserve_e2m1x2(q.x, 2), p3 = mlxserve_e2m1x2(q.x, 3);
            float d = fma(float(v0[0]), p0.x, float(v1[0]) * p0.y);
            d = fma(float(v0[1]), p1.x, fma(float(v1[1]), p1.y, d));
            d = fma(float(v0[2]), p2.x, fma(float(v1[2]), p2.y, d));
            d = fma(float(v0[3]), p3.x, fma(float(v1[3]), p3.y, d));
            const float2 q0 = mlxserve_e2m1x2(q.y, 0), q1 = mlxserve_e2m1x2(q.y, 1);
            const float2 q2 = mlxserve_e2m1x2(q.y, 2), q3 = mlxserve_e2m1x2(q.y, 3);
            d = fma(float(v2[0]), q0.x, fma(float(v3[0]), q0.y, d));
            d = fma(float(v2[1]), q1.x, fma(float(v3[1]), q1.y, d));
            d = fma(float(v2[2]), q2.x, fma(float(v3[2]), q2.y, d));
            d = fma(float(v2[3]), q3.x, fma(float(v3[3]), q3.y, d));
            acc[r] = fma(d, mlxserve_fp4_scale(uint(srow[r * SPR + g])), acc[r]);
        }
    }
    const float sc = SCALED ? SC[ex] : 1.0f;
    for (uint r = 0; r < 2; r++) {
        float gv = acc[2 * r], uv = acc[2 * r + 1];
        gv += simd_shuffle_xor(gv, 8); gv += simd_shuffle_xor(gv, 4); gv += simd_shuffle_xor(gv, 2); gv += simd_shuffle_xor(gv, 1);
        uv += simd_shuffle_xor(uv, 8); uv += simd_shuffle_xor(uv, 4); uv += simd_shuffle_xor(uv, 2); uv += simd_shuffle_xor(uv, 1);
        if (slane == r) {
            const float g = PRESCALE ? (gv * 4194304.0f * sc) : (float(T(gv * 4194304.0f)) * sc);
            const float u = PRESCALE ? (uv * 4194304.0f * sc) : (float(T(uv * 4194304.0f)) * sc);
            const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
            ACT[e * I + row0 + r] = T(capped / (1.0f + precise::exp(-g)));
        }
    }
} else {
    const uint row0 = thread_position_in_grid.y * ROWS;
    const size_t base = size_t(ex) * (2 * I) + 2 * row0;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* srow = (const device uchar*)WS + base * SPR;
    const device uchar* arow = (const device uchar*)WQ + base * WBR;
    const device S* sa = (const device S*)WS + base * SPR;
    const device S* ba = (const device S*)WB + base * SPR;
    float acc[2 * ROWS];
    for (uint r = 0; r < 2 * ROWS; r++) acc[r] = 0.0f;
    for (uint k = lane * 16; k < K; k += 512) {
        float xr[16];
        for (uint i = 0; i < 16; i++) xr[i] = float(x[k + i]);
        for (uint r = 0; r < 2 * ROWS; r++)
            acc[r] += FMT == 3 ? msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, k / 16, xr)
                               : msv_ink2_chunk(FMT, wrow + r * WPR, srow + r * SPR, k / 16, xr);
    }
    const float sc = SCALED ? SC[ex] : 1.0f;
    for (uint r = 0; r < ROWS; r++) {
        const float g = PRESCALE ? (simd_sum(acc[2 * r]) * sc) : (float(T(simd_sum(acc[2 * r]))) * sc);
        const float u = PRESCALE ? (simd_sum(acc[2 * r + 1]) * sc) : (float(T(simd_sum(acc[2 * r + 1]))) * sc);
        const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
        if (lane == r) ACT[e * I + row0 + r] = T(capped / (1.0f + precise::exp(-g)));
    }
}
