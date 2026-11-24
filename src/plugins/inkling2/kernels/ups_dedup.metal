// Inkling 2, multi-row verify (S = 2..16): grouped expert-deduplicated gate_up + soft-capped SwiGLU.
// Each threadgroup owns one dedup slot s (uniform across the TG), exiting immediately if SLOT_E[s] == 0xFFFFFFFFu.
// SUB16 == 1 (FMT == 0 NVFP4, I%2==0, K%16==0): 16-lane sub-warps with uint2 (128B exact cache-line loads, 9 exact
// full iters on K=2304, HALF scale decodes, hoisted 2^22, fma()). Each half-warp owns 1 intermediate row (2 per SG)
// across up to M=4 (row, pick) pairs -> 265 GB/s on M1 Max.
const uint s = thread_position_in_grid.z;
const uint ex = SLOT_E[s];
if (ex == 0xFFFFFFFFu) return;
const uint lane = thread_index_in_simdgroup;
const uint p0 = SLOT_P[s * 4 + 0];
const uint p1 = SLOT_P[s * 4 + 1];
const uint p2 = SLOT_P[s * 4 + 2];
const uint p3 = SLOT_P[s * 4 + 3];
const device T* xs[4] = {
    X + size_t(p0 / KSEL) * K,
    X + size_t((p1 == 0xFFFFFFFFu ? p0 : p1) / KSEL) * K,
    X + size_t((p2 == 0xFFFFFFFFu ? p0 : p2) / KSEL) * K,
    X + size_t((p3 == 0xFFFFFFFFu ? p0 : p3) / KSEL) * K
};
constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const uint ps[4] = { p0, p1, p2, p3 };

if (SUB16) {
    const uint hid = lane >> 4;
    const uint slane = lane & 15u;
    const uint m = thread_position_in_grid.y * 2 + hid;
    const size_t base = size_t(ex) * (2 * I) + 2 * m;
    const device uint2* wrow = (const device uint2*)(WQ + base * WPR);
    const device uchar* srow = (const device uchar*)WS + base * SPR;
    constexpr uint NG = K / 16;
    float acc[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#define INK2_UD16_LOOP(N) \
    for (uint g = slane; g < NG; g += 16) { \
        vec<T, 4> xv0[N], xv1[N], xv2[N], xv3[N]; \
        for (uint i = 0; i < N; i++) { \
            const device vec<T, 4>* xp = (const device vec<T, 4>*)(xs[i] + g * 16); \
            xv0[i] = xp[0]; xv1[i] = xp[1]; xv2[i] = xp[2]; xv3[i] = xp[3]; \
        } \
        for (uint r = 0; r < 2; r++) { \
            const uint2 q = wrow[r * NG + g]; \
            const float sf = mlxserve_fp4_scale(uint(srow[r * SPR + g])); \
            const float2 f0 = mlxserve_e2m1x2(q.x, 0), f1 = mlxserve_e2m1x2(q.x, 1); \
            const float2 f2 = mlxserve_e2m1x2(q.x, 2), f3 = mlxserve_e2m1x2(q.x, 3); \
            const float2 h0 = mlxserve_e2m1x2(q.y, 0), h1 = mlxserve_e2m1x2(q.y, 1); \
            const float2 h2 = mlxserve_e2m1x2(q.y, 2), h3 = mlxserve_e2m1x2(q.y, 3); \
            for (uint i = 0; i < N; i++) { \
                float d = fma(float(xv0[i][0]), f0.x, float(xv1[i][0]) * f0.y); \
                d = fma(float(xv0[i][1]), f1.x, fma(float(xv1[i][1]), f1.y, d)); \
                d = fma(float(xv0[i][2]), f2.x, fma(float(xv1[i][2]), f2.y, d)); \
                d = fma(float(xv0[i][3]), f3.x, fma(float(xv1[i][3]), f3.y, d)); \
                d = fma(float(xv2[i][0]), h0.x, fma(float(xv3[i][0]), h0.y, d)); \
                d = fma(float(xv2[i][1]), h1.x, fma(float(xv3[i][1]), h1.y, d)); \
                d = fma(float(xv2[i][2]), h2.x, fma(float(xv3[i][2]), h2.y, d)); \
                d = fma(float(xv2[i][3]), h3.x, fma(float(xv3[i][3]), h3.y, d)); \
                acc[r][i] = fma(d, sf, acc[r][i]); \
            } \
        } \
    }
    if (p1 == 0xFFFFFFFFu) { INK2_UD16_LOOP(1) }
    else if (p2 == 0xFFFFFFFFu) { INK2_UD16_LOOP(2) }
    else if (p3 == 0xFFFFFFFFu) { INK2_UD16_LOOP(3) }
    else { INK2_UD16_LOOP(4) }
#undef INK2_UD16_LOOP
    const float sc = (SCALED ? SC[ex] : 1.0f) * 4194304.0f;
    for (uint i = 0; i < 4; i++) {
        const uint pi = ps[i];
        if (pi == 0xFFFFFFFFu) break;
        float gv = acc[0][i], uv = acc[1][i];
        gv += simd_shuffle_xor(gv, 8); gv += simd_shuffle_xor(gv, 4); gv += simd_shuffle_xor(gv, 2); gv += simd_shuffle_xor(gv, 1);
        uv += simd_shuffle_xor(uv, 8); uv += simd_shuffle_xor(uv, 4); uv += simd_shuffle_xor(uv, 2); uv += simd_shuffle_xor(uv, 1);
        if (slane == 0) {
            const float g = SCALED ? (gv * sc) : float(T(gv * sc));
            const float u = SCALED ? (uv * sc) : float(T(uv * sc));
            const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
            ACT[size_t(pi) * I + m] = T(capped / (1.0f + precise::exp(-g)));
        }
    }
} else {
    const uint m = thread_position_in_grid.y;
    const size_t base = size_t(ex) * (2 * I) + 2 * m;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* srow = (const device uchar*)WS + base * SPR;
    const device uchar* arow = (const device uchar*)WQ + base * WBR;
    const device S* sa = (const device S*)WS + base * SPR;
    const device S* ba = (const device S*)WB + base * SPR;
    const uint nact = (p1 == 0xFFFFFFFFu) ? 1u : (p2 == 0xFFFFFFFFu ? 2u : (p3 == 0xFFFFFFFFu ? 3u : 4u));
    float ag[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float au[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint k = lane * 16; k < K; k += 512) {
        for (uint i = 0; i < nact; i++) {
            float xr[16];
            const device vec<T, 4>* xp = (const device vec<T, 4>*)(xs[i] + k);
            for (uint v = 0; v < 4; v++) {
                const vec<T, 4> x4 = xp[v];
                xr[4 * v + 0] = float(x4[0]);
                xr[4 * v + 1] = float(x4[1]);
                xr[4 * v + 2] = float(x4[2]);
                xr[4 * v + 3] = float(x4[3]);
            }
            if (FMT == 3) {
                ag[i] += msv_ink2_affine_chunk<S, BITS, GS>(arow, sa, ba, k / 16, xr);
                au[i] += msv_ink2_affine_chunk<S, BITS, GS>(arow + WBR, sa + SPR, ba + SPR, k / 16, xr);
            } else {
                ag[i] += msv_ink2_chunk(FMT, wrow, srow, k / 16, xr);
                au[i] += msv_ink2_chunk(FMT, wrow + WPR, srow + SPR, k / 16, xr);
            }
        }
    }
    const float sc = SCALED ? SC[ex] : 1.0f;
    for (uint i = 0; i < 4; i++) {
        const uint pi = ps[i];
        if (pi == 0xFFFFFFFFu) break;
        const float g = SCALED ? (simd_sum(ag[i]) * sc) : float(T(simd_sum(ag[i])));
        const float u = SCALED ? (simd_sum(au[i]) * sc) : float(T(simd_sum(au[i])));
        const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
        if (lane == 0) ACT[size_t(pi) * I + m] = T(capped / (1.0f + precise::exp(-g)));
    }
}
