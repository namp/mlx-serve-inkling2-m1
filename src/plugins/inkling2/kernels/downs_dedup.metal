// Inkling 2, multi-row verify (S = 2..16): grouped expert-deduplicated down projection.
// Each threadgroup owns one dedup slot s (uniform across the TG), exiting immediately if SLOT_E[s] == 0xFFFFFFFFu.
// SUB8 == 1 (FMT == 0 NVFP4, D%8==0, I%16==0): 8-lane sub-warps with uint2 (9 exact full iters on I=1152, HALF
// scale decodes, hoisted 2^22, fma()). Each quarter-warp owns 2 output columns (8 per SG) across up to M=4 pairs
// -> 267 GB/s on M1 Max. Writes SC[ex]-scaled f32 DN [S * KSEL, 1, D].
const uint s = thread_position_in_grid.z;
const uint ex = SLOT_E[s];
if (ex == 0xFFFFFFFFu) return;
const uint lane = thread_index_in_simdgroup;
const uint p0 = SLOT_P[s * 4 + 0];
const uint p1 = SLOT_P[s * 4 + 1];
const uint p2 = SLOT_P[s * 4 + 2];
const uint p3 = SLOT_P[s * 4 + 3];
const device T* xs[4] = {
    ACT + size_t(p0) * I,
    ACT + size_t(p1 == 0xFFFFFFFFu ? p0 : p1) * I,
    ACT + size_t(p2 == 0xFFFFFFFFu ? p0 : p2) * I,
    ACT + size_t(p3 == 0xFFFFFFFFu ? p0 : p3) * I
};
constexpr uint WPR = FMT == 2 ? I / 4 : I / 8;
constexpr uint WBR = I * BITS / 8;
constexpr uint SPR = I / GS;
const uint ps[4] = { p0, p1, p2, p3 };

if (SUB8) {
    const uint qid = lane >> 3;
    const uint qlane = lane & 7u;
    const uint col0 = (thread_position_in_grid.y * 4 + qid) * 2;
    const size_t base = size_t(ex) * D + col0;
    const device uint2* wrow = (const device uint2*)(WQ + base * WPR);
    const device uchar* srow = (const device uchar*)WS + base * SPR;
    constexpr uint NG = I / 16;
    float acc[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#define INK2_DD8_LOOP(N) \
    for (uint g = qlane; g < NG; g += 8) { \
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
    if (p1 == 0xFFFFFFFFu) { INK2_DD8_LOOP(1) }
    else if (p2 == 0xFFFFFFFFu) { INK2_DD8_LOOP(2) }
    else if (p3 == 0xFFFFFFFFu) { INK2_DD8_LOOP(3) }
    else { INK2_DD8_LOOP(4) }
#undef INK2_DD8_LOOP
    const float sc = (SCALED ? SC[ex] : 1.0f) * 4194304.0f;
    for (uint i = 0; i < 4; i++) {
        const uint pi = ps[i];
        if (pi == 0xFFFFFFFFu) break;
        for (uint r = 0; r < 2; r++) {
            float v = acc[r][i];
            v += simd_shuffle_xor(v, 4); v += simd_shuffle_xor(v, 2); v += simd_shuffle_xor(v, 1);
            if (qlane == r) DN[size_t(pi) * D + col0 + r] = v * sc;
        }
    }
} else {
    const uint col0 = thread_position_in_grid.y * ROWS;
    const size_t base = size_t(ex) * D + col0;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* srow = (const device uchar*)WS + base * SPR;
    const device uchar* arow = (const device uchar*)WQ + base * WBR;
    const device S* sa = (const device S*)WS + base * SPR;
    const device S* ba = (const device S*)WB + base * SPR;
    const uint nact = (p1 == 0xFFFFFFFFu) ? 1u : (p2 == 0xFFFFFFFFu ? 2u : (p3 == 0xFFFFFFFFu ? 3u : 4u));
    float a0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float a1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint k = lane * 16; k < I; k += 512) {
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
                a0[i] += msv_ink2_affine_chunk<S, BITS, GS>(arow, sa, ba, k / 16, xr);
                if (ROWS == 2) a1[i] += msv_ink2_affine_chunk<S, BITS, GS>(arow + WBR, sa + SPR, ba + SPR, k / 16, xr);
            } else {
                a0[i] += msv_ink2_chunk(FMT, wrow, srow, k / 16, xr);
                if (ROWS == 2) a1[i] += msv_ink2_chunk(FMT, wrow + WPR, srow + SPR, k / 16, xr);
            }
        }
    }
    const float sc = SCALED ? SC[ex] : 1.0f;
    for (uint i = 0; i < 4; i++) {
        const uint pi = ps[i];
        if (pi == 0xFFFFFFFFu) break;
        const float v0 = simd_sum(a0[i]) * sc;
        if (lane == 0) DN[size_t(pi) * D + col0] = v0;
        if (ROWS == 2) {
            const float v1 = simd_sum(a1[i]) * sc;
            if (lane == 1) DN[size_t(pi) * D + col0 + 1] = v1;
        }
    }
}
