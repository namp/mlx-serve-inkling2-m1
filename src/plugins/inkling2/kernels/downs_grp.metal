// Inkling 2, a few rows: the down projection of the picks grouped by expert (group.metal), a tile of up to 4 picks reading
// the expert's rows once. Applies SC[exp_id] in float32 BEFORE storing T(val) so NVFP4 never overflows FP16.
// SUB8 == 1 (FMT == 0 NVFP4, D%8==0, I%16==0): 8-lane sub-warps (9 exact full iters on I=1152, zero idle lanes,
// uint2 + fma() + hoisted 2^22 * SC[exp_id] + cnt-specialized unroll) -> 267 GB/s on M1 Max.
// SUB8 == 0: 16-lane sub-warp general fallback supporting MXFP4, MXFP8, affine (2..8 bits) and arbitrary D%4==0.
const uint tile = threadgroup_position_in_grid.x;
const uint exp_id = TEXP[tile];
if (exp_id == 0xFFFFFFFFu) return;
const uint cnt = TCNT[tile];
const uint lane = thread_index_in_simdgroup;
const uint p0 = TPICK[tile * 4u + 0u];
const uint p1 = (cnt > 1u) ? TPICK[tile * 4u + 1u] : p0;
const uint p2 = (cnt > 2u) ? TPICK[tile * 4u + 2u] : p0;
const uint p3 = (cnt > 3u) ? TPICK[tile * 4u + 3u] : p0;

constexpr uint WPR = FMT == 2 ? I / 4 : I / 8;
constexpr uint WBR = I * BITS / 8;
constexpr uint SPR = I / GS;

if (SUB8) {
    const uint qid = lane >> 3;
    const uint qlane = lane & 7u;
    const uint col0 = (thread_position_in_grid.y * 4u + qid) * 2u;
    const size_t base = size_t(exp_id) * D + col0;
    const device uint2* wrow = (const device uint2*)(WQ + base * WPR);
    const device uchar* srow = (const device uchar*)WS + base * SPR;
    constexpr uint NG = I / 16u;
    const device T* xs[4] = {
        ACT + size_t(p0) * I,
        ACT + size_t(p1) * I,
        ACT + size_t(p2) * I,
        ACT + size_t(p3) * I
    };
    float acc[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#define INK2_DG8_LOOP(N) \
    for (uint g = qlane; g < NG; g += 8u) { \
        vec<T, 4> xv0[N], xv1[N], xv2[N], xv3[N]; \
        for (uint i = 0; i < N; i++) { \
            const device vec<T, 4>* xp = (const device vec<T, 4>*)(xs[i] + g * 16u); \
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
    if (cnt == 1u) { INK2_DG8_LOOP(1) }
    else if (cnt == 2u) { INK2_DG8_LOOP(2) }
    else if (cnt == 3u) { INK2_DG8_LOOP(3) }
    else { INK2_DG8_LOOP(4) }
#undef INK2_DG8_LOOP
    const float sc = (SCALED ? SC[exp_id] : 1.0f) * 4194304.0f;
    const uint ps[4] = { p0, p1, p2, p3 };
    for (uint i = 0; i < cnt; i++) {
        const uint pk = ps[i];
        for (uint r = 0; r < 2; r++) {
            float v = acc[r][i];
            v += simd_shuffle_xor(v, 4);
            v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 1);
            if (qlane == r) DN[size_t(pk) * D + col0 + r] = T(v * sc);
        }
    }
} else {
    const uint sub = lane & 15u;
    const uint half_sg = lane >> 4;
    const uint row0 = (thread_position_in_grid.y * 2u + half_sg) * 2u;
    const size_t base = size_t(exp_id) * D + row0;
    const device uint* wrow0 = WQ + base * WPR;
    const device uint* wrow1 = wrow0 + WPR;
    const device uchar* srow0 = (const device uchar*)WS + base * SPR;
    const device uchar* srow1 = srow0 + SPR;
    const device uchar* arow0 = (const device uchar*)WQ + base * WBR;
    const device uchar* arow1 = arow0 + WBR;
    const device S* sa0 = (const device S*)WS + base * SPR;
    const device S* sa1 = sa0 + SPR;
    const device S* ba0 = (const device S*)WB + base * SPR;
    const device S* ba1 = ba0 + SPR;

    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;

    for (uint k = sub * 16u; k < I; k += 256u) {
        const uint c = k >> 4;
        half4 w00, w01, w02, w03, w10, w11, w12, w13;
        float s0 = 0.0f, s1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
        if (FMT == 3) {
            msv_ink2_decode16_affine_v4<S, BITS, GS>(arow0, sa0, ba0, c, w00, w01, w02, w03, s0, b0);
            msv_ink2_decode16_affine_v4<S, BITS, GS>(arow1, sa1, ba1, c, w10, w11, w12, w13, s1, b1);
        } else {
            msv_ink2_decode16_v4(FMT, wrow0, srow0, c, w00, w01, w02, w03, s0);
            msv_ink2_decode16_v4(FMT, wrow1, srow1, c, w10, w11, w12, w13, s1);
        }
        const float4 f00 = float4(w00), f01 = float4(w01), f02 = float4(w02), f03 = float4(w03);
        const float4 f10 = float4(w10), f11 = float4(w11), f12 = float4(w12), f13 = float4(w13);

        #define ACC_DN(PICK_IDX, A0, A1) do { \
            const device vec<T, 4>* xp = (const device vec<T, 4>*)(ACT + size_t(PICK_IDX) * I + k); \
            const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]); \
            const float d0 = dot(x0, f00) + dot(x1, f01) + dot(x2, f02) + dot(x3, f03); \
            const float d1 = dot(x0, f10) + dot(x1, f11) + dot(x2, f12) + dot(x3, f13); \
            if (FMT == 3) { \
                const float sx = dot(x0 + x1 + x2 + x3, float4(1.0f)); \
                A0 += d0 * s0 + b0 * sx; \
                A1 += d1 * s1 + b1 * sx; \
            } else { \
                A0 += d0 * s0; \
                A1 += d1 * s1; \
            } \
        } while (0)

        ACC_DN(p0, a00, a10);
        if (cnt > 1u) ACC_DN(p1, a01, a11);
        if (cnt > 2u) ACC_DN(p2, a02, a12);
        if (cnt > 3u) ACC_DN(p3, a03, a13);
        #undef ACC_DN
    }

    #define SUM16(v) do { \
        v += simd_shuffle_xor(v, 8); \
        v += simd_shuffle_xor(v, 4); \
        v += simd_shuffle_xor(v, 2); \
        v += simd_shuffle_xor(v, 1); \
    } while (0)

    SUM16(a00); SUM16(a10);
    if (cnt > 1u) { SUM16(a01); SUM16(a11); }
    if (cnt > 2u) { SUM16(a02); SUM16(a12); }
    if (cnt > 3u) { SUM16(a03); SUM16(a13); }
    #undef SUM16

    const float sc = SCALED ? SC[exp_id] : 1.0f;
    if (sub < cnt) {
        const float r0v = (sub == 0u) ? a00 : ((sub == 1u) ? a01 : ((sub == 2u) ? a02 : a03));
        const float r1v = (sub == 0u) ? a10 : ((sub == 1u) ? a11 : ((sub == 2u) ? a12 : a13));
        const uint pk = (sub == 0u) ? p0 : ((sub == 1u) ? p1 : ((sub == 2u) ? p2 : p3));
        DN[size_t(pk) * D + row0] = T(r0v * sc);
        DN[size_t(pk) * D + row0 + 1u] = T(r1v * sc);
    }
}
