// Inkling 2, a few rows: the picked experts' gate and up projections and the soft-capped SwiGLU in one launch, the
// packed bank read in place. grid (32, I / ROWS, S * KSEL), threadgroup (32, SGS, 1): a SIMD group owns ROWS
// intermediate rows of one pick (their gate rows 2r and up rows 2r + 1 of WQ [E, 2 I, K * bits / 32] u32, scales WS
// [E, 2 I, K / GS] u8) and keeps x in registers across the rows. FMT 0 (NVFP4) uses an 8-value-per-word loop so
// K=2304 divides cleanly into 288/32 = 9 full iterations with zero idle SIMD lanes. FMT 3 is affine.
// When SCALED (NVFP4), multiplies by SC[IDX[e]] in f32 BEFORE any rounding to T so fp16 activations never overflow.
const uint lane = thread_index_in_simdgroup;
const uint row0 = thread_position_in_grid.y * ROWS;
const uint e = thread_position_in_grid.z;
const device T* x = X + size_t(e / KSEL) * K;
constexpr uint WPR = FMT == 2 ? K / 4 : K / 8;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const uint ex = IDX[e];
const size_t base = size_t(ex) * (2 * I) + 2 * row0;
const device uint* wrow = WQ + base * WPR;
const device uchar* srow = (const device uchar*)WS + base * SPR;
const device uchar* arow = (const device uchar*)WQ + base * WBR;
const device S* sa = (const device S*)WS + base * SPR;
const device S* ba = (const device S*)WB + base * SPR;
float acc[2 * ROWS];
for (uint r = 0; r < 2 * ROWS; r++) acc[r] = 0.0f;
if (FMT == 0) {
    for (uint w = lane; w < WPR; w += 32) {
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + w * 8);
        const vec<T, 4> x0 = xp[0], x1 = xp[1];
        const float xr0 = float(x0[0]), xr1 = float(x0[1]), xr2 = float(x0[2]), xr3 = float(x0[3]);
        const float xr4 = float(x1[0]), xr5 = float(x1[1]), xr6 = float(x1[2]), xr7 = float(x1[3]);
        const uint sidx = w >> 1;
        for (uint r = 0; r < 2 * ROWS; r++) {
            const uint q = wrow[r * WPR + w];
            float d = 0.0f;
            const float2 p0 = mlxserve_e2m1x2(q, 0); d += xr0 * p0.x + xr4 * p0.y;
            const float2 p1 = mlxserve_e2m1x2(q, 1); d += xr1 * p1.x + xr5 * p1.y;
            const float2 p2 = mlxserve_e2m1x2(q, 2); d += xr2 * p2.x + xr6 * p2.y;
            const float2 p3 = mlxserve_e2m1x2(q, 3); d += xr3 * p3.x + xr7 * p3.y;
            acc[r] += d * mlxserve_fp4_scale(uint(srow[r * SPR + sidx])) * 4194304.0f;
        }
    }
} else {
    for (uint k = lane * 16; k < K; k += 512) {
        float xr[16];
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + k);
        for (uint v = 0; v < 4; v++) {
            const vec<T, 4> x4 = xp[v];
            xr[4 * v + 0] = float(x4[0]);
            xr[4 * v + 1] = float(x4[1]);
            xr[4 * v + 2] = float(x4[2]);
            xr[4 * v + 3] = float(x4[3]);
        }
        for (uint r = 0; r < 2 * ROWS; r++)
            acc[r] += FMT == 3 ? msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, k / 16, xr)
                               : msv_ink2_chunk(FMT, wrow + r * WPR, srow + r * SPR, k / 16, xr);
    }
}
const float sc = SCALED ? SC[ex] : 1.0f;
for (uint r = 0; r < ROWS; r++) {
    const float g = SCALED ? (simd_sum(acc[2 * r]) * sc) : float(T(simd_sum(acc[2 * r])));
    const float u = SCALED ? (simd_sum(acc[2 * r + 1]) * sc) : float(T(simd_sum(acc[2 * r + 1])));
    const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
    if (lane == r) ACT[e * I + row0 + r] = T(capped / (1.0f + precise::exp(-g)));
}
