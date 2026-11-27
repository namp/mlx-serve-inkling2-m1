// Inkling 2, barrier-free down projections and weighted sum across all KSEL experts in one launch; with TAIL
// (one row) the MLP's short conv, residual add, and 1-row conv-state update NST too. grid (32, D / ROWS, S),
// threadgroup (32, BG, 1): each SIMD group owns ROWS output columns of one token row and sweeps all KSEL * (I / 16)
// chunks with zero threadgroup barriers and zero idle SIMD lanes (8 * 72 = 576 = 18 * 32 on Inkling-2-Mini).
// FMT == 0 hoists 2^22 into sc_cache and uses fma() directly over uint2 groups -> 237 GB/s on M1 Max.
const uint lane = thread_index_in_simdgroup;
const uint row0 = thread_position_in_grid.y * ROWS;
const uint srow = thread_position_in_grid.z;
constexpr uint WPR = FMT == 2 ? I / 4 : I / 8;
constexpr uint WBR = I * BITS / 8;
constexpr uint SPR = I / GS;
constexpr uint NC = I / 16;
constexpr uint TOT = KSEL * NC;
const uint SW = ST_shape[2];
const uint OFF = SW - D;
float acc[ROWS];
for (uint r = 0; r < ROWS; r++) acc[r] = 0.0f;
uint ex_cache[KSEL];
float sc_cache[KSEL];
for (uint e = 0; e < KSEL; e++) {
    const uint p = srow * KSEL + e;
    const uint ex = IDX[p];
    ex_cache[e] = ex;
    sc_cache[e] = (SCALED ? SC[ex] : 1.0f) * WTS[p] * (FMT == 0 ? 4194304.0f : 1.0f);
}
if (FMT == 0) {
    for (uint idx = lane; idx < TOT; idx += 32) {
        const uint e = idx / NC;
        const uint c = idx % NC;
        const uint p = srow * KSEL + e;
        const uint ex = ex_cache[e];
        const float ws = sc_cache[e];
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(ACT + size_t(p) * I + c * 16);
        const vec<T, 4> v0 = xp[0], v1 = xp[1], v2 = xp[2], v3 = xp[3];
        const size_t base = size_t(ex) * D + row0;
        const device uint2* wrow = (const device uint2*)(WQ + base * WPR);
        const device uchar* scales = (const device uchar*)WS + base * SPR;
        for (uint r = 0; r < ROWS; r++) {
            const uint2 q = wrow[r * NC + c];
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
            acc[r] = fma(d * mlxserve_fp4_scale(uint(scales[r * SPR + c])), ws, acc[r]);
        }
    }
} else {
    for (uint idx = lane; idx < TOT; idx += 32) {
        const uint e = idx / NC;
        const uint c = idx % NC;
        const uint p = srow * KSEL + e;
        const uint ex = ex_cache[e];
        const float ws = sc_cache[e];
        const device vec<T, 4>* xp = (const device vec<T, 4>*)(ACT + size_t(p) * I + c * 16);
        const float4 v0 = float4(xp[0]), v1 = float4(xp[1]), v2 = float4(xp[2]), v3 = float4(xp[3]);
        float xr[16] = {
            v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
            v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w,
        };
        const size_t base = size_t(ex) * D + row0;
        if (FMT == 3) {
            const device uchar* arow = (const device uchar*)WQ + base * WBR;
            const device S* sa = (const device S*)WS + base * SPR;
            const device S* ba = (const device S*)WB + base * SPR;
            for (uint r = 0; r < ROWS; r++)
                acc[r] += msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, c, xr) * ws;
        } else {
            const device uint* wrow = WQ + base * WPR;
            const device uchar* scales = (const device uchar*)WS + base * SPR;
            for (uint r = 0; r < ROWS; r++)
                acc[r] += msv_ink2_chunk(FMT, wrow + r * WPR, scales + r * SPR, c, xr) * ws;
        }
    }
}
for (uint r = 0; r < ROWS; r++) {
    const float x = simd_sum(acc[r]);
    if (lane == r) {
        const uint col = row0 + r;
        const T xm = T(x);
        XM[size_t(srow) * D + col] = xm;
        if (TAIL) {
            float y = 0.0f;
            for (uint j = 0; j < KT; j++) y += ((j + 1 < KT) ? ST[j * SW + OFF + col] : float(xm)) * float(MW[col * KT + j]);
            y = float(WT(y));
            HOUT[col] = T(RES[col] + float(T((y + float(xm)) * PR[0])));
            for (uint j = 0; j + 1 < KT - 1; j++) NST[j * SW + OFF + col] = ST[(j + 1) * SW + OFF + col];
            NST[(KT - 2) * SW + OFF + col] = float(xm);
        }
    }
}
if (TAIL && srow == 0) {
    const uint gid = thread_position_in_grid.y * 32 + lane;
    const uint stride = (D / ROWS) * 32;
    const uint C = KP_shape[2];
    for (uint col = gid; col < OFF; col += stride) {
        for (uint j = 0; j + 1 < KT - 1; j++) NST[j * SW + col] = ST[(j + 1) * SW + col];
        float val;
        if (col < C) val = float(KP[col]);
        else if (col < 2 * C) val = float(VP[col - C]);
        else val = float(OP[col - 2 * C]);
        NST[(KT - 2) * SW + col] = val;
    }
}
