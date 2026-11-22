// Inkling 2, barrier-free down projections and weighted sum across all KSEL experts in one launch; with TAIL
// (one row) the MLP's short conv and residual add too. grid (32, D / ROWS, S), threadgroup (32, BG, 1):
// each SIMD group owns ROWS output columns of one token row and sweeps all KSEL * (I / 16) chunks with zero
// threadgroup barriers and zero idle SIMD lanes (8 * 72 = 576 = 18 * 32 on Inkling-2-Mini).
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
    sc_cache[e] = (SCALED ? SC[ex] : 1.0f) * WTS[p];
}
for (uint idx = lane; idx < TOT; idx += 32) {
    const uint e = idx / NC;
    const uint c = idx % NC;
    const uint p = srow * KSEL + e;
    const uint ex = ex_cache[e];
    const float ws = sc_cache[e];
    const device vec<T, 4>* xp = (const device vec<T, 4>*)(ACT + size_t(p) * I + c * 16);
    float xr[16];
    for (uint v = 0; v < 4; v++) {
        const vec<T, 4> x4 = xp[v];
        xr[4 * v + 0] = float(x4[0]);
        xr[4 * v + 1] = float(x4[1]);
        xr[4 * v + 2] = float(x4[2]);
        xr[4 * v + 3] = float(x4[3]);
    }
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
        }
    }
}
