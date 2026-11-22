// Inkling 2: what the attention read needs from a visit's projections for S rows. grid (H * HD + 2 * HKV * HD + the
// tail, S), threadgroup (HD, 1): one thread per output element of a row, a head of q, k or v one threadgroup.
// Optimized: uses simd_sum + 4-slot shared reduction instead of 128-iter loop, and early-outs zero BIAS columns
// before computing DREL norms.
const uint i = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
constexpr uint NSG = (HD + 31) / 32;
const uint ROWS = QP_shape[1];
const uint QD = H * HD;
const uint C = HKV * HD;
const uint R = RJ_shape[1];
const uint SW = ST_shape[2];
const float qk_thr = PR[0], rel_thr = PR[1], eps = PR[2], tau = PR[3], cs = PR[4];
threadgroup float sbuf[NSG];
if (i < QD) {
    const uint at = row * QD + i;
    const float mine = float(QP[at]) * float(QS[i]);
    float sg_sum = simd_sum(mine * mine);
    if (lane == 0) sbuf[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float n2 = 0.0f;
    for (uint g = 0; g < NSG; g++) n2 += sbuf[g];
    const float f = min(qk_thr / (sqrt(n2) + 1e-6f), 1.0f);
    T q = T(mine * f);
    if (tau != 1.0f) q = T(float(q) * tau);
    Q[at] = q;
    GATE[at] = 1.0f / (1.0f + exp(-float(GP[at])));
    return;
}
uint r = i - QD;
if (r < 2 * C) {
    const bool is_v = r >= C;
    const uint c = is_v ? r - C : r;
    const uint h = c / HD;
    const uint off = is_v ? C : 0u;
    const float x = float(is_v ? VP[row * C + c] : KP[row * C + c]);
    float y = 0.0f;
    for (uint j = 0; j < KT; j++) {
        float wv = x;
        if (SEQ ? row + j < KT - 1 : j + 1 < KT) wv = ST[(SEQ ? row + j : row * (KT - 1) + j) * SW + off + c];
        else if (j + 1 < KT) wv = float(is_v ? VP[(row + j - (KT - 1)) * C + c] : KP[(row + j - (KT - 1)) * C + c]);
        float wt;
        if (DYN) {
            const uint lag = KT - 1 - j;
            const uint tap = (row * HKV + h) * KT + lag;
            wt = float(is_v ? VW[lag * C + c] : KW[lag * C + c]) + float(is_v ? VDY[tap] : KDY[tap]);
        } else {
            wt = float(is_v ? VW[c * KT + j] : KW[c * KT + j]);
        }
        y += wv * wt;
    }
    if (!DYN) y = float(WT(y));
    const float mine = float(T((y + x) * cs));
    float sg_sum = simd_sum(mine * mine);
    if (lane == 0) sbuf[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float n2 = 0.0f;
    for (uint g = 0; g < NSG; g++) n2 += sbuf[g];
    const uint at = (h * ROWS + row) * HD + c % HD;
    if (is_v) V[at] = T(mine * rsqrt(n2 / float(HD) + eps));
    else K[at] = T(mine * min(qk_thr / (sqrt(n2) + 1e-6f), 1.0f));
    return;
}
r -= 2 * C;
if (!WIDE) {
    const uint KV = uint(PR[5]);
    const uint col = r % KV;
    if (col + R < KV) {
        BIAS[r] = T(0.0f);
        return;
    }
    const uint h = r / KV;
    const uint rp = row * H * DREL + h * DREL;
    float n2 = 0.0f;
    for (uint j = 0; j < DREL; j++) {
        const float v = float(RP[rp + j]) * float(RS[h * DREL + j]);
        n2 += v * v;
    }
    const float f = min(rel_thr / (sqrt(n2) + 1e-6f), 1.0f);
    const uint d = col + R - KV;
    float acc = 0.0f;
    for (uint j = 0; j < DREL; j++) acc += float(T(float(RP[rp + j]) * float(RS[h * DREL + j]) * f)) * RJ[j * R + d];
    BIAS[r] = T(acc * tau);
    return;
}
const uint h = r / DREL;
const uint rp = row * H * DREL + h * DREL;
float n2 = 0.0f;
for (uint j = 0; j < DREL; j++) {
    const float v = float(RP[rp + j]) * float(RS[h * DREL + j]);
    n2 += v * v;
}
const float f = min(rel_thr / (sqrt(n2) + 1e-6f), 1.0f);
const uint j = r % DREL;
REL[rp + j] = T(float(RP[rp + j]) * float(RS[h * DREL + j]) * f);
