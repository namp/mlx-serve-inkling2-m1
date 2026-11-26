// Inkling 2 prefill native FP16 NVFP4 expert-grouped GEMM for non-NAX Apple GPUs (M1/M2/M3/M4).
// Replaces BF16-promoted gather_qmm on S >= 64 chunks: each 128-thread threadgroup owns one (expert, 16-pair tile,
// 64-output tile). Swapped grid (item on grid.y, output tile on grid.z) keeps each expert's 64-row weight slice hot
// in L2 across consecutive tiles of the same expert. Per BK=32 K-step, 128 threads load 128 half4 X-chunks + 128
// NVFP4 16-weight groups (1:1 thread mapping), dequantizing e2m1 * fp8_e4m3 EXACTLY into threadgroup half Ws
// ([9.77e-4, 2688.0], <= 7 mantissa bits), accumulating in simdgroup_float8x8 and applying SC[e] in f32 before
// storing T (or fusing soft-capped SwiGLU directly from each lane's even/odd (gate, up) register pair when SWIGLU=1).
// grid (128, max_tiles, ceil(N_SIZE / 64)), threadgroup (128, 1, 1).
constexpr int BM = 16;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int WN = 4;
constexpr int TN = BN / WN;
constexpr int FM = BM / 8;
constexpr int FN = TN / 8;
constexpr int LD = BK + 8;
constexpr int WPR = K_SIZE / 8;
constexpr int SPR = K_SIZE / 16;
constexpr int OUT_N = SWIGLU ? (N_SIZE / 2) : N_SIZE;

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = sg * 32u + lane;
const int z = int(threadgroup_position_in_grid.y);
if (z >= int(ITEM_START[NE])) return;

// Binary search for expert e with ITEM_START[e] <= z < ITEM_START[e + 1].
int lo = 0, hi = NE;
while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (int(ITEM_START[mid]) <= z) lo = mid; else hi = mid;
}
const int e = lo;
const int first = int(SEG_START[e]) + (z - int(ITEM_START[e])) * BM;
const int count = min(BM, int(SEG_START[e + 1]) - first);
const int n0 = int(threadgroup_position_in_grid.z) * BN;

threadgroup half Xs[BM * LD];
threadgroup half Ws[BN * LD];
threadgroup uint rows_x[BM];
threadgroup uint rows_p[BM];

if (tid < BM) {
    const uint p = ORDER[first + min(int(tid), count - 1)];
    rows_p[tid] = p;
    rows_x[tid] = p / XDIV;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const int xr_row = int(tid >> 3);
const int xr_c4 = int(tid & 7u);
const int wr_row = int(tid >> 1);
const int wr_g = int(tid & 1u);
const bool w_ok = (n0 + wr_row) < N_SIZE;
const size_t w_abs = size_t(e) * N_SIZE + size_t(n0 + wr_row);
const device uint2* w_ptr = (const device uint2*)(WQ + w_abs * WPR);
const auto s_ptr = WS + w_abs * SPR;
const device vec<T, 4>* x_ptr4 = (const device vec<T, 4>*)(X + size_t(rows_x[xr_row]) * K_SIZE);

simdgroup_float8x8 acc[FM][FN];
#pragma clang loop unroll(full)
for (int i = 0; i < FM; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < FN; ++j) {
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
}

vec<T, 4> x_reg;
uint2 w_reg = uint2(0u, 0u);
uchar s_reg = 0u;

auto prefetch = [&](int k0) {
    x_reg = x_ptr4[(k0 >> 2) + xr_c4];
    if (w_ok) {
        const int g = (k0 >> 4) + wr_g;
        w_reg = w_ptr[g];
        s_reg = s_ptr[g];
    }
};

prefetch(0);

for (int k0 = 0; k0 < K_SIZE; k0 += BK) {
    *(threadgroup half4*)(Xs + xr_row * LD + xr_c4 * 4) = half4(x_reg[0], x_reg[1], x_reg[2], x_reg[3]);

    threadgroup half4* wd4 = (threadgroup half4*)(Ws + wr_row * LD + wr_g * 16);
    if (w_ok) {
        const float sv = mlxserve_fp4_scale(uint(s_reg)) * 4194304.0f;
        const float2 p0 = mlxserve_e2m1x2(w_reg.x, 0);
        const float2 p1 = mlxserve_e2m1x2(w_reg.x, 1);
        const float2 p2 = mlxserve_e2m1x2(w_reg.x, 2);
        const float2 p3 = mlxserve_e2m1x2(w_reg.x, 3);
        const float2 q0 = mlxserve_e2m1x2(w_reg.y, 0);
        const float2 q1 = mlxserve_e2m1x2(w_reg.y, 1);
        const float2 q2 = mlxserve_e2m1x2(w_reg.y, 2);
        const float2 q3 = mlxserve_e2m1x2(w_reg.y, 3);
        wd4[0] = half4(float4(p0.x, p1.x, p2.x, p3.x) * sv);
        wd4[1] = half4(float4(p0.y, p1.y, p2.y, p3.y) * sv);
        wd4[2] = half4(float4(q0.x, q1.x, q2.x, q3.x) * sv);
        wd4[3] = half4(float4(q0.y, q1.y, q2.y, q3.y) * sv);
    } else {
        #pragma clang loop unroll(full)
        for (int u = 0; u < 4; ++u) wd4[u] = half4(0.0h);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k0 + BK < K_SIZE) prefetch(k0 + BK);

    #pragma clang loop unroll(full)
    for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_half8x8 a[FM];
        simdgroup_half8x8 b[FN];
        #pragma clang loop unroll(full)
        for (int i = 0; i < FM; ++i) {
            simdgroup_load(a[i], Xs + i * 8 * LD + kk, LD);
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < FN; ++j) {
            simdgroup_load(b[j], Ws + (int(sg) * TN + j * 8) * LD + kk, LD, ulong2(0), true);
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < FM; ++i) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < FN; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
    }

    if (k0 + BK < K_SIZE) threadgroup_barrier(mem_flags::mem_threadgroup);
}

const float sc = SCALED ? SC[e] : 1.0f;
#pragma clang loop unroll(full)
for (int i = 0; i < FM; ++i) {
    const int r = i * 8 + int(fm);
    if (r < count) {
        device T* dst = OUT + size_t(rows_p[r]) * OUT_N;
        #pragma clang loop unroll(full)
        for (int j = 0; j < FN; ++j) {
            const int col = n0 + int(sg) * TN + j * 8 + int(fn);
            if constexpr (SWIGLU != 0) {
                if (col + 1 < N_SIZE) {
                    const float g = acc[i][j].thread_elements()[0] * sc;
                    const float u = acc[i][j].thread_elements()[1] * sc;
                    const float capped = (precise::tanh(g * PR[0]) * PR[1]) * (precise::tanh(u * PR[2]) * PR[3]);
                    dst[col >> 1] = T(capped / (1.0f + precise::exp(-g)));
                }
            } else {
                if (col < N_SIZE) dst[col] = T(acc[i][j].thread_elements()[0] * sc);
                if (col + 1 < N_SIZE) dst[col + 1] = T(acc[i][j].thread_elements()[1] * sc);
            }
        }
    }
}
