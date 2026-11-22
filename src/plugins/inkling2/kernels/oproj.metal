// Inkling 2, one row: the attention read ATT [H * HD] (T) times its output gate GATE [H * HD] f32, rounded to T as
// `gated` leaves it, then the output projection Y [1, 1, N] (T): an affine weight (FMT 3: BITS-bit rows of WQ in MLX's
// packing, group GS scales WS and biases WB of S) or a dense [N, K] weight of T (FMT 4). grid (32, N / ROWS, 1),
// threadgroup (32, SGS, 1): a SIMD group owns ROWS output rows, reusing xr[16] across them.
const uint lane = thread_index_in_simdgroup;
const uint n0 = thread_position_in_grid.y * ROWS;
constexpr uint WBR = K * BITS / 8;
constexpr uint SPR = K / GS;
const device uchar* arow = (const device uchar*)WQ + size_t(n0) * WBR;
const device S* sa = (const device S*)WS + size_t(n0) * SPR;
const device S* ba = (const device S*)WB + size_t(n0) * SPR;
const device T* drow = (const device T*)WQ + size_t(n0) * K;
float acc[ROWS];
for (uint r = 0; r < ROWS; r++) acc[r] = 0.0f;
for (uint c = lane; c < K / 16; c += 32) {
  float xr[16];
  const device vec<T, 4>* ap = (const device vec<T, 4>*)(ATT + c * 16);
  const device float4* gp = (const device float4*)(GATE + c * 16);
  for (uint v = 0; v < 4; v++) {
    const vec<T, 4> a4 = ap[v];
    const float4 g4 = gp[v];
    xr[4 * v + 0] = float(T(float(a4[0]) * g4.x));
    xr[4 * v + 1] = float(T(float(a4[1]) * g4.y));
    xr[4 * v + 2] = float(T(float(a4[2]) * g4.z));
    xr[4 * v + 3] = float(T(float(a4[3]) * g4.w));
  }
  for (uint r = 0; r < ROWS; r++) {
    if (FMT == 3) {
      acc[r] += msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, c, xr);
    } else {
      const device vec<T, 4>* dp = (const device vec<T, 4>*)(drow + r * K + c * 16);
      float d = 0.0f;
      for (uint v = 0; v < 4; v++) {
        const vec<T, 4> w4 = dp[v];
        d += float(w4[0]) * xr[4 * v + 0] + float(w4[1]) * xr[4 * v + 1] + float(w4[2]) * xr[4 * v + 2] + float(w4[3]) * xr[4 * v + 3];
      }
      acc[r] += d;
    }
  }
}
for (uint r = 0; r < ROWS; r++) {
  const float v = simd_sum(acc[r]);
  if (lane == r) Y[n0 + r] = T(v);
}
