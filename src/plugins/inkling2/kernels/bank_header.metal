// Chunk `c` (16 values) of one packed expert row against x, as a float. FMT 0 = NVFP4 (e2m1 nibbles, an e4m3 scale
// per 16), 1 = MXFP4 (an e8m0 scale per 32), 2 = MXFP8 (e4m3 bytes, an e8m0 scale per 32). `row` is the row's
// words, `srow` its scales.
inline float msv_ink2_chunk(int fmt, const device uint* row, const device uchar* srow, uint c, thread const float* x) {
  if (fmt == 2) {
    const uint4 q = ((const device uint4*)row)[c];
    const uint words[4] = {q.x, q.y, q.z, q.w};
    float d = 0.0f;
    for (int i = 0; i < 4; ++i) {
      const float2 lo = mlxserve_e4m3x2(words[i]);
      const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
      d += x[4 * i] * lo.x + x[4 * i + 1] * hi.x + x[4 * i + 2] * lo.y + x[4 * i + 3] * hi.y;
    }
    const uint e = uint(srow[c / 2]);
    // The byte decode leaves 2^8 out.
    return d * as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
  }
  const float d = mlxserve_fp4_dot16(((const device uint2*)row)[c], x);
  const uint b = uint(srow[fmt == 0 ? c : c / 2]);
  const float s = fmt == 0 ? mlxserve_fp4_scale(b) : as_type<float>(b == 0u ? 0x400000u : (b << 23)) * 0.00390625f;
  // The nibble and scale decodes leave 2^14 and 2^8 out.
  return d * s * 4194304.0f;
}

// Chunk `c` (16 values) of one affine row against x: scale * sum(q x) + bias * sum(x) over the chunk's group.
// Fast paths for 4-bit (uint2) and 8-bit (uint4) avoid 64-bit shift loops on Apple GPUs.
template <typename S, int BITS, int GS>
inline float msv_ink2_affine_chunk(const device uchar* row, const device S* srow, const device S* brow, uint c, thread const float* x) {
  const uint g = (c * 16) / GS;
  float qx = 0.0f, sx = 0.0f;
  if (BITS == 4) {
    const device uint* p = (const device uint*)(row + c * 8);
    const uint w0 = p[0];
    const uint w1 = p[1];
    qx += float(w0 & 15u) * x[0] + float((w0 >> 4) & 15u) * x[1] + float((w0 >> 8) & 15u) * x[2] + float((w0 >> 12) & 15u) * x[3];
    qx += float((w0 >> 16) & 15u) * x[4] + float((w0 >> 20) & 15u) * x[5] + float((w0 >> 24) & 15u) * x[6] + float(w0 >> 28) * x[7];
    qx += float(w1 & 15u) * x[8] + float((w1 >> 4) & 15u) * x[9] + float((w1 >> 8) & 15u) * x[10] + float((w1 >> 12) & 15u) * x[11];
    qx += float((w1 >> 16) & 15u) * x[12] + float((w1 >> 20) & 15u) * x[13] + float((w1 >> 24) & 15u) * x[14] + float(w1 >> 28) * x[15];
    sx = (x[0] + x[1] + x[2] + x[3]) + (x[4] + x[5] + x[6] + x[7]) + (x[8] + x[9] + x[10] + x[11]) + (x[12] + x[13] + x[14] + x[15]);
    return float(srow[g]) * qx + float(brow[g]) * sx;
  }
  if (BITS == 8) {
    const device uint* p = (const device uint*)(row + c * 16);
    const uint w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3];
    qx += float(w0 & 255u) * x[0] + float((w0 >> 8) & 255u) * x[1] + float((w0 >> 16) & 255u) * x[2] + float(w0 >> 24) * x[3];
    qx += float(w1 & 255u) * x[4] + float((w1 >> 8) & 255u) * x[5] + float((w1 >> 16) & 255u) * x[6] + float(w1 >> 24) * x[7];
    qx += float(w2 & 255u) * x[8] + float((w2 >> 8) & 255u) * x[9] + float((w2 >> 16) & 255u) * x[10] + float(w2 >> 24) * x[11];
    qx += float(w3 & 255u) * x[12] + float((w3 >> 8) & 255u) * x[13] + float((w3 >> 16) & 255u) * x[14] + float(w3 >> 24) * x[15];
    sx = (x[0] + x[1] + x[2] + x[3]) + (x[4] + x[5] + x[6] + x[7]) + (x[8] + x[9] + x[10] + x[11]) + (x[12] + x[13] + x[14] + x[15]);
    return float(srow[g]) * qx + float(brow[g]) * sx;
  }
  const device ushort* p = (const device ushort*)(row + c * 2 * BITS);
  ulong lo = 0, hi = 0;
  for (int i = 0; i < BITS; ++i) {
    const ulong v = ulong(p[i]);
    if (16 * i < 64) lo |= v << (16 * i); else hi |= v << (16 * i - 64);
  }
  for (int i = 0; i < 16; ++i) {
    const int off = i * BITS;
    const ulong w = off + BITS <= 64 ? lo >> off : (off >= 64 ? hi >> (off - 64) : ((lo >> off) | (hi << (64 - off))));
    qx += float(w & ((1ul << BITS) - 1ul)) * x[i];
    sx += x[i];
  }
  return float(srow[g]) * qx + float(brow[g]) * sx;
}
