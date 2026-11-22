#define UNR _Pragma("clang loop unroll(full)")
// Inkling 2 verify (S = 2..5 rows) directly over Q4/Q8 KV cache + S tiny FP16 candidate rows (k_new, v_new).
// Zero KV dequantization, zero concat copy, reads Q4 KV ONCE for all S * GQA query rows simultaneously!
// q: [1, HQ, S, DK] (non-contiguous ok), kq/vq: [1, HKV, Tk_old, DK/VPW] u32,
// k_new/v_new: [1, HKV, S, DK] (T), rel: [1, S, HQ, R] (T).
constexpr int VPW = 32 / BITS;
constexpr int M = SROWS * GQA;
constexpr int NHT = (M + 7) / 8;
constexpr int NDK = DK / 8;
constexpr int NDV = DV / 8;
constexpr int NSG = 4;
constexpr int QTG = NHT * 8 * DK;
constexpr int TGF = QTG > 8 * DV ? QTG : 8 * DV;
const uint tid = thread_position_in_threadgroup.x;
const ushort lane = ushort(tid % 32);
const uint sg = tid / 32;
const uint hkv = threadgroup_position_in_grid.y;
const uint split = threadgroup_position_in_grid.z;
const uint nsplit = threadgroups_per_grid.z;
const int Tk = tk[0];
const float scale = scl[0];
const uint mask_bits = (1u << BITS) - 1u;
const int per = Tk > 0 ? (((Tk + int(nsplit * NSG) - 1) / int(nsplit * NSG) + 7) / 8 * 8) : 0;
const int s0 = int(split * NSG + sg) * per;
const int s1 = min(Tk, s0 + per);

const ushort qid = lane / 4;
const ushort fm = (qid & 4) + ((lane / 2) % 4);
const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;

threadgroup float tg[TGF];
threadgroup float ml_tg[2 * NSG * NHT * 8];
for (int i = int(tid); i < QTG; i += NSG * 32) {
  const int idx = i / DK, e = i % DK;
  if (idx < M) {
    const int r = idx / GQA;
    const int gi = idx % GQA;
    const long hq = (long)hkv * GQA + gi;
    tg[i] = float(q[hq * (long)q_strides[1] + (long)r * (long)q_strides[2] + e]) * scale;
  } else {
    tg[i] = 0.0f;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_float8x8 O[NHT][NDV];
float m[NHT], mt[NHT], l[NHT];
UNR for (int h = 0; h < NHT; ++h) {
  m[h] = -1e30f; mt[h] = -1e30f; l[h] = 0.0f;
  UNR for (int j = 0; j < NDV; ++j) O[h][j] = simdgroup_float8x8(0.0f);
}

// 1) Committed Q4/Q8 keys [s0, s1)
if (s0 < s1) {
  const device uint* kqh = kq + (long)hkv * (long)kq_strides[1];
  const device T* ksh = ksc + (long)hkv * (long)ksc_strides[1];
  const device T* kbh = kbi + (long)hkv * (long)kbi_strides[1];
  const device uint* vqh = vq + (long)hkv * (long)vq_strides[1];
  const device T* vsh = vsc + (long)hkv * (long)vsc_strides[1];
  const device T* vbh = vbi + (long)hkv * (long)vbi_strides[1];
  const long kqs = kq_strides[2], kss = ksc_strides[2], kbs = kbi_strides[2];
  const long vqs = vq_strides[2], vss = vsc_strides[2], vbs = vbi_strides[2];

  for (int t0 = s0; t0 < s1; t0 += 8) {
    const int ra = t0 + fn, rb = ra + 1;
    const bool a_in = ra < s1, b_in = rb < s1;
    const long rka = (long)min(ra, Tk - 1), rkb = (long)min(rb, Tk - 1);
    simdgroup_float8x8 S[NHT];
    UNR for (int h = 0; h < NHT; ++h) S[h] = simdgroup_float8x8(0.0f);
    {
      const device uint4* ka4 = (const device uint4*)(kqh + rka * kqs);
      const device uint4* kb4 = (const device uint4*)(kqh + rkb * kqs);
      float ksa[DK / GS], kba[DK / GS], ksb[DK / GS], kbb[DK / GS];
      UNR for (int g = 0; g < DK / GS; ++g) {
        ksa[g] = float(ksh[rka * kss + g]); kba[g] = float(kbh[rka * kbs + g]);
        ksb[g] = float(ksh[rkb * kss + g]); kbb[g] = float(kbh[rkb * kbs + g]);
      }
      UNR for (int w4 = 0; w4 < DK / VPW / 4; ++w4) {
        const uint4 WA = ka4[w4], WB = kb4[w4];
        UNR for (int u = 0; u < 4 * VPW / 8; ++u) {
          const int j = w4 * (4 * VPW / 8) + u;
          const int d = 8 * j + fm;
          const int wl = (d / VPW) % 4;
          const int sh = (d % VPW) * BITS;
          const int g = (8 * j) / GS;
          simdgroup_float8x8 kt;
          kt.thread_elements()[0] = float((WA[wl] >> sh) & mask_bits) * ksa[g] + kba[g];
          kt.thread_elements()[1] = float((WB[wl] >> sh) & mask_bits) * ksb[g] + kbb[g];
          UNR for (int h = 0; h < NHT; ++h) {
            simdgroup_float8x8 qa;
            simdgroup_load(qa, tg + 8 * h * DK + 8 * j, DK);
            simdgroup_multiply_accumulate(S[h], qa, kt, S[h]);
          }
        }
      }
    }
    simdgroup_float8x8 P[NHT];
    UNR for (int h = 0; h < NHT; ++h) {
      const int idx = int(fm) + 8 * h;
      const int r = idx / GQA;
      const int gi = idx % GQA;
      const long hq = (long)hkv * GQA + gi;
      const int da = Tk + r - ra;
      const int db = Tk + r - rb;
      const bool a_ok = (idx < M) && a_in && (WINDOW == 0 || da < WINDOW);
      const bool b_ok = (idx < M) && b_in && (WINDOW == 0 || db < WINDOW);
      float x0 = -INFINITY, x1 = -INFINITY;
      if (a_ok) {
        x0 = S[h].thread_elements()[0];
        if (da < R) x0 += float(rel[((long)r * HQ + hq) * R + da]);
      }
      if (b_ok) {
        x1 = S[h].thread_elements()[1];
        if (db < R) x1 += float(rel[((long)r * HQ + hq) * R + db]);
      }
      float mx = max(x0, x1);
      mx = max(mx, simd_shuffle_xor(mx, 1));
      mx = max(mx, simd_shuffle_xor(mx, 8));
      mt[h] = max(mt[h], mx);
      if (simd_any(mx > m[h] + 8.0f)) {
        const float mn = max(m[h], mx);
        const float c = metal::exp(m[h] - mn);
        l[h] *= c; m[h] = mn;
        UNR for (int j = 0; j < NDV; ++j) { O[h][j].thread_elements()[0] *= c; O[h][j].thread_elements()[1] *= c; }
      }
      const float p0 = metal::exp(x0 - m[h]), p1 = metal::exp(x1 - m[h]);
      float ps = p0 + p1;
      ps += simd_shuffle_xor(ps, 1);
      ps += simd_shuffle_xor(ps, 8);
      l[h] += ps;
      P[h].thread_elements()[0] = p0; P[h].thread_elements()[1] = p1;
    }
    {
      const int rv = t0 + fm;
      const long rvc = (long)min(rv, Tk - 1);
      const bool rv_ok = rv < s1;
      const device uint4* v4p = (const device uint4*)(vqh + rvc * vqs);
      float svs[DV / GS], bvs[DV / GS];
      UNR for (int g = 0; g < DV / GS; ++g) {
        svs[g] = rv_ok ? float(vsh[rvc * vss + g]) : 0.0f;
        bvs[g] = rv_ok ? float(vbh[rvc * vbs + g]) : 0.0f;
      }
      UNR for (int w4 = 0; w4 < DV / VPW / 4; ++w4) {
        const uint4 W = v4p[w4];
        UNR for (int u = 0; u < 4 * VPW / 8; ++u) {
          const int j = w4 * (4 * VPW / 8) + u;
          const int e = 8 * j + fn;
          const int wl = (e / VPW) % 4;
          const int sh = (e % VPW) * BITS;
          const int g = (8 * j) / GS;
          simdgroup_float8x8 vt;
          vt.thread_elements()[0] = float((W[wl] >> sh) & mask_bits) * svs[g] + bvs[g];
          vt.thread_elements()[1] = float((W[wl] >> (sh + BITS)) & mask_bits) * svs[g] + bvs[g];
          UNR for (int h = 0; h < NHT; ++h) simdgroup_multiply_accumulate(O[h][j], P[h], vt, O[h][j]);
        }
      }
    }
  }
}

// 2) Candidate keys k_new / v_new (SROWS <= 8): one tile on split == 0, sg == 0.
if (split == 0 && sg == 0) {
  const device T* knh = k_new + (long)hkv * (long)k_new_strides[1];
  const device T* vnh = v_new + (long)hkv * (long)v_new_strides[1];
  const long kns = k_new_strides[2], vns = v_new_strides[2];
  const int ca = fn, cb = fn + 1;
  const bool ca_in = ca < SROWS, cb_in = cb < SROWS;
  const long rca = (long)min(ca, SROWS - 1), rcb = (long)min(cb, SROWS - 1);
  simdgroup_float8x8 S[NHT];
  UNR for (int h = 0; h < NHT; ++h) S[h] = simdgroup_float8x8(0.0f);
  UNR for (int j = 0; j < NDK; ++j) {
    const int d = 8 * j + fm;
    simdgroup_float8x8 kt;
    kt.thread_elements()[0] = ca_in ? float(knh[rca * kns + d]) : 0.0f;
    kt.thread_elements()[1] = cb_in ? float(knh[rcb * kns + d]) : 0.0f;
    UNR for (int h = 0; h < NHT; ++h) {
      simdgroup_float8x8 qa;
      simdgroup_load(qa, tg + 8 * h * DK + 8 * j, DK);
      simdgroup_multiply_accumulate(S[h], qa, kt, S[h]);
    }
  }
  simdgroup_float8x8 P[NHT];
  UNR for (int h = 0; h < NHT; ++h) {
    const int idx = int(fm) + 8 * h;
    const int r = idx / GQA;
    const int gi = idx % GQA;
    const long hq = (long)hkv * GQA + gi;
    const bool a_ok = (idx < M) && ca_in && (ca <= r);
    const bool b_ok = (idx < M) && cb_in && (cb <= r);
    float x0 = -INFINITY, x1 = -INFINITY;
    if (a_ok) {
      x0 = S[h].thread_elements()[0];
      const int da = r - ca;
      if (da < R) x0 += float(rel[((long)r * HQ + hq) * R + da]);
    }
    if (b_ok) {
      x1 = S[h].thread_elements()[1];
      const int db = r - cb;
      if (db < R) x1 += float(rel[((long)r * HQ + hq) * R + db]);
    }
    float mx = max(x0, x1);
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    mt[h] = max(mt[h], mx);
    if (simd_any(mx > m[h] + 8.0f)) {
      const float mn = max(m[h], mx);
      const float c = metal::exp(m[h] - mn);
      l[h] *= c; m[h] = mn;
      UNR for (int j = 0; j < NDV; ++j) { O[h][j].thread_elements()[0] *= c; O[h][j].thread_elements()[1] *= c; }
    }
    const float p0 = metal::exp(x0 - m[h]), p1 = metal::exp(x1 - m[h]);
    float ps = p0 + p1;
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l[h] += ps;
    P[h].thread_elements()[0] = p0; P[h].thread_elements()[1] = p1;
  }
  const int rv = fm;
  const bool rv_ok = rv < SROWS;
  const long rvc = (long)min(rv, SROWS - 1);
  UNR for (int j = 0; j < NDV; ++j) {
    const int e = 8 * j + fn;
    simdgroup_float8x8 vt;
    vt.thread_elements()[0] = rv_ok ? float(vnh[rvc * vns + e]) : 0.0f;
    vt.thread_elements()[1] = rv_ok ? float(vnh[rvc * vns + e + 1]) : 0.0f;
    UNR for (int h = 0; h < NHT; ++h) simdgroup_multiply_accumulate(O[h][j], P[h], vt, O[h][j]);
  }
}

// 3) Combine 4 simdgroups in threadgroup
if (fn == 0 && (lane & 8) == 0) {
  UNR for (int h = 0; h < NHT; ++h) { ml_tg[(sg * NHT + h) * 8 + fm] = mt[h]; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float M_f[NHT], L_f[NHT];
UNR for (int h = 0; h < NHT; ++h) {
  float mm = -1e30f;
  UNR for (int k = 0; k < NSG; ++k) mm = max(mm, ml_tg[(k * NHT + h) * 8 + fm]);
  M_f[h] = mm;
  const float c = metal::exp(m[h] - mm);
  l[h] *= c;
  UNR for (int j = 0; j < NDV; ++j) { O[h][j].thread_elements()[0] *= c; O[h][j].thread_elements()[1] *= c; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (fn == 0 && (lane & 8) == 0) {
  UNR for (int h = 0; h < NHT; ++h) ml_tg[((NSG + sg) * NHT + h) * 8 + fm] = l[h];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
UNR for (int h = 0; h < NHT; ++h) {
  float ll = 0.0f;
  UNR for (int k = 0; k < NSG; ++k) ll += ml_tg[((NSG + k) * NHT + h) * 8 + fm];
  L_f[h] = ll;
  for (uint k = 1; k < NSG; ++k) {
    if (sg == k) { UNR for (int j = 0; j < NDV; ++j) simdgroup_store(O[h][j], tg + 8 * j, DV); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      UNR for (int j = 0; j < NDV; ++j) {
        simdgroup_float8x8 t;
        simdgroup_load(t, tg + 8 * j, DV);
        O[h][j].thread_elements()[0] += t.thread_elements()[0];
        O[h][j].thread_elements()[1] += t.thread_elements()[1];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
if (sg == 0) {
  UNR for (int h = 0; h < NHT; ++h) {
    const int idx = int(fm) + 8 * h;
    if (idx < M) {
      const int r = idx / GQA;
      const int gi = idx % GQA;
      const long out_row = (long)r * HQ + (long)hkv * GQA + gi;
      const long slot = out_row * (long)nsplit + (long)split;
      if (fn == 0) { m_out[slot] = M_f[h]; l_out[slot] = L_f[h]; }
      UNR for (int j = 0; j < NDV; ++j) {
        o_out[slot * DV + 8 * j + fn] = O[h][j].thread_elements()[0];
        o_out[slot * DV + 8 * j + fn + 1] = O[h][j].thread_elements()[1];
      }
    }
  }
}
