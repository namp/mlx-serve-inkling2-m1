// Merge NS splits + per-(row, head) sink + fused sigmoid output gate into [1, S, HQ * DV].
const uint e = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.y; // 0 .. S * HQ - 1
const auto mr = m_in + (long)row * NS;
const auto lr = l_in + (long)row * NS;
const float sk = float(sinks[row]);
float M = sk;
for (int i = 0; i < NS; ++i) M = max(M, mr[i]);
float L = metal::exp(sk - M), acc = 0.0f;
const auto orow = o_in + (long)row * NS * DV + e;
for (int i = 0; i < NS; ++i) {
  const float w = metal::exp(mr[i] - M);
  L += lr[i] * w;
  acc += orow[(long)i * DV] * w;
}
const float attn_val = float(T(acc / L));
out[(long)row * DV + e] = T(attn_val * gate[(long)row * DV + e]);
