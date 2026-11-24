// MLX's argmax order on (value, index): a NaN beats every number, the lower index among NaNs; else the larger value,
// the lower index among equal ones. A strict total order, so any reduction tree ends at MLX's argmax.
inline bool msv_ink2_better(float va, uint ia, float vb, uint ib) {
  if (isnan(va)) return !isnan(vb) || ia < ib;
  return !isnan(vb) && (va > vb || (va == vb && ia < ib));
}

// Deterministic Gumbel(0, 1) noise keyed by (seed, position q0, vocab id v) with 23 uniform bits strictly inside (0, 1).
// Uses 23 bits (x >> 9) so float(hi) + 0.5f is exact in IEEE-754 float32 (max 8388607.5 * 2^-23 = 1 - 2^-24 < 1.0f,
// avoiding the 24-bit round-to-1.0f -> +Inf bug). Shared between markov_fold and gumbel_picks.
inline float msv_ink2_gumbel(uint seed, uint q0, uint v) {
  uint key = ((seed ^ (q0 * 0x85EBCA6Bu)) * 0xC2B2AE35u) + 0x27D4EB2Fu;
  uint x = v * 0x9E3779B1u + key;
  x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
  float u = (float(x >> 9) + 0.5f) * 1.1920928955078125e-07f;
  return -metal::precise::log(-metal::precise::log(u));
}
