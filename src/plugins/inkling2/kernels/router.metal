// Inkling 2, a few rows: the router's logits SC [S, E] f32, the MLP's normed input X [1, S, D] (T) over the
// router's rows RT [E, D] f32. MLX's f32 matmul at two rows and more is several times a one-row launch's latency.
// grid (32 * E, S / SROWS), threadgroup (32, 1): a SIMD group reads RT[ex] ONCE across SROWS rows (up to 16),
// cutting router DRAM reads by up to SROWSx on verify passes.
const uint lane = thread_index_in_simdgroup;
const uint ex = thread_position_in_grid.x / 32;
const uint row0 = thread_position_in_grid.y * SROWS;
float dot[SROWS];
for (uint r = 0; r < SROWS; r++) dot[r] = 0.0f;
for (uint i = lane; i < D; i += 32) {
    const float w = RT[ex * D + i];
    for (uint r = 0; r < SROWS; r++) dot[r] += float(X[(row0 + r) * D + i]) * w;
}
for (uint r = 0; r < SROWS; r++) {
    const float d = simd_sum(dot[r]);
    if (lane == 0) SC[(row0 + r) * E + ex] = d;
}
