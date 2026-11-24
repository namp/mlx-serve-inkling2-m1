// Inkling 2: J loop message gates over S rows, one thread per element.
// When TRANS == 1 and PROJ == 0, F and G are laid out [J, RANK, D] for 100% coalesced reads
// across threads in a threadgroup, reading each weight ONCE across SROWS rows (up to 16) with
// ST[SROWS * RANK] staged once in threadgroup SRAM.
// Y [J, S, D] f32 = (1 - tanh(F st)) A + tanh(G st) MS.
const uint i = thread_position_in_grid.x;
const uint row0 = thread_position_in_grid.y * SROWS;
const uint j = thread_position_in_grid.z;
const uint t = thread_position_in_threadgroup.x;
const uint S = ST_shape[1];
if (PROJ) {
    const uint e = (j * S + row0) * D + i;
    Y[e] = (1.0f - precise::tanh(F[e])) * A[e] + precise::tanh(G[e]) * MS[e];
} else if (TRANS) {
    threadgroup float st_tg[SROWS * RANK];
    for (uint k = t; k < SROWS * RANK; k += TG) {
        st_tg[k] = float(ST[(j * S + row0) * RANK + k]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device GT* fp = F + size_t(j * RANK) * D + i;
    const device GT* gp = G + size_t(j * RANK) * D + i;
    float f[SROWS], g[SROWS];
    for (uint r_i = 0; r_i < SROWS; r_i++) { f[r_i] = 0.0f; g[r_i] = 0.0f; }
    for (uint r = 0; r < RANK; r++) {
        const float fw = float(fp[size_t(r) * D]);
        const float gw = float(gp[size_t(r) * D]);
        for (uint r_i = 0; r_i < SROWS; r_i++) {
            const float s = st_tg[r_i * RANK + r];
            f[r_i] += fw * s;
            g[r_i] += gw * s;
        }
    }
    for (uint r_i = 0; r_i < SROWS; r_i++) {
        const uint e = (j * S + row0 + r_i) * D + i;
        Y[e] = (1.0f - precise::tanh(f[r_i])) * A[e] + precise::tanh(g[r_i]) * MS[e];
    }
} else {
    for (uint r_i = 0; r_i < SROWS; r_i++) {
        const uint row = row0 + r_i;
        float f = 0.0f, g = 0.0f;
        for (uint r = 0; r < RANK; r++) {
            const float s = float(ST[(j * S + row) * RANK + r]);
            f += float(F[(j * D + i) * RANK + r]) * s;
            g += float(G[(j * D + i) * RANK + r]) * s;
        }
        const uint e = (j * S + row) * D + i;
        Y[e] = (1.0f - precise::tanh(f)) * A[e] + precise::tanh(g) * MS[e];
    }
}
