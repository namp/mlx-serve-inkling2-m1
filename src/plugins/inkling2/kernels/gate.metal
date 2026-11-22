// Inkling 2: J loop message gates over S rows, one thread per element.
// When TRANS == 1 and PROJ == 0, F and G are laid out [J, RANK, D] for 100% coalesced reads
// across threads in a threadgroup, with ST[RANK] staged once in threadgroup SRAM.
// Y [J, S, D] f32 = (1 - tanh(F st)) A + tanh(G st) MS.
const uint i = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint j = thread_position_in_grid.z;
const uint t = thread_position_in_threadgroup.x;
const uint S = ST_shape[1];
const uint e = (j * S + row) * D + i;
float f = 0.0f, g = 0.0f;
if (PROJ) {
    f = F[e];
    g = G[e];
} else if (TRANS) {
    threadgroup float st_tg[RANK];
    for (uint r = t; r < RANK; r += TG) {
        st_tg[r] = float(ST[(j * S + row) * RANK + r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device GT* fp = F + size_t(j * RANK) * D + i;
    const device GT* gp = G + size_t(j * RANK) * D + i;
    for (uint r = 0; r < RANK; r++) {
        const float s = st_tg[r];
        f += float(fp[size_t(r) * D]) * s;
        g += float(gp[size_t(r) * D]) * s;
    }
} else {
    for (uint r = 0; r < RANK; r++) {
        const float s = float(ST[(j * S + row) * RANK + r]);
        f += float(F[(j * D + i) * RANK + r]) * s;
        g += float(G[(j * D + i) * RANK + r]) * s;
    }
}
// The precise tanh: the fast one is NaN at an infinity and a percent off near zero.
Y[e] = (1.0f - precise::tanh(f)) * A[e] + precise::tanh(g) * MS[e];
