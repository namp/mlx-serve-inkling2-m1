// Inkling 2 LoopCD thinking-block state tracker (Exclude Thinking Blocks mode, arXiv:2610.02185).
// Tracks whether each position in IDS [B, S] is inside <|content_thinking|> (PR[1], 200008) vs closed by any other
// control token >= PR[0] (199998), starting from BEFORE [B, 1] f32. Writes OUT [B, S] f32 (0.0 or 1.0).
// TREE == 0: 256-thread parallel prefix scan per slot (grid (256, B, 1), tg (256, 1, 1)).
// TREE == 1: 1-thread topological scan over verify tree using PARENT [B, S] (grid (1, B, 1), tg (1, 1, 1)).
const uint b = threadgroup_position_in_grid.y;
const int S = IDS_shape[1];
const int first_ctrl = PR[0];
const int think_tok = PR[1];
const int init_st = (BEFORE[b] > 0.5f) ? 1 : 0;

if constexpr (TREE != 0) {
    int st_arr[32];
    for (int r = 0; r < S; ++r) {
        const int p = PARENT[size_t(b) * S + r];
        int cur = (p < 0) ? init_st : st_arr[p];
        const int tok = IDS[size_t(b) * S + r];
        if (tok >= first_ctrl) cur = (tok == think_tok) ? 1 : 0;
        st_arr[r] = cur;
        OUT[size_t(b) * S + r] = float(cur);
    }
} else {
    const uint tid = thread_position_in_threadgroup.x;
    const int chunk = (S + 255) / 256;
    const int lo = int(tid) * chunk;
    const int hi = min(S, lo + chunk);
    int last_pos = -1;
    int last_val = 0;
    for (int t = lo; t < hi; ++t) {
        const int tok = IDS[size_t(b) * S + t];
        if (tok >= first_ctrl) {
            last_pos = t;
            last_val = (tok == think_tok) ? 1 : 0;
        }
    }
    threadgroup int c_pos[256];
    threadgroup int c_val[256];
    threadgroup int prefix_state[256];
    c_pos[tid] = last_pos;
    c_val[tid] = last_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        int cur = init_st;
        for (int c = 0; c < 256; ++c) {
            prefix_state[c] = cur;
            if (c_pos[c] >= 0) cur = c_val[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int st = prefix_state[tid];
    for (int t = lo; t < hi; ++t) {
        const int tok = IDS[size_t(b) * S + t];
        if (tok >= first_ctrl) st = (tok == think_tok) ? 1 : 0;
        OUT[size_t(b) * S + t] = float(st);
    }
}
