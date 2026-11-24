// Inkling 2, a prompt's or verify pass's S rows: the picked experts' outputs back to their rows, the MLP's short
// conv and the final add.
// ROLL == 1 (SEQ == 1 && IDENT == 1, grid = D): each thread owns one channel c across all S rows, holding the
// KT-element sliding conv window and MW taps in registers so each (row, expert) DN value is read ONCE instead of KT
// times (4x fewer DN reads, Sx fewer ST/MW reads) with bit-identical results.
const uint D = DN_shape[2];
const uint SW = ST_shape[2];
const uint OFF = SW - D;
if (ROLL) {
    const uint c = thread_position_in_grid.x;
    const uint S = RES_shape[1];
    float win[KT];
    float mw[KT];
    for (uint j = 0; j < KT - 1; j++) win[j] = ST[j * SW + OFF + c];
    for (uint j = 0; j < KT; j++) mw[j] = float(MW[c * KT + j]);
    for (uint s = 0; s < S; s++) {
        float x = 0.0f;
        for (uint k = 0; k < K; k++) {
            const uint at = s * K + k;
            x += float(DN[size_t(at) * D + c]) * (SCALED ? SC[IDX[at]] : 1.0f) * WTS[at];
        }
        const T xm = T(x);
        const float xmf = float(xm);
        win[KT - 1] = xmf;
        const uint e = s * D + c;
        XM[e] = xm;
        float y = 0.0f;
        for (uint j = 0; j < KT; j++) y += win[j] * mw[j];
        y = float(WT(y));
        HOUT[e] = T(RES[e] + float(T((y + xmf) * PR[0])));
        for (uint j = 0; j + 1 < KT; j++) win[j] = win[j + 1];
    }
} else {
    const uint e = thread_position_in_grid.x;
    const uint s = e / D, c = e % D;
    float win[KT];
    for (uint j = 0; j < KT; j++) {
        if (SEQ ? s + j < KT - 1 : j + 1 < KT) {
            win[j] = ST[(SEQ ? s + j : s * (KT - 1) + j) * SW + OFF + c];
            continue;
        }
        const uint r = SEQ ? s + j - (KT - 1) : s;
        float x = 0.0f;
        for (uint k = 0; k < K; k++) {
            const uint at = IDENT ? (r * K + k) : uint(INV[r * K + k]);
            x += float(DN[size_t(at) * D + c]) * (SCALED ? SC[IDX[at]] : 1.0f) * WTS[r * K + k];
        }
        win[j] = float(T(x));
    }
    XM[e] = T(win[KT - 1]);
    float y = 0.0f;
    for (uint j = 0; j < KT; j++) y += win[j] * float(MW[c * KT + j]);
    y = float(WT(y));
    HOUT[e] = T(RES[e] + float(T((y + win[KT - 1]) * PR[0])));
}
