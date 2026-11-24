// Inkling 2: GPU planner grouping NP = S * KSEL (row, pick) pairs by expert into slots of up to M pairs.
// One threadgroup of TG >= E threads (TG a multiple of 32): thread e < E scans IDX [NP] u32 with packed_uint4 loads,
// packs its m pairs into ceil(m / M) contiguous slots via threadgroup prefix sum, and fills remaining slots up to
// NSLOTS with 0xFFFFFFFFu so ups_dedup / downs_dedup exit empty threadgroups immediately.
const uint e = thread_position_in_threadgroup.x;
threadgroup uint cnt[TG];
uint pairs[KM];
uint m = 0;
if (e < E) {
    uint p = 0;
    for (; p + 4 <= NP; p += 4) {
        const vec<uint, 4> v = *((const device packed_uint4*)(IDX + p));
        if (v.x == e && m < KM) pairs[m++] = p;
        if (v.y == e && m < KM) pairs[m++] = p + 1;
        if (v.z == e && m < KM) pairs[m++] = p + 2;
        if (v.w == e && m < KM) pairs[m++] = p + 3;
    }
    for (; p < NP; p++) {
        if (IDX[p] == e && m < KM) pairs[m++] = p;
    }
}
const uint ns = (m + M - 1) / M;
cnt[e] = ns;
threadgroup_barrier(mem_flags::mem_threadgroup);
vec<uint, 4> offv = vec<uint, 4>(0u), totv = vec<uint, 4>(0u);
for (uint i = 0; i < TG; i += 4) {
    const vec<uint, 4> c4 = *((threadgroup packed_uint4*)(cnt + i));
    totv += c4;
    offv += select(vec<uint, 4>(0u), c4, (vec<uint, 4>(i) + vec<uint, 4>(0u, 1u, 2u, 3u)) < vec<uint, 4>(e));
}
const uint off = offv.x + offv.y + offv.z + offv.w;
const uint total = totv.x + totv.y + totv.z + totv.w;
for (uint s = 0; s < ns; s++) {
    SLOT_E[off + s] = e;
    for (uint i = 0; i < M; i++) SLOT_P[(off + s) * M + i] = (s * M + i < m) ? pairs[s * M + i] : 0xFFFFFFFFu;
}
for (uint s = total + e; s < NSLOTS; s += TG) {
    SLOT_E[s] = 0xFFFFFFFFu;
    for (uint i = 0; i < M; i++) SLOT_P[s * M + i] = 0xFFFFFFFFu;
}
