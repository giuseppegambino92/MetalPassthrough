// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// MSL runtime support emitted at the top of every translated PTX kernel.

enum MSLPrelude {
    static let source = #"""
#include <metal_stdlib>
using namespace metal;

// ---- float -> int conversions with PTX semantics (saturating, NaN -> 0) ----
static inline int _f2s32(float x) { return isnan(x) ? 0 : (x >= 2147483648.0f ? 2147483647 : (x <= -2147483648.0f ? int(0x80000000u) : int(x))); }
static inline uint _f2u32(float x) { return (isnan(x) || x <= 0.0f) ? 0u : (x >= 4294967296.0f ? 0xffffffffu : uint(x)); }
static inline long _f2s64(float x) { return isnan(x) ? 0l : (x >= 9223372036854775808.0f ? 0x7fffffffffffffffl : (x <= -9223372036854775808.0f ? long(0x8000000000000000ul) : long(x))); }
static inline ulong _f2u64(float x) { return (isnan(x) || x <= 0.0f) ? 0ul : (x >= 18446744073709551616.0f ? 0xfffffffffffffffful : ulong(x)); }

// ---- bit manipulation with PTX edge-case semantics ----
static inline uint _prmt(uint a, uint b, uint c) {
    ulong x = (ulong(b) << 32) | ulong(a); uint r = 0u;
    for (uint i = 0; i < 4u; i++) {
        uint sel = (c >> (4u * i)) & 0xfu; uint byte = uint((x >> ((sel & 7u) * 8u)) & 0xfful);
        if (sel & 8u) byte = (byte & 0x80u) ? 0xffu : 0u;
        r |= byte << (8u * i);
    }
    return r;
}
static inline uint _bfe32(uint a, uint pos, uint len, bool sgn) {
    pos &= 0xffu; len &= 0xffu; if (len == 0u) return 0u;
    uint top = min(pos + len - 1u, 31u); uint sbit = sgn ? ((a >> top) & 1u) : 0u;
    uint avail = pos > 31u ? 0u : min(len, 32u - pos);
    uint mask = avail >= 32u ? 0xffffffffu : ((1u << avail) - 1u);
    uint field = avail == 0u ? 0u : ((a >> pos) & mask);
    return field | ((sbit ? 0xffffffffu : 0u) & ~mask);
}
static inline ulong _bfe64(ulong a, uint pos, uint len, bool sgn) {
    pos &= 0xffu; len &= 0xffu; if (len == 0u) return 0ul;
    uint top = min(pos + len - 1u, 63u); ulong sbit = sgn ? ((a >> top) & 1ul) : 0ul;
    uint avail = pos > 63u ? 0u : min(len, 64u - pos);
    ulong mask = avail >= 64u ? 0xfffffffffffffffful : ((1ul << avail) - 1ul);
    ulong field = avail == 0u ? 0ul : ((a >> pos) & mask);
    return field | ((sbit ? 0xfffffffffffffffful : 0ul) & ~mask);
}
static inline uint _bfi32(uint a, uint b, uint pos, uint len) {
    pos &= 0xffu; len &= 0xffu; if (len == 0u || pos >= 32u) return b;
    uint l = min(len, 32u - pos); uint m = (l >= 32u ? 0xffffffffu : ((1u << l) - 1u)) << pos;
    return (b & ~m) | ((a << pos) & m);
}
static inline ulong _bfi64(ulong a, ulong b, uint pos, uint len) {
    pos &= 0xffu; len &= 0xffu; if (len == 0u || pos >= 64u) return b;
    uint l = min(len, 64u - pos); ulong m = (l >= 64u ? 0xfffffffffffffffful : ((1ul << l) - 1ul)) << pos;
    return (b & ~m) | ((a << pos) & m);
}
static inline uint _shf_l(uint a, uint b, uint c, bool wrap) { uint n = wrap ? (c & 31u) : min(c, 32u); ulong x = (ulong(b) << 32) | ulong(a); return uint((x << n) >> 32); }
static inline uint _shf_r(uint a, uint b, uint c, bool wrap) { uint n = wrap ? (c & 31u) : min(c, 32u); ulong x = (ulong(b) << 32) | ulong(a); return uint(x >> n); }
static inline uint _bfind32(uint a, bool sgn, bool shamt) {
    uint x = (sgn && (a & 0x80000000u)) ? ~a : a; if (x == 0u) return 0xffffffffu;
    uint p = 31u - clz(x); return shamt ? 31u - p : p;
}
static inline uint _bfind64(ulong a, bool sgn, bool shamt) {
    ulong x = (sgn && (a >> 63)) ? ~a : a; if (x == 0ul) return 0xffffffffu;
    uint p = 63u - uint(clz(x)); return shamt ? 63u - p : p;
}

// ---- atomics without native Metal equivalents ----
static inline uint _cas_d(device atomic_uint* p, uint cmp, uint val) {
    uint e = cmp;
    while (!atomic_compare_exchange_weak_explicit(p, &e, val, memory_order_relaxed, memory_order_relaxed)) { if (e != cmp) return e; e = cmp; }
    return cmp;
}
static inline uint _cas_t(threadgroup atomic_uint* p, uint cmp, uint val) {
    uint e = cmp;
    while (!atomic_compare_exchange_weak_explicit(p, &e, val, memory_order_relaxed, memory_order_relaxed)) { if (e != cmp) return e; e = cmp; }
    return cmp;
}
static inline uint _inc_d(device atomic_uint* p, uint b) {
    uint o = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &o, (o >= b) ? 0u : o + 1u, memory_order_relaxed, memory_order_relaxed)) {}
    return o;
}
static inline uint _inc_t(threadgroup atomic_uint* p, uint b) {
    uint o = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &o, (o >= b) ? 0u : o + 1u, memory_order_relaxed, memory_order_relaxed)) {}
    return o;
}
static inline uint _dec_d(device atomic_uint* p, uint b) {
    uint o = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &o, (o == 0u || o > b) ? b : o - 1u, memory_order_relaxed, memory_order_relaxed)) {}
    return o;
}
static inline uint _dec_t(threadgroup atomic_uint* p, uint b) {
    uint o = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &o, (o == 0u || o > b) ? b : o - 1u, memory_order_relaxed, memory_order_relaxed)) {}
    return o;
}
static inline float _addf_t(threadgroup atomic_uint* p, float v) {
    uint o = atomic_load_explicit(p, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(p, &o, as_type<uint>(as_type<float>(o) + v), memory_order_relaxed, memory_order_relaxed)) {}
    return as_type<float>(o);
}

// ---- df64: double-float emulation of FP64 (~48-bit mantissa, float exponent range) ----
struct df64 { float hi; float lo; };
static inline df64 _df(float h, float l) { df64 r; r.hi = h; r.lo = l; return r; }
static inline df64 _df_qts(float a, float b) { float s = a + b; float e = b - (s - a); return _df(s, e); }
static inline df64 _df_ts(float a, float b) { float s = a + b; float v = s - a; float e = (a - (s - v)) + (b - v); return _df(s, e); }
static inline df64 _df_add(df64 a, df64 b) {
    df64 s = _df_ts(a.hi, b.hi); df64 t = _df_ts(a.lo, b.lo);
    float e = s.lo + t.hi; s = _df_qts(s.hi, e); e = s.lo + t.lo; return _df_qts(s.hi, e);
}
static inline df64 _df_neg(df64 a) { return _df(-a.hi, -a.lo); }
static inline df64 _df_sub(df64 a, df64 b) { return _df_add(a, _df_neg(b)); }
static inline df64 _df_mul(df64 a, df64 b) {
    float p = a.hi * b.hi; float e = fma(a.hi, b.hi, -p); e = fma(a.hi, b.lo, e); e = fma(a.lo, b.hi, e); return _df_qts(p, e);
}
static inline df64 _df_fma(df64 a, df64 b, df64 c) { return _df_add(_df_mul(a, b), c); }
static inline df64 _df_div(df64 a, df64 b) {
    float q1 = a.hi / b.hi; df64 r = _df_sub(a, _df_mul(b, _df(q1, 0.0f)));
    float q2 = r.hi / b.hi; r = _df_sub(r, _df_mul(b, _df(q2, 0.0f)));
    float q3 = r.hi / b.hi; return _df_add(_df_qts(q1, q2), _df(q3, 0.0f));
}
static inline df64 _df_sqrt(df64 a) {
    if (!(a.hi > 0.0f)) return _df(sqrt(a.hi), 0.0f);
    float x = rsqrt(a.hi); float y = a.hi * x;
    float d = _df_sub(a, _df_mul(_df(y, 0.0f), _df(y, 0.0f))).hi;
    return _df_add(_df(y, 0.0f), _df(d * x * 0.5f, 0.0f));
}
static inline df64 _df_rcp(df64 a) { return _df_div(_df(1.0f, 0.0f), a); }
static inline df64 _df_abs(df64 a) { return (a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) ? _df_neg(a) : a; }
static inline int _df_cmp(df64 a, df64 b) {
    if (isnan(a.hi) || isnan(b.hi)) return 2;
    if (a.hi < b.hi) return -1; if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1; if (a.lo > b.lo) return 1; return 0;
}
static inline df64 _df_min(df64 a, df64 b) { if (isnan(a.hi)) return b; if (isnan(b.hi)) return a; return _df_cmp(a, b) <= 0 ? a : b; }
static inline df64 _df_max(df64 a, df64 b) { if (isnan(a.hi)) return b; if (isnan(b.hi)) return a; return _df_cmp(a, b) >= 0 ? a : b; }
static inline float _df_to_f32(df64 a) { return a.hi + a.lo; }
static inline df64 _df_from_f32(float a) { return _df(a, 0.0f); }
static inline df64 _df_from_s64(long x) {
    float h = float(x); long hl = h >= 9223372036854775808.0f ? 0x7fffffffffffffffl : (h <= -9223372036854775808.0f ? long(0x8000000000000000ul) : long(h));
    return _df_qts(h, float(x - hl));
}
static inline df64 _df_from_u64(ulong x) {
    float h = float(x); ulong hl = h >= 18446744073709551616.0f ? 0xfffffffffffffffful : ulong(h);
    return _df_qts(h, float(long(x - hl)));
}
// mode: 0 = rn (nearest even), 1 = rz, 2 = rm, 3 = rp
static inline long _df_to_s64(df64 a, int mode) {
    if (isnan(a.hi)) return 0l;
    if (a.hi >= 9223372036854775808.0f) return 0x7fffffffffffffffl;
    if (a.hi <= -9223372036854775808.0f) return long(0x8000000000000000ul);
    float t = trunc(a.hi); long r = long(t); float rem = (a.hi - t) + a.lo;
    float fl = floor(rem); long base = r + long(fl); float frac = rem - fl;
    if (frac == 0.0f) return base;
    switch (mode) {
        case 1: return base < 0 ? base + 1 : base;
        case 2: return base;
        case 3: return base + 1;
        default: return (frac > 0.5f || (frac == 0.5f && (base & 1l))) ? base + 1 : base;
    }
}
static inline df64 _df_round(df64 a, int mode) { return (fabs(a.hi) >= 9.0e18f || isnan(a.hi)) ? a : _df_from_s64(_df_to_s64(a, mode)); }
static inline df64 _df_from_bits(ulong b) {
    uint e = uint((b >> 52) & 0x7fful); ulong m = b & 0xffffffffffffful; float sgn = (b >> 63) ? -1.0f : 1.0f;
    if (e == 0x7ffu) return _df(m == 0ul ? sgn * INFINITY : NAN, 0.0f);
    if (e == 0u) return _df(sgn * 0.0f, 0.0f);
    m |= (1ul << 52); int ex = int(e) - 1075;
    float hi = ldexp(float(m >> 29), ex + 29); float lo = ldexp(float(m & 0x1ffffffful), ex);
    df64 r = _df_qts(hi, lo); return _df(sgn * r.hi, sgn * r.lo);
}
static inline ulong _df_to_bits(df64 a) {
    if (isnan(a.hi)) return 0x7ff8000000000000ul;
    if (isinf(a.hi)) return a.hi > 0.0f ? 0x7ff0000000000000ul : 0xfff0000000000000ul;
    if (a.hi == 0.0f) return (as_type<uint>(a.hi) >> 31) ? 0x8000000000000000ul : 0ul;
    int eh = 0, el = 0; float fh = frexp(a.hi, eh); float fl = frexp(a.lo, el);
    long mh = long(fh * 16777216.0f); long ml = long(fl * 16777216.0f);
    int Eh = eh - 24, El = el - 24;
    // Work at scale 2^(Eh-29): hi contributes 53 bits; lo (|lo| <= ulp(hi)/2) is folded in with rounding.
    int E = Eh - 29; long M = mh * (1l << 29);
    if (a.lo != 0.0f) {
        int sh = El - E;
        if (sh >= 0) M += ml * (1l << sh);
        else if (sh > -63) M += (ml >> (-sh)) + ((ml >> (-sh - 1)) & 1l);
    }
    ulong sign = 0ul; if (M < 0) { sign = 1ul; M = -M; }
    ulong U = ulong(M); if (U == 0ul) return sign << 63;
    int n = 63 - int(clz(U));
    if (n > 52) {
        int sh = n - 52; ulong rem = U & ((1ul << sh) - 1ul); ulong hv = 1ul << (sh - 1); U >>= sh; E += sh;
        if (rem > hv || (rem == hv && (U & 1ul))) { U += 1ul; if (U >> 53) { U >>= 1; E += 1; } }
    } else { U <<= (52 - n); E -= (52 - n); }
    int be = E + 52 + 1023;
    if (be >= 2047) return (sign << 63) | 0x7ff0000000000000ul;
    if (be <= 0) return sign << 63;
    return (sign << 63) | (ulong(be) << 52) | (U & 0xffffffffffffful);
}
// fp32-lowered FP64: keep IEEE double *storage* exact, compute in fp32.
static inline float _f64bits_to_f32(ulong b) { df64 d = _df_from_bits(b); return d.hi + d.lo; }
static inline ulong _f32_to_f64bits(float x) { return _df_to_bits(_df(x, 0.0f)); }

"""#
}
