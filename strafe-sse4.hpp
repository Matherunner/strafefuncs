#ifndef FASTSTRAFE_H
#define FASTSTRAFE_H

#include <smmintrin.h>

struct c1_params_t {
    __m128d Ls;
    __m128d zeroLsq;
    __m128d twoLsqs;
};

struct c2_params_t {
    __m128d tauMAs;
    __m128d LtauMAs;
    __m128d LtauMAsqs;
    __m128d zerospdcnst;
    __m128d twospdcnsts;
};

inline double p2l_distsq(__m128d p, __m128d ols, __m128d dls)
{
    __m128d tmp = _mm_sub_pd(ols, p);
    __m128d dot = _mm_dp_pd(tmp, dls, 0x33);
    dot = _mm_mul_pd(dot, dls);
    tmp = _mm_sub_pd(tmp, dot);
    tmp = _mm_dp_pd(tmp, tmp, 0x31);
    double val;
    _mm_store_sd(&val, tmp);
    return val;
}

inline __m128d strafe_newpos(__m128d p, __m128d v, __m128d tau)
{
    return _mm_add_pd(p, _mm_mul_pd(v, tau));
}

void strafe_c1_precom(double L, c1_params_t &params)
{
    params.Ls = _mm_loaddup_pd(&L);
    __m128d Lsqs = _mm_mul_pd(params.Ls, params.Ls);
    params.twoLsqs = _mm_add_pd(Lsqs, Lsqs);
    params.zeroLsq = _mm_move_sd(Lsqs, _mm_setzero_pd());
}

void strafe_c2_precom(double L, double tauMA, c2_params_t &params)
{
    params.tauMAs = _mm_loaddup_pd(&tauMA);
    double LtauMA = L - tauMA;
    params.LtauMAs = _mm_loaddup_pd(&LtauMA);
    params.LtauMAsqs = _mm_mul_pd(params.LtauMAs, params.LtauMAs);
    double spdcnst = tauMA * (L + LtauMA);
    __m128d spdcnsts = _mm_loaddup_pd(&spdcnst);
    params.zerospdcnst = _mm_move_sd(spdcnsts, _mm_setzero_pd());
    params.twospdcnsts = _mm_add_pd(spdcnsts, spdcnsts);
}

inline void strafe_c1_Lspds(__m128d *Lspds, size_t n, __m128d v,
                            const c1_params_t &params)
{
    __m128d v0sqs = _mm_dp_pd(v, v, 0x33);
    v0sqs = _mm_add_pd(v0sqs, params.zeroLsq);
    size_t i = 0;
    while (i < n) {
        __m128d tmp = _mm_div_pd(params.Ls, _mm_sqrt_pd(v0sqs));
        v0sqs = _mm_add_pd(v0sqs, params.twoLsqs);
        Lspds[i++] = _mm_movedup_pd(tmp);
        Lspds[i++] = _mm_unpackhi_pd(tmp, tmp);
    }
}

inline void strafe_c2_ctsts(__m128d *cts, __m128d *sts, size_t n, __m128d v,
                            const c2_params_t &params)
{
    __m128d v0sqs = _mm_dp_pd(v, v, 0x33);
    v0sqs = _mm_add_pd(v0sqs, params.zerospdcnst);
    size_t i = 0;
    while (i < n) {
        __m128d mults = _mm_div_pd(params.tauMAs, v0sqs);
        __m128d cttmp = _mm_mul_pd(params.LtauMAs, mults);
        __m128d sttmp = _mm_sub_pd(v0sqs, params.LtauMAsqs);
        sttmp = _mm_sqrt_pd(sttmp);
        sttmp = _mm_mul_pd(sttmp, mults);
        v0sqs = _mm_add_pd(v0sqs, params.twospdcnsts);
        cts[i] = _mm_movedup_pd(cttmp);
        sts[i] = _mm_movedup_pd(sttmp);
        i++;
        cts[i] = _mm_unpackhi_pd(cttmp, cttmp);
        sts[i] = _mm_unpackhi_pd(sttmp, sttmp);
        i++;
    }
}

inline __m128d strafe_c1_side(__m128d v, __m128d Lspd)
{
    __m128d a = _mm_mul_pd(v, Lspd);
    a = _mm_shuffle_pd(a, a, 1);
    return _mm_addsub_pd(v, a);
}

inline __m128d strafe_c2_side(__m128d v, __m128d ct, __m128d st)
{
    ct = _mm_mul_pd(v, ct);
    st = _mm_mul_pd(v, st);
    v = _mm_add_pd(v, ct);
    st = _mm_shuffle_pd(st, st, 1);
    return _mm_addsub_pd(v, st);
}

#endif