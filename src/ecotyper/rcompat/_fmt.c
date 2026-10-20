/* write.table's number formatting (R 4.4.3 formatReal + EncodeReal0, digits = 15) for whole
 * matrices: the C version of rcompat/io.py:format_real, used by write_table/write_matrix.
 *
 * scientific() comes in the two variants io.py uses:
 *   sci_ld  - R's long-double build (x86-64), identical to ld_scientific in _ld.c;
 *   sci_dbl - the double-precision build (Apple silicon, where long double is double),
 *             identical to io.py's Python emulation.
 * Built on demand with cc -O2 -ffp-contract=off. Output: cells separated by '\t', rows by '\n'.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long double LD;
typedef int64_t i64;

static const LD fmt_tbl[] = {1e00, 1e01, 1e02, 1e03, 1e04, 1e05, 1e06, 1e07, 1e08, 1e09,
                             1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                             1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27};
#define FMT_KP_MAX 27

static void sci_ld(double x, int digits, int *neg, int *kpower, int *nsig, int *widens) {
    double r;
    if (x < 0.0) { *neg = 1; r = -x; } else { *neg = 0; r = x; }
    int kp = (int)floor(log10(r)) - digits + 1;
    LD r_prec = r;
    if (abs(kp) <= FMT_KP_MAX) {
        if (kp > 0) r_prec /= fmt_tbl[kp]; else if (kp < 0) r_prec *= fmt_tbl[-kp];
    } else
        r_prec /= powl(10.0L, (LD)kp);
    if (r_prec < fmt_tbl[digits - 1]) { r_prec *= 10.0; kp--; }
    double alpha = (double)nearbyintl(r_prec);
    *nsig = digits;
    for (int j = 1; j <= digits; j++) {
        alpha /= 10.0;
        if (alpha == floor(alpha)) (*nsig)--; else break;
    }
    if (*nsig == 0 && digits > 0) { *nsig = 1; kp += 1; }
    *kpower = kp + digits - 1;
    int rgt = digits - *kpower;
    rgt = rgt < 0 ? 0 : rgt > FMT_KP_MAX ? FMT_KP_MAX : rgt;
    double fuzz = 0.5 / (double)fmt_tbl[rgt];
    *widens = *kpower > 0 && *kpower <= FMT_KP_MAX && (LD)r < fmt_tbl[*kpower] - (LD)fuzz;
}

static const double dbl_tbl[] = {1e00, 1e01, 1e02, 1e03, 1e04, 1e05, 1e06, 1e07, 1e08, 1e09, 1e10, 1e11,
                                 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
#define DBL_KP_MAX 22

static void sci_dbl(double x, int digits, int *neg, int *kpower, int *nsig, int *widens) {
    double r;
    if (x < 0.0) { *neg = 1; r = -x; } else { *neg = 0; r = x; }
    int kp = (int)floor(log10(r)) - digits + 1;
    double r_prec = r;
    if (abs(kp) <= DBL_KP_MAX) {
        if (kp >= 0) r_prec /= dbl_tbl[kp]; else r_prec *= dbl_tbl[-kp];
    } else if (kp <= -308)
        r_prec = (r_prec * 1e303) / pow(10.0, (double)(kp + 303));
    else
        r_prec /= pow(10.0, (double)kp);
    if (r_prec < dbl_tbl[digits - 1]) { r_prec *= 10.0; kp--; }
    double alpha = nearbyint(r_prec);
    *nsig = digits;
    for (int j = 0; j < digits; j++) {
        alpha /= 10.0;
        if (alpha == floor(alpha)) (*nsig)--; else break;
    }
    if (*nsig == 0 && digits > 0) { *nsig = 1; kp += 1; }
    *kpower = kp + digits - 1;
    int rgt = digits - *kpower;
    rgt = rgt < 0 ? 0 : rgt > DBL_KP_MAX ? DBL_KP_MAX : rgt;
    double fuzz = 0.5 / dbl_tbl[rgt];
    *widens = *kpower > 0 && *kpower <= DBL_KP_MAX && r < dbl_tbl[*kpower] - fuzz;
}

/* one value; returns the number of characters written to out (at most 40) */
static int fmt1(double x, int digits, int use_ld, char *out) {
    if (isnan(x)) { memcpy(out, "NA", 2); return 2; }
    if (isinf(x)) {
        if (x > 0) { memcpy(out, "Inf", 3); return 3; }
        memcpy(out, "-Inf", 4); return 4;
    }
    if (x == 0.0) { out[0] = '0'; return 1; } /* R prints -0 as 0 */
    int neg, kpower, nsig, widens;
    if (use_ld) sci_ld(x, digits, &neg, &kpower, &nsig, &widens);
    else sci_dbl(x, digits, &neg, &kpower, &nsig, &widens);
    int left = kpower + 1;
    if (widens) left--;
    int sleft = neg + (left <= 0 ? 1 : left);
    int rgt = nsig - left;
    if (rgt < 0) rgt = 0;
    int mxsl = left < 0 ? 1 + neg : sleft;
    int wF = mxsl + rgt + (rgt != 0);
    int e = (left > 100 || left <= -99) ? 2 : 1;
    int d = nsig - 1;
    int wE = neg + (d > 0) + d + 4 + e;
    if (wF <= wE) return snprintf(out, 48, "%.*f", rgt, x);
    char tmp[48];
    int len = snprintf(tmp, sizeof tmp, "%.*e", d, x);
    char *ep = strchr(tmp, 'e');
    int mlen = (int)(ep - tmp);
    memcpy(out, tmp, mlen);
    int o = mlen;
    out[o++] = 'e';
    out[o++] = ep[1]; /* sign */
    const char *dig = ep + 2;
    while (*dig == '0') dig++;
    int nd = (int)(tmp + len - dig);
    for (int k = nd; k < e + 1; k++) out[o++] = '0';
    memcpy(out + o, dig, nd);
    return o + nd;
}

/* Formats the C-contiguous nrow x ncol matrix X into buf (cap bytes); returns the number of
 * bytes written, or -1 if cap is too small (48 bytes per cell always suffice). */
i64 fmt_matrix(const double *X, i64 nrow, i64 ncol, int digits, int use_ld, char *buf, i64 cap) {
    i64 o = 0;
    for (i64 i = 0; i < nrow; i++) {
        for (i64 j = 0; j < ncol; j++) {
            if (cap - o < 50) return -1;
            if (j) buf[o++] = '\t';
            o += fmt1(X[i * ncol + j], digits, use_ld, buf + o);
        }
        if (i + 1 < nrow) buf[o++] = '\n';
    }
    return o;
}
