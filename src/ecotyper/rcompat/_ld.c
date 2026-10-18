/* R's long-double reductions (x86-64 x87 80-bit), transcribed from R 4.4.3:
 *   src/main/summary.c  (rsum, real_mean)
 *   src/main/array.c    (colSums/colMeans/rowSums/rowMeans)
 *   src/library/stats/src/cov.c (cov_na_1, cov_na_2, MEAN_, COV_SDEV)
 *
 * Built on demand by _ld.py (cc -O2 -ffp-contract=off). Every accumulator and
 * every intermediate that R declares LDOUBLE is `long double` here; values R
 * stores in `double` (xm[], ANS(), the final result) are rounded to double at
 * exactly the same places. All matrices are C-contiguous.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef long double LD;
typedef int64_t i64;

#define FINITE_(v) isfinite((double)(v))
#define CLAMP(X) ((X) >= 1. ? 1. : ((X) <= -1. ? -1. : (X)))

/* ---- summary.c ---------------------------------------------------------- */

/* sum(x) */
double ld_sum(const double *x, i64 n)
{
    LD s = 0.0;
    for (i64 k = 0; k < n; k++) s += x[k];
    if (s > DBL_MAX) return INFINITY;
    if (s < -DBL_MAX) return -INFINITY;
    return (double)s;
}

/* mean(x) for a double vector (real_mean) */
double ld_mean(const double *x, i64 n)
{
    LD s = 0.0;
    for (i64 k = 0; k < n; k++) s += x[k];
    int finite_s = FINITE_(s);
    if (finite_s) {
        s /= n;
    } else {
        s = 0.;
        for (i64 k = 0; k < n; k++) s += x[k] / n;
    }
    if (finite_s && FINITE_(s)) {
        LD t = 0.0;
        for (i64 k = 0; k < n; k++) t += (x[k] - s);
        s += t / n;
    } else if (FINITE_(s)) {
        LD t = 0.0;
        for (i64 k = 0; k < n; k++) t += (x[k] - s) / n;
        s += t;
    }
    return (double)s;
}

/* apply(X, 1, mean) ; narm != 0: apply(X, 1, mean, na.rm = TRUE) */
void ld_row_mean(const double *X, i64 nr, i64 nc, int narm, double *out)
{
    double *tmp = NULL;
    if (narm) tmp = (double *)malloc(sizeof(double) * (nc > 0 ? nc : 1));
    for (i64 i = 0; i < nr; i++) {
        const double *row = X + i * nc;
        if (!narm) {
            out[i] = ld_mean(row, nc);
        } else {
            i64 c = 0;
            for (i64 j = 0; j < nc; j++)
                if (row[j] == row[j]) tmp[c++] = row[j];
            out[i] = c == 0 ? NAN : ld_mean(tmp, c);
        }
    }
    free(tmp);
}

/* ---- array.c ------------------------------------------------------------ */

/* colSums over a C-contiguous (nr, nc) block: sequential over rows */
void ld_col_sums(const double *X, i64 nr, i64 nc, double *out)
{
    LD *acc = (LD *)calloc(nc > 0 ? nc : 1, sizeof(LD));
    for (i64 i = 0; i < nr; i++) {
        const double *row = X + i * nc;
        for (i64 j = 0; j < nc; j++) acc[j] += row[j];
    }
    for (i64 j = 0; j < nc; j++) out[j] = (double)acc[j];
    free(acc);
}

/* colMeans(X) (keepNA) */
void ld_col_means(const double *X, i64 nr, i64 nc, double *out)
{
    LD *acc = (LD *)calloc(nc > 0 ? nc : 1, sizeof(LD));
    for (i64 i = 0; i < nr; i++) {
        const double *row = X + i * nc;
        for (i64 j = 0; j < nc; j++) acc[j] += row[j];
    }
    for (i64 j = 0; j < nc; j++) {
        acc[j] /= nr;
        out[j] = (double)acc[j];
    }
    free(acc);
}

/* rowMeans(X) (keepNA) */
void ld_row_means(const double *X, i64 nr, i64 nc, double *out)
{
    for (i64 i = 0; i < nr; i++) {
        const double *row = X + i * nc;
        LD s = 0.0;
        for (i64 j = 0; j < nc; j++) s += row[j];
        s /= nc;
        out[i] = (double)s;
    }
}

/* ---- cov.c -------------------------------------------------------------- */

/* MEAN_ for one column; returns the value R stores in xm[] (a double) */
static double cov_mean_(const double *xx, i64 n)
{
    LD sum = 0., tmp;
    for (i64 k = 0; k < n; k++) sum += xx[k];
    tmp = sum / n;
    if (FINITE_(tmp)) {
        sum = 0.;
        for (i64 k = 0; k < n; k++) sum += (xx[k] - tmp);
        tmp = tmp + sum / n;
    }
    return (double)tmp;
}

static int has_nan_(const double *x, i64 n)
{
    for (i64 k = 0; k < n; k++)
        if (x[k] != x[k]) return 1;
    return 0;
}

/* LD sum of products of centred columns / n1 (the unrounded value) */
static LD cov_sum_(const double *xx, double xm, const double *yy, double ym, i64 n, i64 n1)
{
    LD sum = 0., xxm = xm, yym = ym;
    for (i64 k = 0; k < n; k++) sum += (LD)(xx[k] - xxm) * (yy[k] - yym);
    return sum / n1;
}

/* var(x) for a vector (cov_na_1, ncx = 1, cor = FALSE) */
double ld_var(const double *x, i64 n)
{
    if (n <= 1) return NAN;
    if (has_nan_(x, n)) return NAN;
    double m = cov_mean_(x, n);
    return (double)cov_sum_(x, m, x, m, n, n - 1);
}

void ld_row_var(const double *X, i64 nr, i64 nc, double *out)
{
    for (i64 i = 0; i < nr; i++) out[i] = ld_var(X + i * nc, nc);
}

/* cor(X) (cov_na_1 with cor = TRUE) for the columns of X given as the ROWS of
 * XT (nc, n). out is (nc, nc). */
void ld_cor_cols(const double *XT, i64 n, i64 nc, double *out)
{
    if (n <= 1) {
        for (i64 i = 0; i < nc * nc; i++) out[i] = NAN;
        return;
    }
    double *xm = (double *)malloc(sizeof(double) * (nc > 0 ? nc : 1));
    int *has_na = (int *)malloc(sizeof(int) * (nc > 0 ? nc : 1));
    for (i64 i = 0; i < nc; i++) {
        has_na[i] = has_nan_(XT + i * n, n);
        xm[i] = has_na[i] ? NAN : cov_mean_(XT + i * n, n);
    }
    i64 n1 = n - 1;
    /* pairs (j <= i); each is an independent sequential sum */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (i64 i = 0; i < nc; i++) {
        if (has_na[i]) {
            for (i64 j = 0; j <= i; j++) out[j * nc + i] = out[i * nc + j] = NAN;
            continue;
        }
        const double *xx = XT + i * n;
        for (i64 j = 0; j <= i; j++) {
            if (has_na[j]) {
                out[j * nc + i] = out[i * nc + j] = NAN;
            } else {
                double v = (double)cov_sum_(xx, xm[i], XT + j * n, xm[j], n, n1);
                out[j * nc + i] = out[i * nc + j] = v;
            }
        }
    }
    for (i64 i = 0; i < nc; i++)
        if (!has_na[i]) xm[i] = sqrt(out[i * nc + i]);
    for (i64 i = 0; i < nc; i++) {
        if (!has_na[i])
            for (i64 j = 0; j < i; j++) {
                if (xm[i] == 0 || xm[j] == 0) {
                    out[j * nc + i] = out[i * nc + j] = NAN;
                } else {
                    LD sum = out[i * nc + j] / (xm[i] * xm[j]);
                    double v = (double)CLAMP(sum);
                    out[j * nc + i] = out[i * nc + j] = v;
                }
            }
        out[i * nc + i] = 1.0;
    }
    free(xm);
    free(has_na);
}

/* cor(x, y) for two vectors (cov_na_2, cor = TRUE) */
double ld_cor2(const double *x, const double *y, i64 n)
{
    if (n <= 1) return NAN;
    if (has_nan_(x, n) || has_nan_(y, n)) return NAN;
    double xm = cov_mean_(x, n), ym = cov_mean_(y, n);
    i64 n1 = n - 1;
    double ans = (double)cov_sum_(x, xm, y, ym, n, n1);
    double sx = (double)sqrtl(cov_sum_(x, xm, x, xm, n, n1));
    double sy = (double)sqrtl(cov_sum_(y, ym, y, ym, n, n1));
    if (sx == 0. || sy == 0.) return NAN;
    ans /= (sx * sy);
    return CLAMP(ans);
}

/* ---- scale -------------------------------------------------------------- */

/* as.vector(scale(x)): colMeans(na.rm = TRUE), x - center, then
 * sqrt(sum(v^2) / max(1, length(v) - 1)) over the non-NA values */
void ld_scale_vec(const double *x, i64 n, double *out)
{
    LD s = 0.0;
    i64 cnt = 0;
    for (i64 k = 0; k < n; k++)
        if (x[k] == x[k]) {
            s += x[k];
            cnt++;
        }
    s /= cnt;
    double center = (double)s;
    for (i64 i = 0; i < n; i++) out[i] = x[i] - center;
    LD ss = 0.0;
    cnt = 0;
    for (i64 i = 0; i < n; i++) {
        double v = out[i];
        if (v == v) {
            ss += v * v;
            cnt++;
        }
    }
    double ssd = (double)ss; /* sum() returns a double */
    i64 denom = cnt - 1 > 1 ? cnt - 1 : 1;
    double sc = sqrt(ssd / (double)denom);
    for (i64 i = 0; i < n; i++) out[i] = out[i] / sc;
}

void ld_scale_rows(const double *X, i64 nr, i64 nc, double *out)
{
    for (i64 i = 0; i < nr; i++) ld_scale_vec(X + i * nc, nc, out + i * nc);
}

/* apply(X, 2, function(x) x / sum(x[, na.rm = TRUE])) */
void ld_col_normalize(const double *X, i64 nr, i64 nc, int narm, double *out)
{
    LD *acc = (LD *)calloc(nc > 0 ? nc : 1, sizeof(LD));
    for (i64 i = 0; i < nr; i++) {
        const double *row = X + i * nc;
        for (i64 j = 0; j < nc; j++)
            if (!(narm && row[j] != row[j])) acc[j] += row[j];
    }
    double *s = (double *)malloc(sizeof(double) * (nc > 0 ? nc : 1));
    for (i64 j = 0; j < nc; j++) {
        LD a = acc[j];
        s[j] = a > DBL_MAX ? INFINITY : (a < -DBL_MAX ? -INFINITY : (double)a);
    }
    for (i64 i = 0; i < nr; i++)
        for (i64 j = 0; j < nc; j++) out[i * nc + j] = X[i * nc + j] / s[j];
    free(acc);
    free(s);
}
