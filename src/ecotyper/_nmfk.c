/* R-exact Brunet NMF iterations (NMF package v0.28, src/divergence.cpp), blocked for SIMD.
 *
 * Every element of W and H goes through the same floating-point operations, in the same
 * order, as in NMF's divergence_update_H / divergence_update_W:
 *   (WH)[u,j] = sum_k W[u,k] H[k,j]         k = 0..r-1, sequential (fused multiply-adds with -DUSE_FMA)
 *   tmp[i]    = sum_u W[u,i] V[u,j]/(WH)[u,j] u = 0..n-1, sequential
 *   H[i,j]    = H[i,j] * tmp[i] / sumW[i]     sumW[i] = sum_u W[u,i], sequential
 * and the transposed formulas for W. The columns of H (rows of W) are independent within one
 * half-step, so they are processed BW at a time; the compiler vectorises across them without
 * reordering any sum. Compile with -ffp-contract=off (no implicit contraction).
 *
 * Also: eps clamp every 10 iterations and nmf.stop.connectivity (check every check_interval
 * iterations, stop when the partition is unchanged for more than stopconv checks).
 *
 * With OpenMP (-fopenmp) the column blocks of a half-step can be split over nthreads threads;
 * each column still gets exactly the same operations, so the result does not depend on it.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_FMA
#define MADD(a, b, c) fma((a), (b), (c))
#else
#define MADD(a, b, c) ((a) * (b) + (c))
#endif

#ifndef BW
#define BW 32 /* columns per block */
#endif

/* sums[i] = sum_u A[u, i] (u sequential); A is n x r row-major */
static void col_sums(const double *A, int64_t n, int64_t r, double *sums) {
    for (int64_t i = 0; i < r; i++) sums[i] = 0.0;
    for (int64_t u = 0; u < n; u++) {
        const double *Au = A + u * r;
        for (int64_t i = 0; i < r; i++) sums[i] += Au[i];
    }
}

/* B[i, j] = B[i, j] * (sum_u A[u, i] * V[u, j] / (sum_k A[u, k] B[k, j])) / sumA[i]
 * V: n x p, A: n x r, B: r x p (row-major). whb_all: nthreads * n * BW scratch. */
static void mult_update(const double *V, const double *A, double *B, const double *sumA, int64_t n, int64_t r,
                        int64_t p, double *whb_all, int nthreads) {
    const int64_t nblk = (p + BW - 1) / BW;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) if (nthreads > 1)
#endif
    for (int64_t b = 0; b < nblk; b++) {
#ifdef _OPENMP
        double *whb = whb_all + (int64_t)omp_get_thread_num() * n * BW;
#else
        double *whb = whb_all;
#endif
        const int64_t j0 = b * BW;
        const int64_t jb = (p - j0 < BW) ? p - j0 : BW;
        for (int64_t u = 0; u < n; u++) {
            double s[BW];
            const double *Au = A + u * r;
            double *wu = whb + u * BW;
            if (jb == BW) {
                for (int jj = 0; jj < BW; jj++) s[jj] = 0.0;
                for (int64_t k = 0; k < r; k++) {
                    const double w = Au[k];
                    const double *Bk = B + k * p + j0;
                    for (int jj = 0; jj < BW; jj++) s[jj] = MADD(w, Bk[jj], s[jj]);
                }
                const double *Vu = V + u * p + j0;
                for (int jj = 0; jj < BW; jj++) wu[jj] = Vu[jj] / s[jj];
            } else {
                for (int jj = 0; jj < jb; jj++) s[jj] = 0.0;
                for (int64_t k = 0; k < r; k++) {
                    const double w = Au[k];
                    const double *Bk = B + k * p + j0;
                    for (int jj = 0; jj < jb; jj++) s[jj] = MADD(w, Bk[jj], s[jj]);
                }
                const double *Vu = V + u * p + j0;
                for (int jj = 0; jj < jb; jj++) wu[jj] = Vu[jj] / s[jj];
                for (int jj = (int)jb; jj < BW; jj++) wu[jj] = 0.0; /* padding, results discarded */
            }
        }
        int64_t i = 0;
        for (; i + 4 <= r; i += 4) {
            double a0[BW], a1[BW], a2[BW], a3[BW];
            for (int jj = 0; jj < BW; jj++) a0[jj] = a1[jj] = a2[jj] = a3[jj] = 0.0;
            for (int64_t u = 0; u < n; u++) {
                const double *Au = A + u * r + i;
                const double w0 = Au[0], w1 = Au[1], w2 = Au[2], w3 = Au[3];
                const double *wu = whb + u * BW;
                for (int jj = 0; jj < BW; jj++) {
                    const double x = wu[jj];
                    a0[jj] = MADD(w0, x, a0[jj]);
                    a1[jj] = MADD(w1, x, a1[jj]);
                    a2[jj] = MADD(w2, x, a2[jj]);
                    a3[jj] = MADD(w3, x, a3[jj]);
                }
            }
            double *B0 = B + i * p + j0;
            for (int jj = 0; jj < jb; jj++) {
                B0[jj] = B0[jj] * a0[jj] / sumA[i];
                B0[p + jj] = B0[p + jj] * a1[jj] / sumA[i + 1];
                B0[2 * p + jj] = B0[2 * p + jj] * a2[jj] / sumA[i + 2];
                B0[3 * p + jj] = B0[3 * p + jj] * a3[jj] / sumA[i + 3];
            }
        }
        for (; i < r; i++) {
            double a0[BW];
            for (int jj = 0; jj < BW; jj++) a0[jj] = 0.0;
            for (int64_t u = 0; u < n; u++) {
                const double w0 = A[u * r + i];
                const double *wu = whb + u * BW;
                for (int jj = 0; jj < BW; jj++) a0[jj] = MADD(w0, wu[jj], a0[jj]);
            }
            double *B0 = B + i * p + j0;
            for (int jj = 0; jj < jb; jj++) B0[jj] = B0[jj] * a0[jj] / sumA[i];
        }
    }
}

static void transpose(const double *X, int64_t a, int64_t b, double *XT) {
    for (int64_t i = 0; i < a; i++)
        for (int64_t j = 0; j < b; j++) XT[j * a + i] = X[i * b + j];
}

static void pmax(double *X, int64_t len, double lim) {
    for (int64_t t = 0; t < len; t++)
        if (X[t] < lim) X[t] = lim;
}

/* apply(H, 2, which.max) (first maximum, NaN ignored, 0 if all NaN), relabelled by first occurrence */
static void partition(const double *H, int64_t r, int64_t p, int64_t *idx, int64_t *map, int64_t *out) {
    for (int64_t j = 0; j < p; j++) {
        int64_t best = -1;
        double bv = 0.0;
        for (int64_t i = 0; i < r; i++) {
            const double v = H[i * p + j];
            if (v == v && (best < 0 || v > bv)) {
                best = i;
                bv = v;
            }
        }
        idx[j] = best + 1;
    }
    for (int64_t t = 0; t <= r; t++) map[t] = -1;
    int64_t nxt = 0;
    for (int64_t j = 0; j < p; j++) {
        if (map[idx[j]] < 0) map[idx[j]] = nxt++;
        out[j] = map[idx[j]];
    }
}

/* Returns the number of iterations (-1 on allocation failure); *stopped = 1 if the
 * connectivity rule stopped the run. VT (p x n) is only used when update_w. */
int64_t brunet_loop(const double *V, const double *VT, double *W, double *H, int64_t n, int64_t r, int64_t p,
                    int64_t max_iter, int update_w, int64_t stopconv, int64_t check_interval, double eps,
                    int64_t *stopped, int nthreads) {
    const int64_t nm = n > p ? n : p;
#ifndef _OPENMP
    nthreads = 1;
#endif
    if (nthreads < 1) nthreads = 1;
    double *whb = malloc(sizeof(double) * nm * BW * nthreads);
    double *sums = malloc(sizeof(double) * r);
    double *WT = update_w ? malloc(sizeof(double) * r * n) : NULL;
    double *HT = update_w ? malloc(sizeof(double) * p * r) : NULL;
    int64_t *idx = malloc(sizeof(int64_t) * p), *map = malloc(sizeof(int64_t) * (r + 1));
    int64_t *cons = malloc(sizeof(int64_t) * p), *consold = malloc(sizeof(int64_t) * p);
    int64_t it = -1;
    if (!whb || !sums || !idx || !map || !cons || !consold || (update_w && (!WT || !HT))) goto done;
    int have_old = 0, stop = 0;
    int64_t inc = 0;
    it = 0;
    for (;;) {
        if (it > 0 && it % check_interval == 0) {
            partition(H, r, p, idx, map, cons);
            int same = have_old;
            if (same) {
                for (int64_t j = 0; j < p; j++)
                    if (cons[j] != consold[j]) {
                        same = 0;
                        break;
                    }
            }
            if (same) {
                inc++;
            } else {
                memcpy(consold, cons, sizeof(int64_t) * p);
                have_old = 1;
                inc = 0;
            }
            if (inc > stopconv) stop = 1;
        }
        if (stop || it >= max_iter) break;
        it++;
        col_sums(W, n, r, sums);
        mult_update(V, W, H, sums, n, r, p, whb, nthreads);
        if (update_w) {
            transpose(H, r, p, HT);
            transpose(W, n, r, WT);
            col_sums(HT, p, r, sums);
            mult_update(VT, HT, WT, sums, p, r, n, whb, nthreads);
            transpose(WT, r, n, W);
        }
        if (it % 10 == 0) {
            pmax(H, r * p, eps);
            if (update_w) pmax(W, n * r, eps);
        }
    }
    *stopped = stop;
done:
    free(whb);
    free(sums);
    free(WT);
    free(HT);
    free(idx);
    free(map);
    free(cons);
    free(consold);
    return it;
}
