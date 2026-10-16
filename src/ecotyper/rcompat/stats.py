"""R base/stats numerics reproduced to the last bit where practical.

R accumulates ``sum``/``mean``/``colMeans`` in ``long double``; on Apple
silicon (the reference platform) that is the same as ``double`` so a plain
sequential loop reproduces R exactly. ``cov``/``cor``/``var`` follow
``src/library/stats/src/cov.c`` (two-pass means, FMA-contracted cross
products as compiled by clang on arm64).
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy import special
from scipy.stats import hypergeom

from .fma import fma

DBL_EPS = np.finfo(np.float64).eps


# ----------------------------------------------------------------------------
# sums and means
# ----------------------------------------------------------------------------
@nb.njit(cache=True, error_model="numpy")
def _seq_sum(x):
    s = 0.0
    for v in x:
        s += v
    return s


@nb.njit(cache=True, error_model="numpy")
def _r_mean(x):
    n = x.shape[0]
    s = 0.0
    for v in x:
        s += v
    s /= n
    if np.isfinite(s):
        t = 0.0
        for v in x:
            t += v - s
        s += t / n
    return s


def r_sum(x) -> float:
    """``sum(x)`` (sequential accumulation)."""
    return float(_seq_sum(np.ascontiguousarray(x, dtype=np.float64).ravel()))


def r_mean(x, na_rm: bool = False) -> float:
    """``mean(x)`` (two-pass, as in summary.c)."""
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    if na_rm:
        x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(_r_mean(x))


@nb.njit(cache=True, error_model="numpy")
def _row_means(X, out):
    for i in range(X.shape[0]):
        out[i] = _r_mean(X[i])


@nb.njit(cache=True, error_model="numpy")
def _row_means_narm(X, out):
    for i in range(X.shape[0]):
        row = X[i]
        cnt = 0
        for v in row:
            if v == v:
                cnt += 1
        if cnt == 0:
            out[i] = np.nan
            continue
        tmp = np.empty(cnt)
        k = 0
        for v in row:
            if v == v:
                tmp[k] = v
                k += 1
        out[i] = _r_mean(tmp)


def apply_mean_rows(X, na_rm: bool = False) -> np.ndarray:
    """``apply(X, 1, mean[, na.rm=TRUE])``."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty(X.shape[0])
    if X.shape[1] == 0:
        out[:] = np.nan
        return out
    (_row_means_narm if na_rm else _row_means)(X, out)
    return out


@nb.njit(cache=True, error_model="numpy")
def _col_means(X, out):
    # colMeans: sum in long double (== double here) then divide
    nr, nc = X.shape
    for j in range(nc):
        s = 0.0
        for i in range(nr):
            s += X[i, j]
        out[j] = s / nr


def col_means(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = np.empty(X.shape[1])
    _col_means(np.ascontiguousarray(X), out)
    return out


def row_means(X) -> np.ndarray:
    """``rowMeans(X)`` (column-by-column accumulation; same per-row order)."""
    return col_means(np.asarray(X, dtype=np.float64).T)


# ----------------------------------------------------------------------------
# var / cor (cov.c)
# ----------------------------------------------------------------------------
@nb.njit(cache=True, error_model="numpy")
def _mean2(x):
    n = x.shape[0]
    s = 0.0
    for v in x:
        s += v
    tmp = s / n
    if np.isfinite(tmp):
        s = 0.0
        for v in x:
            s += v - tmp
        tmp = tmp + s / n
    return tmp


@nb.njit(cache=True, error_model="numpy")
def _var(x):
    n = x.shape[0]
    if n < 2:
        return np.nan
    for v in x:
        if v != v:
            return np.nan
    m = _mean2(x)
    # cov.c (cov_na_1) as compiled by clang on arm64: the first 8*floor(n/8)
    # terms are vectorised (separate multiply, sequential add), the tail is fused
    n8 = (n // 8) * 8
    s = 0.0
    for k in range(n8):
        d = x[k] - m
        s = s + d * d
    for k in range(n8, n):
        d = x[k] - m
        s = fma(d, d, s)
    return s / (n - 1)


@nb.njit(cache=True, error_model="numpy")
def _row_var(X, out):
    for i in range(X.shape[0]):
        out[i] = _var(X[i])


def r_var(x) -> float:
    """``var(x)`` for a numeric vector."""
    return float(_var(np.ascontiguousarray(x, dtype=np.float64).ravel()))


def r_sd(x) -> float:
    return math.sqrt(r_var(x))


def row_var(X) -> np.ndarray:
    """``apply(X, 1, var)``."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty(X.shape[0])
    _row_var(X, out)
    return out


@nb.njit(cache=True, error_model="numpy")
def _cor_cols(X, out):
    """cov_na_1 + cor (use = "everything"): correlation between columns."""
    n, nc = X.shape
    xm = np.empty(nc)
    has_na = np.zeros(nc, dtype=np.bool_)
    for i in range(nc):
        for k in range(n):
            if X[k, i] != X[k, i]:
                has_na[i] = True
                break
        if has_na[i]:
            xm[i] = np.nan
        else:
            xm[i] = _mean2(X[:, i])
    n1 = n - 1
    n8 = (n // 8) * 8
    for i in range(nc):
        if has_na[i]:
            for j in range(i + 1):
                out[j, i] = np.nan
                out[i, j] = np.nan
            continue
        for j in range(i + 1):
            if has_na[j]:
                out[j, i] = np.nan
                out[i, j] = np.nan
                continue
            # vectorised head (mul + add), fused tail: see _var
            s = 0.0
            for k in range(n8):
                s = s + (X[k, i] - xm[i]) * (X[k, j] - xm[j])
            for k in range(n8, n):
                s = fma(X[k, i] - xm[i], X[k, j] - xm[j], s)
            out[j, i] = s / n1
            out[i, j] = out[j, i]
    sd = np.empty(nc)
    for i in range(nc):
        sd[i] = np.sqrt(out[i, i]) if not has_na[i] else np.nan
    for i in range(nc):
        if not has_na[i]:
            for j in range(i):
                if sd[i] == 0 or sd[j] == 0:
                    out[j, i] = np.nan
                    out[i, j] = np.nan
                else:
                    v = out[i, j] / (sd[i] * sd[j])
                    if v > 1.0:
                        v = 1.0
                    elif v < -1.0:
                        v = -1.0
                    out[j, i] = v
                    out[i, j] = v
        out[i, i] = 1.0


def cor_matrix(X) -> np.ndarray:
    """``cor(X)``: Pearson correlation between the columns of ``X``."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty((X.shape[1], X.shape[1]))
    _cor_cols(np.ascontiguousarray(X), out)
    return out


@nb.njit(cache=True, error_model="numpy")
def _cor2(x, y):
    n = x.shape[0]
    for v in x:
        if v != v:
            return np.nan
    for v in y:
        if v != v:
            return np.nan
    xm = _mean2(x)
    ym = _mean2(y)
    n1 = n - 1
    s = 0.0
    for k in range(n):
        s = fma(x[k] - xm, y[k] - ym, s)
    ans = s / n1
    sx = 0.0
    for k in range(n):
        sx = fma(x[k] - xm, x[k] - xm, sx)
    sx /= n1
    sy = 0.0
    for k in range(n):
        sy = fma(y[k] - ym, y[k] - ym, sy)
    sy /= n1
    sx = np.sqrt(sx)
    sy = np.sqrt(sy)
    if sx == 0.0 or sy == 0.0:
        return np.nan
    ans /= (sx * sy)
    if ans > 1.0:
        ans = 1.0
    elif ans < -1.0:
        ans = -1.0
    return ans


def cor_vectors(x, y) -> float:
    """``cor(x, y)`` for two numeric vectors (cov_na_2 path)."""
    return float(_cor2(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)))


def fast_cor(X) -> np.ndarray:
    """``HiClimR::fastCor(X)``: correlation between the columns of ``X`` via
    ``crossprod`` of centred, unit-norm columns (BLAS; ULP-level differences)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - col_means(X)[None, :]
    norm = np.sqrt(np.sum(Xc * Xc, axis=0))
    Xn = Xc / norm[None, :]
    return Xn.T @ Xn


# ----------------------------------------------------------------------------
# scale
# ----------------------------------------------------------------------------
@nb.njit(cache=True, error_model="numpy")
def _scale_vec(x, out):
    """``scale(x)`` for a vector (as a one-column matrix)."""
    n = x.shape[0]
    # colMeans with na.rm = TRUE
    s = 0.0
    cnt = 0
    for v in x:
        if v == v:
            s += v
            cnt += 1
    center = s / cnt if cnt > 0 else np.nan
    for i in range(n):
        out[i] = x[i] - center
    # f(v) = sqrt(sum(v^2) / max(1, length(v) - 1)) over non-NA v
    ss = 0.0
    cnt = 0
    for i in range(n):
        v = out[i]
        if v == v:
            ss += v * v
            cnt += 1
    denom = max(1, cnt - 1)
    sc = np.sqrt(ss / denom)
    for i in range(n):
        out[i] = out[i] / sc


@nb.njit(cache=True, error_model="numpy")
def _scale_rows(X, out):
    for i in range(X.shape[0]):
        _scale_vec(X[i], out[i])


def scale_rows(X) -> np.ndarray:
    """``t(apply(X, 1, scale))``."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty_like(X)
    _scale_rows(X, out)
    return out


def scale_vec(x) -> np.ndarray:
    """``as.vector(scale(x))``."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty_like(x)
    _scale_vec(x, out)
    return out


@nb.njit(cache=True, error_model="numpy")
def _log2p1(X, out):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            out[i, j] = math.log2(X[i, j] + 1.0)


def log2p1(X) -> np.ndarray:
    """``log2(X + 1)`` with libm's log2 (as R calls it), not numpy's SIMD log2."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty_like(X)
    _log2p1(X, out)
    return out


@nb.njit(cache=True, error_model="numpy")
def _col_normalize(X, out, narm):
    nr, nc = X.shape
    for j in range(nc):
        s = 0.0
        for i in range(nr):
            v = X[i, j]
            if narm and v != v:
                continue
            s += v
        for i in range(nr):
            out[i, j] = X[i, j] / s


def col_normalize(X, na_rm: bool = False) -> np.ndarray:
    """``apply(X, 2, function(x) x / sum(x[, na.rm = TRUE]))``."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty_like(X)
    _col_normalize(X, out, na_rm)
    return out


def which_max(x) -> int:
    """``which.max(x)`` → 0-based index of the first maximum ignoring NaN; -1 if none."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    if not ok.any():
        return -1
    xm = np.where(ok, x, -np.inf)
    m = xm.max()
    if m == -np.inf:
        # all non-NA values are -Inf: R returns the first of them
        return int(np.flatnonzero(ok)[0])
    return int(np.flatnonzero(xm == m)[0])


def col_apply_mean(X, na_rm: bool = False) -> np.ndarray:
    """``apply(X, 2, mean[, na.rm = TRUE])``."""
    return apply_mean_rows(np.asarray(X, dtype=np.float64).T, na_rm=na_rm)


# ----------------------------------------------------------------------------
# order statistics
# ----------------------------------------------------------------------------
def quantile7(x, probs, na_rm: bool = False):
    """``quantile(x, probs, type = 7, names = FALSE)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if na_rm:
        x = x[~np.isnan(x)]
    elif np.isnan(x).any():
        raise ValueError("missing values and NaN's not allowed if 'na.rm' is FALSE")
    scalar = np.ndim(probs) == 0
    probs = np.atleast_1d(np.asarray(probs, dtype=np.float64))
    n = x.size
    if n == 0:
        out = np.full(probs.shape, np.nan)
        return float(out[0]) if scalar else out
    xs = np.sort(x)
    index = 1 + max(n - 1, 0) * probs
    lo = np.floor(index).astype(np.int64)
    hi = np.ceil(index).astype(np.int64)
    qs = xs[lo - 1].copy()
    xhi = xs[hi - 1]
    i = (index > lo) & (xhi != qs)
    h = (index - lo)[i]
    qs[i] = (1 - h) * qs[i] + h * xhi[i]
    return float(qs[0]) if scalar else qs


def r_median(x) -> float:
    """``median(x)`` (``mean(sort(x, partial = half + 0:1)[half + 0:1])``)."""
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    if n == 0:
        return np.nan
    half = (n + 1) // 2
    if n % 2 == 1:
        return float(x[half - 1])
    return r_mean(x[half - 1 : half + 1])


def r_mad(x) -> float:
    """``mad(x)`` (constant 1.4826, center = median)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    center = r_median(x)
    return 1.4826 * r_median(np.abs(x - center))


def rank_average(x) -> np.ndarray:
    """``rank(x, ties.method = "average")`` (no NAs)."""
    from scipy.stats import rankdata

    return rankdata(np.asarray(x, dtype=np.float64), method="average")


# ----------------------------------------------------------------------------
# multiple testing
# ----------------------------------------------------------------------------
def p_adjust_bh(p) -> np.ndarray:
    """``p.adjust(p, "BH")`` (NAs kept; n = number of non-NA p-values)."""
    p = np.asarray(p, dtype=np.float64).copy()
    out = p.copy()
    nna = ~np.isnan(p)
    q = p[nna]
    n = q.size
    if n <= 1:
        return out
    i = np.arange(n, 0, -1, dtype=np.float64)
    o = np.argsort(-q, kind="stable")
    ro = np.argsort(o, kind="stable")
    adj = np.minimum(1.0, np.minimum.accumulate((n / i) * q[o]))[ro]
    out[nna] = adj
    return out


# ----------------------------------------------------------------------------
# distributions
# ----------------------------------------------------------------------------
_A = (2.2352520354606839287, 161.02823106855587881, 1067.6894854603709582,
      18154.981253343561249, 0.065682337918207449113)
_B = (47.20258190468824187, 976.09855173777669322, 10260.932208618978205, 45507.789335026729956)
_C = (0.39894151208813466764, 8.8831497943883759412, 93.506656132177855979, 597.27027639480026226,
      2494.5375852903726711, 6848.1904505362823326, 11602.651437647350124, 9842.7148383839780218,
      1.0765576773720192317e-8)
_D = (22.266688044328115691, 235.38790178262499861, 1519.377599407554805, 6485.558298266760755,
      18615.571640885098091, 34900.952721145977266, 38912.003286093271411, 19685.429676859990727)
_P = (0.21589853405795699, 0.1274011611602473639, 0.022235277870649807, 0.001421619193227893466,
      2.9112874951168792e-5, 0.02307344176494017303)
_Q = (1.28426009614491121, 0.468238212480865118, 0.0659881378689285515, 0.00378239633202758244,
      7.29751555083966205e-5)
_M_SQRT_32 = 5.656854249492380195206754896838
_M_1_SQRT_2PI = 0.398942280401432677939946059934


@nb.njit(cache=True, error_model="numpy")
def _pnorm_both(x):
    """R's ``pnorm_both`` (Cody 1993) for log_p = FALSE → (cum, ccum)."""
    a = _A
    b = _B
    c = _C
    d = _D
    p = _P
    q = _Q
    if x != x:
        return x, x
    eps = DBL_EPS * 0.5
    y = abs(x)
    if y <= 0.67448975:
        if y > eps:
            xsq = x * x
            xnum = a[4] * xsq
            xden = xsq
            for i in range(3):
                xnum = (xnum + a[i]) * xsq
                xden = (xden + b[i]) * xsq
        else:
            xnum = 0.0
            xden = 0.0
        temp = x * (xnum + a[3]) / (xden + b[3])
        return 0.5 + temp, 0.5 - temp
    elif y <= _M_SQRT_32:
        xnum = c[8] * y
        xden = y
        for i in range(7):
            xnum = (xnum + c[i]) * y
            xden = (xden + d[i]) * y
        temp = (xnum + c[7]) / (xden + d[7])
        xsq = math.ldexp(math.trunc(math.ldexp(y, 4)), -4)
        dl = (y - xsq) * (y + xsq)
        cum = math.exp(-xsq * math.ldexp(xsq, -1)) * math.exp(-math.ldexp(dl, -1)) * temp
        ccum = 1.0 - cum
        if x > 0.0:
            cum, ccum = ccum, cum
        return cum, ccum
    elif -37.5193 < x < 37.5193:
        # covers both the lower (x < 8.29) and upper (x > -8.29) windows for
        # whichever tail is requested; outside them the tail value is 0/1 below
        xsq = 1.0 / (x * x)
        xnum = p[5] * xsq
        xden = xsq
        for i in range(4):
            xnum = (xnum + p[i]) * xsq
            xden = (xden + q[i]) * xsq
        temp = xsq * (xnum + p[4]) / (xden + q[4])
        temp = (_M_1_SQRT_2PI - temp) / y
        xsq = math.ldexp(math.trunc(math.ldexp(x, 4)), -4)
        dl = (x - xsq) * (x + xsq)
        cum = math.exp(-xsq * math.ldexp(xsq, -1)) * math.exp(-math.ldexp(dl, -1)) * temp
        ccum = 1.0 - cum
        if x > 0.0:
            cum, ccum = ccum, cum
        return cum, ccum
    else:
        if x > 0:
            return 1.0, 0.0
        return 0.0, 1.0


@nb.njit(cache=True, error_model="numpy")
def _pnorm(x, lower):
    cum, ccum = _pnorm_both(x)
    if lower:
        # lower tail valid only for -37.5193 < x < 8.2924 in the 3rd branch
        if abs(x) > _M_SQRT_32 and not (-37.5193 < x < 8.2924):
            return 0.0 if x <= 0 else 1.0
        return cum
    if abs(x) > _M_SQRT_32 and not (-8.2924 < x < 37.5193):
        return 1.0 if x <= 0 else 0.0
    return ccum


def pnorm(x, lower_tail: bool = True):
    """``pnorm(x, lower.tail)`` (standard normal, R's algorithm)."""
    if np.ndim(x) == 0:
        return float(_pnorm(float(x), lower_tail))
    x = np.asarray(x, dtype=np.float64)
    return np.array([_pnorm(float(v), lower_tail) for v in x.ravel()]).reshape(x.shape)


def pt(t: float, df: float, lower_tail: bool = True) -> float:
    """``pt(t, df)`` following R's formulation (pbeta via scipy)."""
    if np.isnan(t) or np.isnan(df):
        return t + df
    if not np.isfinite(t):
        return (0.0 if t < 0 else 1.0) if lower_tail else (1.0 if t < 0 else 0.0)
    if not np.isfinite(df):
        return pnorm(t, lower_tail)
    nx = 1 + (t / df) * t
    if nx > 1e100:
        lval = -0.5 * df * (2 * math.log(abs(t)) - math.log(df)) - special.betaln(0.5 * df, 0.5) - math.log(0.5 * df)
        val = math.exp(lval)
    elif df > t * t:
        val = special.betaincc(0.5, df / 2.0, t * t / (df + t * t))
    else:
        val = special.betainc(df / 2.0, 0.5, 1.0 / nx)
    lower = lower_tail
    if t <= 0.0:
        lower = not lower
    val /= 2.0
    return 0.5 - val + 0.5 if lower else val


class TTestError(ValueError):
    """Raised where R's t.test() stops (caught by EcoTyper's tryCatch)."""


def welch_t_test_p(x, y) -> float:
    """``t.test(x, y)$p.value`` (Welch, two-sided); raises TTestError where R errors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = x.size, y.size
    if nx < 2:
        raise TTestError("not enough 'x' observations")
    if ny < 2:
        raise TTestError("not enough 'y' observations")
    mx = r_mean(x)
    vx = r_var(x)
    my = r_mean(y)
    vy = r_var(y)
    stderrx = math.sqrt(vx / nx)
    stderry = math.sqrt(vy / ny)
    stderr = math.sqrt(stderrx**2 + stderry**2)
    df = stderr**4 / (stderrx**4 / (nx - 1) + stderry**4 / (ny - 1))
    if stderr < 10 * DBL_EPS * max(abs(mx), abs(my)):
        raise TTestError("data are essentially constant")
    tstat = (mx - my - 0) / stderr
    return 2 * pt(-abs(tstat), df)


def phyper_upper(q: float, m: float, n: float, k: float) -> float:
    """``1 - phyper(q, m, n, k)`` as computed by EcoTyper."""
    return 1.0 - float(hypergeom.cdf(q, m + n, m, k))


# ----------------------------------------------------------------------------
# matrixTests::row_wilcoxon_twosample (normal approximation path)
# ----------------------------------------------------------------------------
@nb.njit(cache=True, error_model="numpy")
def _wilcox_rows(X, nx, out_stat, out_ties):
    """Average ranks of each row; returns rank-sum of the first nx entries and
    sum(t^3 - t) over tie groups."""
    nr, nc = X.shape
    for r in range(nr):
        row = X[r]
        order = np.argsort(row, kind="mergesort")
        ranks = np.empty(nc)
        tsum = 0.0
        i = 0
        while i < nc:
            j = i
            while j + 1 < nc and row[order[j + 1]] == row[order[i]]:
                j += 1
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            t = j - i + 1
            if t > 1:
                tsum += float(t) ** 3 - t
            i = j + 1
        s = 0.0
        for k in range(nx):
            s += ranks[k]
        out_stat[r] = s
        out_ties[r] = tsum


def row_wilcoxon_twosample(x: np.ndarray, y: np.ndarray, correct: bool = True) -> np.ndarray:
    """p-values of ``matrixTests::row_wilcoxon_twosample(x, y)`` (two-sided).

    Exact p-values (``pwilcox``) are used when both groups have < 50
    observations and the row has no ties, as in matrixTests; otherwise the
    normal approximation with continuity and tie correction.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.isnan(x).any() or np.isnan(y).any():
        raise NotImplementedError("NA values are not supported")
    nx = x.shape[1]
    ny = y.shape[1]
    X = np.ascontiguousarray(np.hstack([x, y]))
    stat = np.empty(X.shape[0])
    ties = np.empty(X.shape[0])
    _wilcox_rows(X, nx, stat, ties)
    statistic = stat - nx * (nx + 1) * 0.5
    out = np.full(X.shape[0], np.nan)
    if nx < 1 or ny < 1:
        return out
    exact = (nx < 50) and (ny < 50)
    for r in range(X.shape[0]):
        if exact and ties[r] == 0:
            out[r] = _pwilcox_two_sided(statistic[r], nx, ny)
            continue
        z = statistic[r] - nx * ny * 0.5
        corr = (np.sign(z) * 0.5) if correct else 0.0
        z = z - corr
        sigma = math.sqrt((nx * ny / 12) * ((nx + ny + 1) - ties[r] / ((nx + ny) * (nx + ny - 1))))
        z = z / sigma
        out[r] = 2 * min(_pnorm(z, True), _pnorm(z, False))
    return out


def _pwilcox_two_sided(stat, nx, ny):
    # EcoTyper never reaches this branch (both groups have >= 50 observations
    # whenever doDE is called); refuse rather than approximate.
    raise NotImplementedError("exact Wilcoxon p-values (n < 50 per group, no ties) are not ported")
