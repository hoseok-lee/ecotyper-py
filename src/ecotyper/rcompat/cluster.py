"""Hierarchical clustering and silhouettes exactly as in R.

Line-by-line ports of R's ``stats`` Fortran routines ``HCLUST``/``HCASS2``
(``src/library/stats/src/hclust.f``), ``cutree`` (``hclust-utils.c``),
``cophenetic.default``, ``dist(method = "euclidean")`` (``distance.c``) and
``cluster::silhouette.default`` (``sildist.c``). The same loop order and strict
comparisons are kept so that ties are broken identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .fma import fma

METHODS = {"ward.D": 1, "single": 2, "complete": 3, "average": 4, "mcquitty": 5,
           "median": 6, "centroid": 7, "ward.D2": 8}


@nb.njit(cache=True)
def _ioffst(n, i, j):
    # 1-based (i < j) -> 1-based index into the dist vector
    return j + (i - 1) * n - (i * (i + 1)) // 2


@nb.njit(cache=True)
def _hclust(n, iopt, diss, membr):
    INF = 1e300
    ia = np.zeros(n + 1, dtype=np.int64)
    ib = np.zeros(n + 1, dtype=np.int64)
    crit = np.zeros(n + 1)
    nn = np.zeros(n + 1, dtype=np.int64)
    disnn = np.zeros(n + 1)
    flag = np.ones(n + 1, dtype=np.bool_)
    # 1-based views: diss[ind-1]
    im = 0
    jj = 0
    jm = 0
    ncl = n
    if iopt == 8:
        for i in range(diss.shape[0]):
            diss[i] = diss[i] * diss[i]
    for i in range(1, n):
        dmin = INF
        for j in range(i + 1, n + 1):
            ind = _ioffst(n, i, j)
            if dmin > diss[ind - 1]:
                dmin = diss[ind - 1]
                jm = j
        nn[i] = jm
        disnn[i] = dmin
    while True:
        dmin = INF
        for i in range(1, n):
            if flag[i] and disnn[i] < dmin:
                dmin = disnn[i]
                im = i
                jm = nn[i]
        ncl -= 1
        i2 = min(im, jm)
        j2 = max(im, jm)
        ia[n - ncl] = i2
        ib[n - ncl] = j2
        is_ward = iopt == 1 or iopt == 8
        if iopt == 8:
            dmin = np.sqrt(dmin)
        crit[n - ncl] = dmin
        flag[j2] = False
        dmin = INF
        for k in range(1, n + 1):
            if flag[k] and k != i2:
                if i2 < k:
                    ind1 = _ioffst(n, i2, k)
                else:
                    ind1 = _ioffst(n, k, i2)
                if j2 < k:
                    ind2 = _ioffst(n, j2, k)
                else:
                    ind2 = _ioffst(n, k, j2)
                d12 = diss[_ioffst(n, i2, j2) - 1]
                d1 = diss[ind1 - 1]
                d2 = diss[ind2 - 1]
                if is_ward:
                    v = (membr[i2] + membr[k]) * d1 + (membr[j2] + membr[k]) * d2 - membr[k] * d12
                    v = v / (membr[i2] + membr[j2] + membr[k])
                elif iopt == 2:
                    v = min(d1, d2)
                elif iopt == 3:
                    v = max(d1, d2)
                elif iopt == 4:
                    # gfortran: fmadd(MEMBR(I2), DISS(IND1), MEMBR(J2)*DISS(IND2))
                    v = fma(membr[i2], d1, membr[j2] * d2) / (membr[i2] + membr[j2])
                elif iopt == 5:
                    v = (d1 + d2) / 2
                elif iopt == 6:
                    v = ((d1 + d2) - d12 / 2) / 2
                else:
                    v = (fma(membr[i2], d1, membr[j2] * d2)
                         - membr[i2] * membr[j2] * d12 / (membr[i2] + membr[j2])) / (membr[i2] + membr[j2])
                diss[ind1 - 1] = v
                if i2 < k:
                    if v < dmin:
                        dmin = v
                        jj = k
                else:
                    if v < disnn[k]:
                        disnn[k] = v
                        nn[k] = i2
        membr[i2] = membr[i2] + membr[j2]
        disnn[i2] = dmin
        nn[i2] = jj
        for i in range(1, n):
            if flag[i] and (nn[i] == i2 or nn[i] == j2):
                dmin = INF
                for j in range(i + 1, n + 1):
                    if flag[j]:
                        ind = _ioffst(n, i, j)
                        if diss[ind - 1] < dmin:
                            dmin = diss[ind - 1]
                            jj = j
                nn[i] = jj
                disnn[i] = dmin
        if ncl <= 1:
            break
    return ia, ib, crit


@nb.njit(cache=True)
def _hcass2(n, ia, ib):
    iia = np.zeros(n + 1, dtype=np.int64)
    iib = np.zeros(n + 1, dtype=np.int64)
    iorder = np.zeros(n + 2, dtype=np.int64)
    for i in range(1, n + 1):
        iia[i] = ia[i]
        iib[i] = ib[i]
    for i in range(1, n - 1):
        k = min(ia[i], ib[i])
        for j in range(i + 1, n):
            if ia[j] == k:
                iia[j] = -i
            if ib[j] == k:
                iib[j] = -i
    for i in range(1, n):
        iia[i] = -iia[i]
        iib[i] = -iib[i]
    for i in range(1, n):
        if iia[i] > 0 and iib[i] < 0:
            k = iia[i]
            iia[i] = iib[i]
            iib[i] = k
        if iia[i] > 0 and iib[i] > 0:
            k1 = min(iia[i], iib[i])
            k2 = max(iia[i], iib[i])
            iia[i] = k1
            iib[i] = k2
    iorder[1] = iia[n - 1]
    iorder[2] = iib[n - 1]
    loc = 2
    for i in range(n - 2, 0, -1):
        for j in range(1, loc + 1):
            if iorder[j] == i:
                iorder[j] = iia[i]
                if j == loc:
                    loc += 1
                    iorder[loc] = iib[i]
                else:
                    loc += 1
                    for k in range(loc, j + 1, -1):
                        iorder[k] = iorder[k - 1]
                    iorder[j + 1] = iib[i]
                break
    for i in range(1, n + 1):
        iorder[i] = -iorder[i]
    return iia, iib, iorder


@dataclass
class HClust:
    merge: np.ndarray  # (n-1) x 2, R convention (negative = singleton)
    height: np.ndarray
    order: np.ndarray  # 1-based
    labels: list | None = None
    method: str = "complete"


def hclust(d: np.ndarray, n: int, method: str = "complete", labels=None) -> HClust:
    """``hclust(d, method)`` where ``d`` is a ``dist`` vector (as.dist order)."""
    if method == "ward":
        method = "ward.D"
    iopt = METHODS[method]
    if n < 2:
        raise ValueError("must have n >= 2 objects to cluster")
    diss = np.array(d, dtype=np.float64, copy=True)
    membr = np.ones(n + 1)
    ia, ib, crit = _hclust(n, iopt, diss, membr)
    iia, iib, order = _hcass2(n, ia, ib)
    merge = np.column_stack([iia[1:n], iib[1:n]])
    return HClust(merge=merge, height=crit[1:n].copy(), order=order[1 : n + 1].copy(),
                  labels=labels, method=method)


@nb.njit(cache=True)
def _cutree(merge, k):
    n = merge.shape[0] + 1
    sing = np.ones(n + 1, dtype=np.bool_)
    m_nr = np.zeros(n + 1, dtype=np.int64)
    z = np.zeros(n + 1, dtype=np.int64)
    ans = np.zeros(n, dtype=np.int64)
    for kk in range(1, n):
        m1 = merge[kk - 1, 0]
        m2 = merge[kk - 1, 1]
        if m1 < 0 and m2 < 0:
            m_nr[-m1] = kk
            m_nr[-m2] = kk
            sing[-m1] = False
            sing[-m2] = False
        elif m1 < 0 or m2 < 0:
            if m1 < 0:
                j = -m1
                m1 = m2
            else:
                j = -m2
            for l in range(1, n + 1):
                if m_nr[l] == m1:
                    m_nr[l] = kk
            m_nr[j] = kk
            sing[j] = False
        else:
            for l in range(1, n + 1):
                if m_nr[l] == m1 or m_nr[l] == m2:
                    m_nr[l] = kk
        if k == n - kk:
            for l in range(1, n + 1):
                z[l] = 0
            nclust = 0
            for l in range(1, n + 1):
                if sing[l]:
                    nclust += 1
                    ans[l - 1] = nclust
                else:
                    if z[m_nr[l]] == 0:
                        nclust += 1
                        z[m_nr[l]] = nclust
                    ans[l - 1] = z[m_nr[l]]
    if k == n:
        for l in range(1, n + 1):
            ans[l - 1] = l
    return ans


def cutree(tree: HClust, k: int) -> np.ndarray:
    """``cutree(tree, k)`` → 1-based cluster ids in observation order."""
    n = tree.merge.shape[0] + 1
    k = int(k)
    if k < 1 or k > n:
        raise ValueError(f"elements of 'k' must be between 1 and {n}")
    return _cutree(np.ascontiguousarray(tree.merge, dtype=np.int64), k)


def cophenetic(tree: HClust) -> np.ndarray:
    """``cophenetic(tree)`` as a dist vector (as.dist order)."""
    n = len(tree.order)
    out = np.zeros((n, n))
    members: list[list[int]] = []
    for i in range(n - 1):
        a, b = tree.merge[i]
        ids1 = [-a - 1] if a < 0 else members[a - 1]
        ids2 = [-b - 1] if b < 0 else members[b - 1]
        members.append(ids1 + ids2)
        out[np.ix_(ids1, ids2)] = tree.height[i]
    full = out + out.T
    return np.concatenate([full[j + 1 :, j] for j in range(n - 1)])


@nb.njit(cache=True)
def _dist_euclidean_cols(X):
    # dist(t(X)) : distances between columns of X (rows of t(X)); R_euclidean
    nr, nc = X.shape
    out = np.empty(nc * (nc - 1) // 2)
    ij = 0
    for j in range(nc):
        for i in range(j + 1, nc):
            dist = 0.0
            count = 0
            for k in range(nr):
                a = X[k, i]
                b = X[k, j]
                if a == a and b == b:
                    dev = a - b
                    if dev == dev:
                        dist = fma(dev, dev, dist)
                        count += 1
            if count == 0:
                out[ij] = np.nan
            else:
                if count != nr:
                    dist /= (count / nr)
                out[ij] = np.sqrt(dist)
            ij += 1
    return out


def dist_euclidean_rows(X: np.ndarray) -> np.ndarray:
    """``dist(X)`` (Euclidean distances between rows of ``X``)."""
    return _dist_euclidean_cols(np.ascontiguousarray(np.asarray(X, dtype=np.float64).T))


@nb.njit(cache=True)
def _sildist(d, n, clustering, k, ismat):
    diC = np.zeros(n * k)
    counts = np.zeros(k, dtype=np.int64)
    si = np.zeros(n)
    l = 0
    for i in range(n):
        ci = clustering[i] - 1
        counts[ci] += 1
        if ismat:
            l = n * i + i + 1
        for j in range(i + 1, n):
            cj = clustering[j] - 1
            diC[k * i + cj] += d[l]
            diC[k * j + ci] += d[l]
            l += 1
    for i in range(n):
        ki = k * i
        ci = clustering[i] - 1
        compute = True
        for j in range(k):
            if j == ci:
                if counts[j] == 1:
                    compute = False
                else:
                    diC[ki + j] /= (counts[j] - 1)
            else:
                diC[ki + j] /= counts[j]
        ai = diC[ki + ci]
        if ci == 0:
            bi = diC[ki + 1]
        else:
            bi = diC[ki]
        for j in range(1, k):
            if j != ci:
                if bi > diC[ki + j]:
                    bi = diC[ki + j]
        if compute and bi != ai:
            si[i] = (bi - ai) / max(ai, bi) if not (ai != ai or bi != bi) else np.nan
        else:
            si[i] = 0.0
    return si


def silhouette_widths(clustering, d, ismat: bool) -> np.ndarray | None:
    """``cluster::silhouette.default``: widths in observation order, or None (NA)."""
    x = np.asarray(clustering, dtype=np.int64)
    n = len(x)
    ux = np.unique(x)
    k = len(ux)
    if k <= 1 or k >= n:
        return None
    if np.any(ux < 1) or np.any(ux > k):
        x = np.searchsorted(ux, x) + 1
    dd = np.asarray(d, dtype=np.float64)
    dd = dd.ravel(order="F") if ismat else dd
    return _sildist(np.ascontiguousarray(dd), n, x, k, ismat)


def silhouette_avg_width(clustering, dmatrix: np.ndarray) -> float:
    """``summary(silhouette(x, dmatrix = D))$avg.width`` (NaN when R returns NA)."""
    from .stats import r_mean

    s = silhouette_widths(clustering, dmatrix, ismat=True)
    if s is None:
        return np.nan
    return r_mean(s)


def silhouette_avg_width_dist(clustering, d: np.ndarray) -> float:
    """``summary(silhouette(x, d))$avg.width`` for a ``dist`` vector."""
    from .stats import r_mean

    s = silhouette_widths(clustering, d, ismat=False)
    if s is None:
        return np.nan
    return r_mean(s)
