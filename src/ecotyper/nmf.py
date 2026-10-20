"""Port of the parts of the R package NMF (v0.28) that EcoTyper uses.

* ``nmf(x, rank, method="brunet", seed=<int>, .opt="P1")`` → :func:`nmf_brunet`
* ``NMFpredict`` (EcoTyper ``lib/ecotyper.R``) → :func:`nmf_predict`
* ``NMFfitX(list, .merge=TRUE)``, ``cophcor``, ``dispersion``, ``silhouette`` →
  :func:`merge_runs` and friends.

The multiplicative updates replicate NMF's C++ kernels
(``src/divergence.cpp``) operation by operation — same accumulation order and
the fused multiply-adds clang emits for them — so that, given the same initial
matrices, W and H agree with R bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from .rcompat.fma import fma
from .rcompat.rng import RRandom

EPS = np.finfo(np.float64).eps  # .Machine$double.eps
MAX_ITER = 2000  # nmf.getOption('maxIter')
STOPCONV = 40  # nmf.stop.connectivity(stopconv = 40, check.interval = 10)
CHECK_INTERVAL = 10


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------
@nb.njit(cache=True, error_model="numpy")
def _update_h(V, W, H, sumW, wh, tmp):
    """In-place ``divergence_update_H`` (Brunet H step)."""
    n, r = W.shape
    p = H.shape[1]
    for j in range(p):
        # wh[u] = V[u,j] / (W[u,] . H[,j])   (computed when iH == 0 in C)
        for u in range(n):
            s = 0.0
            for k in range(r):
                s = fma(W[u, k], H[k, j], s)
            wh[u] = V[u, j] / s
        for i in range(r):
            tmp[i] = 0.0
        if j == 0:
            for i in range(r):
                sumW[i] = 0.0
        # one accumulator per row of H, each summed over u in C order
        for u in range(n):
            w_u = wh[u]
            for i in range(r):
                tmp[i] = fma(W[u, i], w_u, tmp[i])
            if j == 0:
                for i in range(r):
                    sumW[i] += W[u, i]
        for i in range(r):
            H[i, j] = H[i, j] * tmp[i] / sumW[i]


@nb.njit(cache=True, error_model="numpy")
def _update_w(V, W, H, sumH, wh, tmp):
    """In-place ``divergence_update_W`` (Brunet W step)."""
    n, r = W.shape
    p = H.shape[1]
    for iw in range(n):
        for u in range(p):
            s = 0.0
            for k in range(r):
                s = fma(W[iw, k], H[k, u], s)
            wh[u] = V[iw, u] / s
        for jw in range(r):
            tmp[jw] = 0.0
        if iw == 0:
            for jw in range(r):
                sumH[jw] = 0.0
        for u in range(p):
            w_u = wh[u]
            for jw in range(r):
                tmp[jw] = fma(H[jw, u], w_u, tmp[jw])
            if iw == 0:
                for jw in range(r):
                    sumH[jw] += H[jw, u]
        for jw in range(r):
            W[iw, jw] = W[iw, jw] * tmp[jw] / sumH[jw]


@nb.njit(cache=True, error_model="numpy")
def _pmax_inplace(X, lim):
    n0, n1 = X.shape
    for a in range(n0):
        for b in range(n1):
            if X[a, b] < lim:
                X[a, b] = lim


@nb.njit(cache=True, error_model="numpy")
def _which_max_cols(H, out):
    """``apply(h, 2, which.max)`` (first maximum, NaN ignored); 0 if all NaN."""
    r, p = H.shape
    for j in range(p):
        best = -1
        bv = 0.0
        for i in range(r):
            v = H[i, j]
            if v == v and (best < 0 or v > bv):
                best = i
                bv = v
        out[j] = best + 1


@nb.njit(cache=True, error_model="numpy")
def _canonical(idx, out):
    """Relabel by first occurrence: equal partitions ⇔ equal connectivity."""
    p = idx.shape[0]
    m = np.full(p + 2, -1, dtype=np.int64)
    nxt = 0
    for j in range(p):
        v = idx[j]
        if m[v] < 0:
            m[v] = nxt
            nxt += 1
        out[j] = m[v]


@nb.njit(cache=True, error_model="numpy")
def _brunet_loop(V, W, H, max_iter, update_w, stopconv, check_interval, eps):
    n, r = W.shape
    p = H.shape[1]
    sum_r = np.empty(r)
    tmp = np.empty(r)
    whn = np.empty(n)
    whp = np.empty(p)
    idx = np.empty(p, dtype=np.int64)
    cons = np.empty(p, dtype=np.int64)
    consold = np.empty(p, dtype=np.int64)
    have_old = False  # .consold starts as a zero matrix != any connectivity
    inc = 0
    i = 0
    stop = False
    while True:
        # nmf.stop.connectivity(i)
        if i > 0 and i % check_interval == 0:
            _which_max_cols(H, idx)
            _canonical(idx, cons)
            same = have_old
            if same:
                for j in range(p):
                    if cons[j] != consold[j]:
                        same = False
                        break
            if same:
                inc += 1
            else:
                consold[:] = cons
                have_old = True
                inc = 0
            if inc > stopconv:
                stop = True
        if stop or i >= max_iter:
            break
        i += 1
        _update_h(V, W, H, sum_r, whn, tmp)
        if update_w:
            _update_w(V, W, H, sum_r, whp, tmp)
        if i % 10 == 0:
            _pmax_inplace(H, eps)
            if update_w:
                _pmax_inplace(W, eps)
    return i, stop


@nb.njit(cache=True, error_model="numpy")
def _kl(V, WH):
    """NMF's ``KL_divergence`` (reverse loop order as in distance.cpp)."""
    n, p = V.shape
    dist = 0.0
    for i in range(n - 1, -1, -1):
        for j in range(p - 1, -1, -1):
            x = V[i, j]
            y = WH[i, j]
            if x == 0:
                dev = y
            elif x == x and y == y:
                # clang emits fnmsub: fma(x, log(x/y), -x), then + y
                dev = fma(x, np.log(x / y), -x) + y
            else:
                return np.nan
            if np.isfinite(dev):
                dist += dev
            else:
                return dev
    return dist


# ----------------------------------------------------------------------------
# Python API
# ----------------------------------------------------------------------------
def posneg(x: np.ndarray) -> np.ndarray:
    """``NMF::posneg``: ``rbind(pmax(x, 0), pmax(-x, 0))`` if any entry < 0."""
    x = np.asarray(x, dtype=np.float64)
    if x.min() < 0:
        return np.vstack([np.where(x < 0, 0.0, x), np.where(-x < 0, 0.0, -x)])
    return x.copy()


def rnmf(rank: int, target: np.ndarray, rng: RRandom) -> tuple[np.ndarray, np.ndarray]:
    """``rnmf(rank, target)``: W (n×r) then H (r×p) ~ U(0, max(max(target), 1)),
    clamped with ``pmin(., max(abs(target)))``."""
    n, p = target.shape
    finite = target[~np.isnan(target)]
    upper = max(float(finite.max()), 1.0)
    W = rng.runif(n * rank, 0.0, upper).reshape((n, rank), order="F")
    H = rng.runif(rank * p, 0.0, upper).reshape((rank, p), order="F")
    if finite.size:
        m = float(np.abs(finite).max())
        W = np.minimum(W, m)
        H = np.minimum(H, m)
    return W, H


@dataclass
class NMFFit:
    """Result of a single NMF run (``NMFfit``)."""

    W: np.ndarray
    H: np.ndarray
    niter: int
    converged: bool
    deviance: float = np.nan
    seed: int | None = None

    def predict(self) -> np.ndarray:
        """``predict(fit)``: 1-based index of the dominant row of H per column."""
        return predict_columns(self.H)

    def connectivity(self) -> np.ndarray:
        cl = self.predict()
        return (cl[:, None] == cl[None, :]).astype(np.float64)


def _check_target(V: np.ndarray) -> None:
    if np.nanmin(V) < 0:
        raise ValueError("NMF::nmf - Input matrix contains some negative entries.")
    if np.nanmin(np.nansum(V, axis=1)) == 0:
        raise ValueError("NMF::nmf - Input matrix contains at least one null or NA-filled row.")


def _brunet_loop_fast(V, W, H, max_iter, update_w, stopconv, check_interval, eps):
    """``exact=False``: the same algorithm as :func:`_brunet_loop` (Brunet updates, eps clamp
    every 10 iterations, connectivity stop) with BLAS matrix products. The sums are accumulated
    in a different order, so W and H differ from R's in the last bits."""
    n, r = W.shape
    p = H.shape[1]
    WH = np.empty((n, p))
    rt = np.empty((r, p))
    wt = np.empty((n, r))
    consold = None
    inc = 0
    i = 0
    stop = False
    while True:
        if i > 0 and i % check_interval == 0:
            idx = np.argmax(H, axis=0)  # first maximum, as which.max (H has no NaN here)
            # equal partitions <=> equal labels after relabelling by first occurrence
            _, first, inv = np.unique(idx, return_index=True, return_inverse=True)
            lab = np.argsort(np.argsort(first))[inv]
            if consold is not None and np.array_equal(lab, consold):
                inc += 1
            else:
                consold = lab
                inc = 0
            if inc > stopconv:
                stop = True
        if stop or i >= max_iter:
            break
        i += 1
        # H <- H * (t(W) %*% (V / (W %*% H))) / colSums(W)
        np.matmul(W, H, out=WH)
        np.divide(V, WH, out=WH)
        np.matmul(W.T, WH, out=rt)
        H *= rt
        H /= W.sum(axis=0)[:, None]
        if update_w:
            # W <- W * ((V / (W %*% H)) %*% t(H)) / rowSums(H)
            np.matmul(W, H, out=WH)
            np.divide(V, WH, out=WH)
            np.matmul(WH, H.T, out=wt)
            W *= wt
            W /= H.sum(axis=1)[None, :]
        if i % 10 == 0:
            np.maximum(H, eps, out=H)
            if update_w:
                np.maximum(W, eps, out=W)
    return i, stop


def run_brunet(V, W0, H0, *, update_w=True, max_iter=MAX_ITER, exact: bool = True,
               threads: int = 1) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Run Brunet multiplicative updates from (W0, H0) with the connectivity stop.

    ``exact=False`` uses BLAS products (:func:`_brunet_loop_fast`) instead of the R-exact kernels.
    ``threads``: OpenMP threads for the C kernel (same result for any value)."""
    V = np.ascontiguousarray(V, dtype=np.float64)
    W = np.ascontiguousarray(W0, dtype=np.float64).copy()
    H = np.ascontiguousarray(H0, dtype=np.float64).copy()
    if exact:
        from ._ckernel import nmf_lib

        lib = nmf_lib()
        if lib is not None:
            return _run_c(lib, V, W, H, bool(update_w), int(max_iter), threads)
    loop = _brunet_loop if exact else _brunet_loop_fast
    niter, stop = loop(V, W, H, int(max_iter), bool(update_w), STOPCONV, CHECK_INTERVAL, EPS)
    return W, H, int(niter), bool(stop)


def _run_c(lib, V, W, H, update_w, max_iter, threads=1):
    """:func:`_brunet_loop` in ``_nmfk.c`` (same operations, vectorised across columns)."""
    import ctypes as C

    n, r = W.shape
    p = H.shape[1]
    VT = np.ascontiguousarray(V.T) if update_w else V
    dp = C.POINTER(C.c_double)
    stopped = C.c_int64(0)
    niter = lib.brunet_loop(V.ctypes.data_as(dp), VT.ctypes.data_as(dp), W.ctypes.data_as(dp),
                            H.ctypes.data_as(dp), n, r, p, max_iter, int(update_w), STOPCONV, CHECK_INTERVAL, EPS,
                            C.byref(stopped), int(threads))
    if niter < 0:
        raise MemoryError("brunet_loop: allocation failed")
    return W, H, int(niter), bool(stopped.value)


def fitted(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """``W %*% H`` via BLAS dgemm called as R does (column-major, no transposes)."""
    from scipy.linalg import blas

    return blas.dgemm(1.0, np.asfortranarray(W, dtype=np.float64), np.asfortranarray(H, dtype=np.float64))


def kl_divergence(V: np.ndarray, W: np.ndarray, H: np.ndarray, exact: bool = True) -> float:
    if exact:
        return float(_kl(np.ascontiguousarray(V, dtype=np.float64), np.ascontiguousarray(fitted(W, H))))
    WH = W @ H
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(V == 0, WH, V * np.log(V / WH) - V + WH)
    return float(dev.sum())


def nmf_brunet(V: np.ndarray, rank: int, seed: int, *, max_iter: int = MAX_ITER, exact: bool = True) -> NMFFit:
    """``nmf(V, rank, method="brunet", seed=seed, .opt="P1")`` (single run).

    ``exact=False``: same initialisation and stopping rule, BLAS updates (not bit-identical to R)."""
    V = np.asarray(V, dtype=np.float64)
    _check_target(V)
    rng = RRandom(seed)
    W0, H0 = rnmf(rank, V, rng)
    W, H, niter, conv = run_brunet(V, W0, H0, update_w=True, max_iter=max_iter, exact=exact)
    return NMFFit(W=W, H=H, niter=niter, converged=conv, deviance=kl_divergence(V, W, H, exact), seed=seed)


NMF_METHODS = ("brunet", "cd")


def nmf_cd(V: np.ndarray, rank: int, seed: int, *, max_iter: int = MAX_ITER, tol: float = 1e-4) -> NMFFit:
    """Alternative solver (``nmf_method="cd"``, **not** EcoTyper/R-equivalent): Frobenius-loss
    NMF by coordinate descent (scikit-learn ``NMF(solver="cd")``), started from the same
    ``rnmf`` initialisation as :func:`nmf_brunet` and stopped by scikit-learn's tolerance
    rule instead of the connectivity rule. ``deviance`` is the Frobenius reconstruction error
    (used to pick the best restart)."""
    import warnings

    try:
        from sklearn.decomposition import NMF
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as e:  # pragma: no cover
        raise ImportError('nmf_method="cd" needs scikit-learn (pip install "ecotyper[sklearn]")') from e
    V = np.asarray(V, dtype=np.float64)
    _check_target(V)
    W0, H0 = rnmf(rank, V, RRandom(seed))
    m = NMF(n_components=rank, init="custom", solver="cd", beta_loss="frobenius", max_iter=max_iter, tol=tol,
            shuffle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W = m.fit_transform(V, W=np.ascontiguousarray(W0), H=np.ascontiguousarray(H0))
    return NMFFit(W=np.ascontiguousarray(W), H=np.ascontiguousarray(m.components_), niter=int(m.n_iter_),
                  converged=int(m.n_iter_) < max_iter, deviance=float(m.reconstruction_err_), seed=seed)


def nmf_predict(W: np.ndarray, new_data: np.ndarray, rng: RRandom, *, max_iter: int = MAX_ITER,
                exact: bool = True, threads: int = 1):
    """EcoTyper's ``NMFpredict(W, new_data)``.

    ``W`` has ``2g`` rows (``<gene>__pos`` then ``<gene>__neg``) and ``new_data``
    is the (scaled) ``g × p`` matrix already aligned to the training genes. Rows
    of the posneg-transformed data with zero variance are dropped from both,
    an initial H is drawn with ``rnmf`` (after an unused W draw, as in R), and
    only H is updated. Returns ``(H, niter, keep_rows)``.
    """
    from .rcompat.stats import row_var

    to_predict = posneg(np.asarray(new_data, dtype=np.float64))
    keep = row_var(to_predict) > 0
    ws = np.asarray(W, dtype=np.float64)[keep]
    to_predict = to_predict[keep]
    _, H0 = rnmf(W.shape[1], to_predict, rng)
    _check_target(to_predict)
    _, H, niter, _ = run_brunet(to_predict, ws, H0, update_w=False, max_iter=max_iter, exact=exact,
                                threads=threads)
    return H, niter, keep


def predict_columns(H: np.ndarray) -> np.ndarray:
    """``.predict.nmf(t(H))``: ``which.max(abs(h))`` per column (1-based)."""
    out = np.empty(H.shape[1], dtype=np.int64)
    _which_max_cols(np.abs(np.ascontiguousarray(H)), out)
    return out


def factor_codes(values: np.ndarray) -> np.ndarray:
    """Codes of ``as.factor(values)`` for positive integers (levels sorted numerically)."""
    levels = np.unique(values)
    return np.searchsorted(levels, values) + 1


# ----------------------------------------------------------------------------
# Multiple runs
# ----------------------------------------------------------------------------
@dataclass
class NMFMerged:
    """``NMFfitX1``: best fit + consensus matrix over restarts."""

    fit: NMFFit
    consensus: np.ndarray
    nrun: int
    runs_used: list = field(default_factory=list)


def merge_runs(fits: list[NMFFit]) -> NMFMerged:
    """``NMFfitX(list_of_fits, .merge=TRUE)``."""
    consensus = None
    best = None
    best_res = np.inf
    for f in fits:
        con = f.connectivity()
        consensus = con.copy() if consensus is None else consensus + con
        if f.deviance < best_res:
            best = f
            best_res = f.deviance
    return NMFMerged(fit=best, consensus=consensus / len(fits), nrun=len(fits))


def cophcor(consensus: np.ndarray) -> float:
    """``cophcor(consensus)`` with average linkage."""
    from .rcompat.cluster import cophenetic, hclust
    from .rcompat.stats import cor_vectors

    C = np.asarray(consensus, dtype=np.float64)
    iu = np.triu_indices(C.shape[0], 1)
    if np.all(C[iu] == 0) and np.all(np.diag(C) == C[0, 0]):
        return 1.0
    D = 1.0 - C
    d = _as_dist(D)
    hc = hclust(d, C.shape[0], "average")
    return cor_vectors(d, cophenetic(hc))


def dispersion(consensus: np.ndarray) -> float:
    from .rcompat.stats import r_sum

    C = np.asarray(consensus, dtype=np.float64)
    return r_sum((4 * (C - 0.5) ** 2).ravel(order="F")) / C.shape[0] ** 2


def _as_dist(D: np.ndarray) -> np.ndarray:
    """``as.dist(D)``: lower triangle, column by column."""
    n = D.shape[0]
    return np.concatenate([D[j + 1 :, j] for j in range(n - 1)]) if n > 1 else np.empty(0)


def silhouettes(merged: NMFMerged) -> dict:
    """Average silhouette widths as in ``state_discovery_combine_NMF_restarts.R``:
    ``samples``, ``features``, ``chc``, ``consensus``."""
    from .rcompat.cluster import cutree, hclust, silhouette_avg_width
    from .rcompat.stats import cor_matrix

    fit = merged.fit
    out = {}
    # samples: clusters from H, dissimilarity 1 - cor between samples (columns of H)
    p = predict_columns(fit.H)
    out["samples"] = silhouette_avg_width(factor_codes(p), 1.0 - cor_matrix(np.abs(fit.H).T.T))
    # features: clusters from W rows, dissimilarity 1 - cor between features
    pf = predict_columns(fit.W.T)
    out["features"] = silhouette_avg_width(factor_codes(pf), 1.0 - cor_matrix(fit.W.T))
    # chc / consensus: average-linkage tree on 1 - consensus cut at rank
    C = merged.consensus
    hc = hclust(_as_dist(1.0 - C), C.shape[0], "average")
    cl = cutree(hc, fit.W.shape[1])
    dm = 1.0 - C
    out["chc"] = silhouette_avg_width(cl, dm)
    # 'consensus': same clusters (relabelled), widths reordered by the dendrogram
    # order of reorder(as.dendrogram(hc), rowMeans(consensus)) before averaging
    from .rcompat.cluster import silhouette_widths
    from .rcompat.stats import r_mean, row_means

    w = silhouette_widths(cl, dm, ismat=True)
    if w is None:
        out["consensus"] = np.nan
    else:
        o = reorder_dendrogram_order(hc.merge, row_means(C))
        out["consensus"] = r_mean(w[o - 1])
    return out


def reorder_dendrogram_order(merge: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """``order.dendrogram(reorder(as.dendrogram(hc), wts))`` (1-based leaves)."""
    n = merge.shape[0] + 1
    nodes: dict[int, tuple] = {}

    def child(x):
        if x < 0:
            return ("leaf", -x, float(wts[-x - 1]))
        return nodes[x]

    for k in range(n - 1):
        a, b = merge[k]
        ca, cb = child(a), child(b)
        vals = [ca[-1], cb[-1]]
        kids = [ca, cb]
        if vals[1] < vals[0]:  # sort.list (stable)
            kids = [cb, ca]
            vals = [vals[1], vals[0]]
        nodes[k + 1] = ("node", kids, (0.0 + vals[0]) + vals[1])

    # iterative depth-first walk (degenerate chain-like trees exceed the recursion limit)
    acc: list[int] = []
    stack = [nodes[n - 1]]
    while stack:
        nd = stack.pop()
        if nd[0] == "leaf":
            acc.append(nd[1])
        else:
            stack.extend(reversed(nd[1]))
    return np.array(acc, dtype=np.int64)
