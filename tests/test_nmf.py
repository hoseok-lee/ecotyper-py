import numpy as np
import pytest

from conftest import DATA, hexmatrix
from ecotyper.nmf import nmf_brunet

N = DATA / "nmf_small"


@pytest.mark.parametrize("it", [1, 2, 10, 11, 25])
def test_brunet_iterations_bit_exact(it):
    X = hexmatrix(N / "X.txt")
    fit = nmf_brunet(X, 4, 123, max_iter=it)
    assert np.array_equal(fit.W, hexmatrix(N / f"W_it{it}.txt"))
    assert np.array_equal(fit.H, hexmatrix(N / f"H_it{it}.txt"))


def test_brunet_full_run():
    X = hexmatrix(N / "X.txt")
    fit = nmf_brunet(X, 4, 123)
    niter, dev = (N / "full_info.txt").read_text().split()
    assert fit.niter == int(niter)
    assert fit.deviance == float.fromhex(dev)
    assert np.array_equal(fit.W, hexmatrix(N / "W_full.txt"))
    assert np.array_equal(fit.H, hexmatrix(N / "H_full.txt"))


@pytest.mark.parametrize("seed", [1235, 1236, 1237])
def test_brunet_discovery_like(seed):
    X2 = hexmatrix(N / "X2.txt")
    fit = nmf_brunet(X2, 5, seed)
    niter, dev = (N / f"info2_{seed}.txt").read_text().split()
    assert fit.niter == int(niter)
    assert fit.deviance == float.fromhex(dev)
    assert np.array_equal(fit.W, hexmatrix(N / f"W2_{seed}.txt"))
    assert np.array_equal(fit.H, hexmatrix(N / f"H2_{seed}.txt"))


@pytest.mark.parametrize("update_w", [True, False])
def test_c_kernel_matches_numba(update_w):
    """_nmfk.c (vectorised) and the numba kernels perform the same operations: bit-identical."""
    from ecotyper import nmf
    from ecotyper._ckernel import nmf_lib

    if nmf_lib() is None:
        pytest.skip("no C compiler")
    rng = np.random.default_rng(7)
    for n, p, r, it in [(1, 1, 1, 5), (37, 70, 6, 120), (101, 33, 4, 60), (64, 65, 9, 40), (300, 10, 3, 400)]:
        V = np.abs(rng.standard_normal((n, p)))
        V[rng.random((n, p)) < 0.3] = 0
        V[:, 0] += 1e-3
        V[0] += 1e-3
        W0, H0 = rng.random((n, r)) * 3, rng.random((r, p)) * 3
        Wa, Ha = W0.copy(), H0.copy()
        ia, sa = nmf._brunet_loop(V, Wa, Ha, it, update_w, nmf.STOPCONV, nmf.CHECK_INTERVAL, nmf.EPS)
        Wb, Hb, ib, sb = nmf._run_c(nmf_lib(), V, W0.copy(), H0.copy(), update_w, it)
        assert (ia, sa) == (ib, sb)
        assert np.array_equal(Wa, Wb) and np.array_equal(Ha, Hb), (n, p, r)
