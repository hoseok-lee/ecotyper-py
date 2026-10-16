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
