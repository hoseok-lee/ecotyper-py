import numpy as np

from conftest import DATA, hexmatrix, hexread
from ecotyper.nmf import posneg, rnmf
from ecotyper.rcompat.rng import RRandom

S = DATA / "stats_small"


def test_runif_matches_r():
    assert np.array_equal(RRandom(42).runif(1000), hexread(S / "runif_42.txt"))


def test_sample_rejection_sampling():
    ref = np.array([int(x) for x in (S / "sample_1234.txt").read_text().split()])
    assert np.array_equal(RRandom(1234).sample_int(5000, 2500), ref)


def test_sample_continues_stream():
    ref = np.array([int(x) for x in (S / "sample_seq.txt").read_text().split()])
    rng = RRandom(1234)
    got = np.concatenate([rng.sample_int(n, min(500, n)) for n in (37, 600, 1200)])
    assert np.array_equal(got, ref)


def test_permute_rows():
    m = hexread(S / "perm_in.txt", (6, 10))
    out = hexread(S / "perm_out.txt", (6, 10))
    assert np.array_equal(RRandom(101).permute_rows(m), out)


def test_rnmf_initialisation():
    X = hexmatrix(DATA / "nmf_small" / "X.txt")
    W, H = rnmf(4, X, RRandom(123))
    assert np.array_equal(W, hexmatrix(DATA / "nmf_small" / "init_W.txt"))
    assert np.array_equal(H, hexmatrix(DATA / "nmf_small" / "init_H.txt"))


def test_posneg():
    V = hexmatrix(DATA / "nmf_small" / "V_raw.txt")
    assert np.array_equal(posneg(V), hexmatrix(DATA / "nmf_small" / "X.txt"))
