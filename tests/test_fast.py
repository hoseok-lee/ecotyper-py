"""Fast mode (``exact=False``): same algorithm with BLAS NMF updates and no R text round
trips. Not bit-identical to R, but the discrete results (ranks, states, ecotypes) must agree
with the exact mode and the numbers must be close."""
import numpy as np
import pytest

from conftest import R_WORK
from ecotyper import nmf as N
from ecotyper import workflows as wf
from ecotyper.io import read_annotation, read_bulk_matrix
from ecotyper.model import EcoTyperModel

EX = R_WORK / "example_data"
MODELS = R_WORK / "EcoTyper"


def need_examples():
    if not (EX / "bulk_lymphoma_data.txt").exists() or not (MODELS / "Lymphoma").exists():
        pytest.skip("EcoTyper example data / models not available (R working copy)")


def test_nmf_fast_matches_exact():
    rng = np.random.default_rng(1)
    n, p, r = 300, 80, 4
    W = rng.random((n, r)) ** 4
    H = np.zeros((r, p))
    H[rng.integers(0, r, p), np.arange(p)] = 1.0
    V = W @ (H + 0.05 * rng.random((r, p))) + 0.01 * rng.random((n, p))
    a = N.nmf_brunet(V, r, 1235)
    b = N.nmf_brunet(V, r, 1235, exact=False)
    assert np.array_equal(a.predict(), b.predict())
    assert abs(a.niter - b.niter) <= N.CHECK_INTERVAL
    assert np.allclose(a.H, b.H, rtol=1e-8, atol=1e-12 * np.abs(a.H).max())
    assert b.deviance == pytest.approx(a.deviance, rel=1e-9)


def test_recovery_bulk_fast():
    need_examples()
    model = EcoTyperModel.load(MODELS / "Lymphoma" / "Lymphoma_Fractions")
    expr = read_bulk_matrix(EX / "bulk_lymphoma_data.txt")
    ann = read_annotation(EX / "bulk_lymphoma_annotation.txt")
    s1, e1 = wf.recover_bulk_frame(expr, model, ann)
    s2, e2 = wf.recover_bulk_frame(expr, model, ann, exact=False)
    for ct in s1:
        assert s1[ct].assignment.equals(s2[ct].assignment), ct
        assert np.allclose(s1[ct].abundances.to_numpy(), s2[ct].abundances.to_numpy(), rtol=1e-6, atol=1e-9,
                           equal_nan=True), ct
    assert e1.initial_assignment[["ID", "Ecotype", "MaxEcotype"]].equals(
        e2.initial_assignment[["ID", "Ecotype", "MaxEcotype"]])


def test_discovery_profiles_fast():
    need_examples()
    d = EX / "Tutorial_6" / "PresortedDiscovery"
    cts = ["B.cells", "CD8.T.cells", "Fibroblasts", "Monocytes.and.Macrophages"]
    profiles = wf.read_profiles(d, cts)
    ann = read_annotation(EX / "Tutorial_6" / "PresortedDiscovery_annotation.txt")
    kw = dict(max_states=4, restarts=2, n_jobs=4, verbose=False)
    a = wf.discover_from_profiles(profiles, ann, **kw)
    b = wf.discover_from_profiles(profiles, ann, exact=False, **kw)
    assert list(a.ranks["Chosen_Rank"]) == list(b.ranks["Chosen_Rank"])
    for ct, st in a.states.items():
        assert st.state_assignment.equals(b.states[ct].state_assignment), ct
    assert a.ecotypes.ecotypes.equals(b.ecotypes.ecotypes)
