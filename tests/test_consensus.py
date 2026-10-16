import numpy as np
import pytest

from conftest import DATA, hexmatrix
from ecotyper.nmf import cophcor, dispersion, merge_runs, nmf_brunet, silhouettes
from ecotyper.rcompat.cluster import cutree, dist_euclidean_rows, hclust, silhouette_avg_width_dist
from ecotyper.rcompat.stats import cor_matrix

C = DATA / "consensus_small"


@pytest.fixture(scope="module")
def runs():
    X2 = hexmatrix(DATA / "nmf_small" / "X2.txt")
    return {r: merge_runs([nmf_brunet(X2, r, 1234 + k) for k in range(1, 6)]) for r in range(2, 7)}


def test_consensus_and_rank_metrics(runs):
    ref = hexmatrix(C / "rank_metrics.txt")
    for row in ref:
        r = int(row[0])
        m = runs[r]
        assert np.array_equal(m.consensus, hexmatrix(C / f"cons_r{r}.txt"))
        assert np.array_equal(m.fit.W, hexmatrix(C / f"bestW_r{r}.txt"))
        sil = silhouettes(m)
        got = [cophcor(m.consensus), sil["samples"], sil["features"], sil["chc"], sil["consensus"],
               dispersion(m.consensus)]
        np.testing.assert_array_equal(np.array(got), row[1:7])
        pred = [int(x) for x in (C / f"predict_r{r}.txt").read_text().split()]
        assert list(m.fit.predict()) == pred


def test_hclust_average_with_ties():
    Cm = hexmatrix(C / "C_ties.txt")
    n = Cm.shape[0]
    D = 1 - Cm
    d = np.concatenate([D[j + 1:, j] for j in range(n - 1)])
    hc = hclust(d, n, "average")
    ref = [line.split("\t") for line in (C / "hc_avg.txt").read_text().split("\n") if line]
    assert np.array_equal(hc.merge, np.array([[int(a), int(b)] for a, b, _ in ref]))
    assert np.array_equal(hc.height, np.array([float.fromhex(h) for _, _, h in ref]))
    assert list(hc.order) == [int(x) for x in (C / "hc_avg_order.txt").read_text().split()]
    cuts = (C / "hc_avg_cutree.txt").read_text().split()
    for k, line in zip(range(2, 11), cuts):
        assert list(cutree(hc, k)) == [int(x) for x in line.split(",")]
    sil = float.fromhex((C / "sil_ties.txt").read_text().strip())
    assert silhouette_avg_width_dist(cutree(hc, 4), d) == sil


def test_dist_cor_complete_linkage():
    A = hexmatrix(C / "A.txt")
    D = dist_euclidean_rows(A.T)
    full = hexmatrix(C / "distA.txt")
    n = full.shape[0]
    assert np.array_equal(D, np.concatenate([full[j + 1:, j] for j in range(n - 1)]))
    assert np.array_equal(cor_matrix(A), hexmatrix(C / "corA.txt"))
    hc = hclust(D, n, "complete")
    assert list(hc.order) == [int(x) for x in (C / "hc_complete_order.txt").read_text().split()]
