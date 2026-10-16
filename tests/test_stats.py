import numpy as np

from conftest import DATA, hexread
from ecotyper.rcompat.io import format_real
from ecotyper.rcompat.stats import (p_adjust_bh, phyper_upper, quantile7, r_mad, r_median,
                                    row_wilcoxon_twosample, welch_t_test_p)

S = DATA / "stats_small"


def test_write_table_number_formatting():
    vals = hexread(S / "fmt_in.txt")
    ref = (S / "fmt_out.txt").read_text().split("\n")[: len(vals)]
    assert [format_real(float(v)) for v in vals] == ref


def test_welch_t_test():
    x = hexread(S / "tt_x.txt", (40, 30))
    p = np.array([welch_t_test_p(x[:12, j], x[12:, j]) for j in range(30)])
    ref = hexread(S / "tt_p.txt")
    # scipy's regularised incomplete beta (Boost) vs R's TOMS 708: last-digit differences
    np.testing.assert_allclose(p, ref, rtol=1e-13, atol=0)


def test_p_adjust_bh():
    p = hexread(S / "tt_p.txt")
    got = p_adjust_bh(np.concatenate([p, [np.nan], p[:5]]))
    ref = hexread(S / "bh.txt")
    np.testing.assert_allclose(got, ref, rtol=1e-15, atol=0, equal_nan=True)


def test_phyper_upper_tail():
    got = np.array([phyper_upper(q, 25, 75, 20) for q in range(21)])
    np.testing.assert_allclose(got, hexread(S / "phyper.txt"), rtol=1e-9, atol=1e-15)


def test_quantile_median_mad():
    y = hexread(S / "q_x.txt")
    probs = 0.1 + np.arange(19) * 0.05
    assert np.array_equal(quantile7(y, probs), hexread(S / "q_out.txt"))
    assert np.array_equal(np.array([r_median(y), r_mad(y), r_median(y[1:]), r_mad(y[1:])]), hexread(S / "medmad.txt"))


def test_row_wilcoxon_normal_approximation():
    a = hexread(S / "wx_in.txt", (50, 130))
    p = row_wilcoxon_twosample(a[:, :60], a[:, 60:])
    np.testing.assert_allclose(p, hexread(S / "wx_p.txt"), rtol=1e-12, atol=0)
