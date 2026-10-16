"""End-to-end parity with the R pipeline on the README tutorials.

Recovery tutorials run in a few minutes; discovery tutorials (t4/t5/t6) take
long and run only with ECOTYPER_RUN_SLOW=1 (otherwise existing outputs in
$ECOTYPER_PY_OUT are compared, if present)."""
import os
import sys
from pathlib import Path

import pytest

from conftest import PY_OUT, R_REF, R_WORK

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import tutorials  # noqa: E402

pytestmark = pytest.mark.e2e

# p-values from t.test differ in the last digits (scipy's incomplete beta vs R's TOMS 708)
PVAL_FILES = ("assignment_p_vals.txt", "ecotype_assignment.txt", "initial_ecotype_assignment.txt")
RTOL = 1e-12


def _check(tut, run):
    if not (R_WORK / "example_data").exists() or not all((R_REF / r).exists() for _, r in tutorials.PAIRS[tut]):
        pytest.skip("R reference outputs / example data not available")
    out = PY_OUT
    if run:
        tutorials.run(tut, out, int(os.environ.get("ECOTYPER_JOBS", "4")))
    elif not all((out / p).exists() for p, _ in tutorials.PAIRS[tut]):
        pytest.skip("python outputs not present; set ECOTYPER_RUN_SLOW=1 to run")
    df = tutorials.compare(tut, out)
    assert not df["missing"].any(), df[df["missing"]]["file"].tolist()
    assert df["shape_equal"].all(), df[~df["shape_equal"]]["file"].tolist()
    assert df["discrete_equal"].all(), df[~df["discrete_equal"]]["file"].tolist()
    assert (df["max_rel"] <= RTOL).all(), df[df["max_rel"] > RTOL][["file", "max_rel"]]
    not_identical = df[~df["identical_bytes"]]
    assert all(f.endswith(PVAL_FILES) for f in not_identical["file"]), not_identical["file"].tolist()
    return df


@pytest.mark.parametrize("tut", ["t1_lung", "t1_lymphoma", "t2_lymphoma", "t2_crc", "t3_visium"])
def test_recovery_tutorials(tut):
    _check(tut, run=True)


@pytest.mark.parametrize("tut", ["t6_presorted", "t4_bulk", "t5_scrna"])
def test_discovery_tutorials(tut):
    _check(tut, run=os.environ.get("ECOTYPER_RUN_SLOW") == "1")
