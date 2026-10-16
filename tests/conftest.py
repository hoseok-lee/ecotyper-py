import os
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[1]
# R reference outputs (text) and the R working copy with the example data;
# see validation/r_reference/run_r_reference.sh
R_REF = Path(os.environ.get("ECOTYPER_R_REF", ROOT / "work" / "r_ref"))
R_WORK = Path(os.environ.get("ECOTYPER_R_WORK", "/tmp/ecotyper_ref"))
PY_OUT = Path(os.environ.get("ECOTYPER_PY_OUT", ROOT / "work" / "py_test_out"))


def hexread(path, shape=None, order="F"):
    vals = [float.fromhex(t) if t not in ("NA", "NaN") else np.nan
            for line in Path(path).read_text().split("\n") if line for t in line.split("\t")]
    a = np.array(vals)
    if shape is not None:
        a = a.reshape(shape, order=order)
    return a


def hexmatrix(path):
    rows = [line.split("\t") for line in Path(path).read_text().split("\n") if line]
    return np.array([[float.fromhex(t) for t in r] for r in rows])


def need(path):
    if not Path(path).exists():
        pytest.skip(f"R reference not available: {path}")
