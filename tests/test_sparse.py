import pytest

import numpy as np
import scipy.sparse as sp
import pandas as pd

from ecotyper.utils import (
    PosNeg,
    sparse_corrcoef,
    sparse_rmse
)


np.random.seed(0)

def test_posneg_matrix():

    pos_only = sp.csr_matrix(
        np.random.randint(
            0, 2,
            size = (100, 100)
        )
    )

    A = PosNeg(pos_only)

    assert(A.is_pos_only())
    # .X attribute should return the original matrix
    assert((A.X != pos_only).nnz == 0)
    assert(A.neg.sum() == 0)

    normal_dist = sp.csr_matrix(
        np.random.normal(
            loc = 0,
            scale = 4,
            size = (100, 100)
        )
    )

    B = PosNeg(normal_dist)

    assert(B.X.shape[0] == normal_dist.shape[0] * 2)
    assert(
        # Floating point issues
        B.pos.sum() - B.neg.sum() == \
            pytest.approx(normal_dist.sum())
    )


def test_sparse_corrcoef():

    A = sp.csr_matrix(
        np.random.normal(
            loc = 0,
            scale = 4,
            size = (100, 100)
        )
    )

    df = pd.DataFrame.sparse.from_spmatrix(A)

    assert(
        # Floating point issues
        np.all(
            sparse_corrcoef(A) ==
            pytest.approx(df.corr().values)
        )
    )


def test_sparse_rmse():

    from sklearn.metrics import mean_squared_error

    A = np.random.normal((100, 100))
    B = np.random.normal((100, 100))

    assert(
        sparse_rmse(sp.csr_matrix(A), sp.csr_matrix(B)) == \
        np.sqrt(mean_squared_error(A, B))
    )