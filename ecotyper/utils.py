import numpy as np
import scipy.sparse as sp
import anndata as ad


class PosNeg:

    def __init__(self, X: sp.spmatrix):

        self._X = sp.csr_matrix(X)

        # Positive portion
        self._pos = self._X.copy()
        self._pos[self._pos < 0] = 0
        self._pos.eliminate_zeros()

        # Negative portion
        self._neg = self._X.copy()
        self._neg[self._neg > 0] = 0
        self._neg = abs(self._neg)
        self._neg.eliminate_zeros()


    @property
    def X(self):

        # No negative values found, return original
        if self.is_pos_only():
            return self._X

        else:
            return sp.vstack([ self._pos, self._neg ])

    @property
    def pos(self):
        return self._pos

    @property
    def neg(self):
        return self._neg

    def is_pos_only(self):
        return self._neg.nnz == 0


# https://stackoverflow.com/questions/19231268/correlation-coefficients-for-sparse-matrix-in-python
def sparse_corrcoef(A: sp.spmatrix) -> sp.spmatrix:

    N = A.shape[0]

    # Compute the covariance matrix
    C = (
        (A.T * A -(sum(A).T * sum(A) / N)) /
        (N - 1)
    ).todense()

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    V = np.sqrt(
        np.diag(C).reshape(1, -1).T *
        np.diag(C).reshape(1, -1)
    )

    coeffs = np.divide(C, V + 1e-119)

    return sp.csc_matrix(coeffs)


def sparse_dropout(A: sp.spmatrix) -> np.ndarray:

    # Ensure column sparseness
    A = A.tocsc()
    dropout = np.zeros(A.shape[1])

    for i in range(A.shape[1]):

        sp_col = A.getcol(i)

        # Produce frequency taking advantage of sparsity
        counts = np.append(
            np.unique(
                sp_col.data,
                return_counts = True
            # We only care about the frequency, not the counts themeselves
            )[1],
            # Count number of zeros
            max(sp_col.shape) - sp_col.nnz
        )

        dropout[i] = max(counts)

    return dropout / A.shape[0]


def sample_top(
    adata: ad.AnnData,
    by: str,
    n_cells: int,
    random_state: int = 0
) -> ad.AnnData:

    # Return original object if set to 0
    if n_cells == 0:
        return adata

    idx = adata.obs \
        .sample(frac = 1, random_state = random_state) \
        .groupby(by) \
        .head(n_cells) \
        .index

    return adata[idx].copy()
