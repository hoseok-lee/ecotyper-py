import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import scanpy as sc

from tqdm import tqdm
from multiprocess import Pool



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
def sparse_corrcoef(A: sp.spmatrix) -> np.ndarray:

    N = A.shape[0]

    # Compute the covariance matrix
    C = (
        (A.T * A - (sum(A).T * sum(A) / N)) /
        (N - 1)
    ).todense()

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    V = np.sqrt(
        np.diag(C).reshape(1, -1).T *
        np.diag(C).reshape(1, -1)
    )

    return np.divide(C, V + 1e-119)


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


def sparse_rmse(A, B) -> float:

    return np.sqrt(
        (A - B) \
            .power(2) \
            .mean()
    )


def idx_sample_top(
    adata: ad.AnnData,
    by: str,
    n_cells: int,
    raw: bool = False,
    random_state: int = 0
) -> ad.AnnData:

    # Return original object if set to 0
    if n_cells == 0:
        return adata

    df = adata.obs

    if raw:
        df.reset_index(inplace = True)

    return df \
        .sample(frac = 1, random_state = random_state) \
        .groupby(by) \
        .head(n_cells) \
        .index


def log2_norm(A: sp.spmatrix) -> sp.spmatrix:

    # EcoTyper uses base2 log
    # log1p --> zscore
    return sp.csr_matrix(
        sc.pp.scale(
            sc.pp.log1p(A, base = 2)
        )
    )


# def log2_deg(
#     adata,
#     groupby: str,
#     n_jobs: int = 1
# ):

#     from scipy.stats import false_discovery_control
#     from scanpy.tools._rank_genes_groups import _RankGenes

#     # Scanpy's Wilcoxon is much faster
#     # But we compute our own log foldchange
#     rgg = _RankGenes(
#         adata,
#         groups = "all",
#         groupby = groupby
#     )

#     # Log Foldchange
#     def _proc(mask_obs):

#         mat1 = g_X[mask_obs]
#         mat2 = g_X[~mask_obs]

#         return list(
#             (
#                 mat1.sum(axis = 0) / mat1.shape[0] -
#                 mat2.sum(axis = 0) / mat2.shape[0]
#             ).flat
#         )

#     # Large shared X
#     def define_global(var):
#         global g_X
#         g_X = var

#     with Pool(
#         processes = n_jobs,
#         initializer = define_global,
#         initargs = (adata.X, )
#     ) as pool:

#         foldchange = list(
#             tqdm(
#                 pool.imap(
#                     _proc,
#                     rgg.groups_masks_obs
#                 ),
#                 total = len(rgg.groups_masks_obs),
#                 desc = "Computing log foldchanges"
#             )
#         )

#     wilcoxon = list(
#         tqdm(
#             rgg.wilcoxon(tie_correct = False),
#             total = len(rgg.groups_masks_obs),
#             desc = "Statistical test"
#         )
#     )

#     for fc, w in zip(foldchange, wilcoxon):

#         df = pd.DataFrame(
#             {
#                 'foldchange': fc,
#                 'pvals': w[2]
#             },
#             index = adata.var_names.tolist()
#         )

#         df['pvals_adj'] = false_discovery_control(df['pvals'], method = 'bh')
#         df = df[(df['foldchange'] > 0) & (df['pvals_adj'] <= 0.05)]

#         yield rgg.groups_order[w[0]], df

    # res = mannwhitneyu(mat1.todense(), mat2.todense())
    # results['pvals'] = res.pvalue
    # results['pvals_adj'] = false_discovery_control(
    #     results['pvals'],
    #     method = 'bh'
    # )

    # Drop all np.nan vaues
    # return pd.DataFrame(results) \
        # .set_index('gene') \
        # .dropna() \
        # .sort_values(
        #     by = ["foldchange"],
        #     ascending = False
        # )


# # Inspired by Scanpy's parallel rankdata function
# # Modified to be broadcasted to matrcies
# # https://github.com/scverse/scanpy/blob/main/src/scanpy/tools/_rank_genes_groups.py
# def fast_rankdata(A):

#     sort_idx = np.argsort(A, axis = 0)
#     sorted_mat = np.take_along_axis(A, sort_idx, axis = 0)

#     # Boolean array of which elements are unique (after sorting)
#     consecutively_unique = np.vstack(
#         (
#             np.ones(sorted_mat.shape[1]),
#             sorted_mat[1:, :] != sorted_mat[:-1, :]
#         )
#     ).astype(int)

#     # Find the unique rank
#     # Identical elements are assigned the same rank
#     unique_rank = np.zeros(A.shape, dtype = int)

#     np.put_along_axis(
#         unique_rank,
#         sort_idx,
#         np.cumsum(consecutively_unique, axis = 0),
#         axis = 0
#     )

#     return np.hstack(
#         [
#             # Compute the average rank --> convert to column vector
#             ( (count[uniq] + count[uniq - 1] + 1) / 2 )[np.newaxis].T

#             for con, uniq in zip(
#                 consecutively_unique.T,
#                 unique_rank.T
#             )

#             # This computes the index of nonzero items in consecutively_unique
#             # Corresponds to the number of counts of each rank
#             if np.any(count := np.append(np.flatnonzero(con), A.shape[0]))
#         ]
#     )