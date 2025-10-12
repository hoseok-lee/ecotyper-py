import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os.path

from tqdm import tqdm
from collections import defaultdict
from multiprocess import Pool
from functools import partial

from .fast_nmf import FastNMF
from .utils import (
    PosNeg,
    sparse_corrcoef,
    sparse_dropout,
    sample_top
)

# Silence warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class EcoTyper:

    def __init__(
        self,
        adata: ad.AnnData,
        cell_type_idx: str,
        sample_idx: str,
        n_jobs: int = 10,

        fractions: int | str = "cell_type_specific",

        n_cells_gene_filter: int = 500,
        n_cells_distance: int = 2500,
        n_genes_top_nmf: int = 1000,
        cophenetic_cutoff: float = 0.95,

        max_clusters: int = 20,
        nmf_restarts: int = 5,

        random_state: int = 0
    ):

        self.adata = adata
        self.adata.uns['ecotyper'] = dict()

        self.cell_type_idx = cell_type_idx
        self.sample_idx = sample_idx

        self.n_jobs = n_jobs

        self.fractions = fractions
        self.n_cells_gene_filter = n_cells_gene_filter
        self.n_cells_distance = n_cells_distance
        self.n_genes_top_nmf = n_genes_top_nmf
        self.cophenetic_cutoff = cophenetic_cutoff

        self.max_clusters = max_clusters
        self.nmf_restarts = nmf_restarts

        self.random_state = random_state


    def run(self):
        import pickle

        if not os.path.isfile("saved.pkl"):

            # Step 1
            self.filter_cell_type_by_abundance()
            self.compute_cell_type_specific_genes()

            # Step 2
            self.compute_distances()

            with open("saved.pkl", "wb") as pkl:
                pickle.dump(self.adata, pkl)

        else:
            with open("saved.pkl", "rb") as pkl:
                self.adata = pickle.load(pkl)

        self.compute_nmf()


    def filter_cell_type_by_abundance(self):

        valid_cell_types = []
        cell_type_counts = self.adata.obs[self.cell_type_idx].value_counts()

        for cell_type, amount in tqdm(
            cell_type_counts.items(),
            total = len(cell_type_counts),
            desc = "Filtering cell types by abundance"
        ):

            if amount < 50:
                print(
                    f"Only {amount} single cells are available for cell type: "
                    f"{cell_type}. At least 50 are required. Skipping this cell "
                    "type!"
                )

            else:
                valid_cell_types.append(cell_type)

        self.adata = self.adata \
            [self.adata.obs[self.cell_type_idx].isin(valid_cell_types)].copy()


    def compute_cell_type_specific_genes(self):

        # Randomly sample 500 cells for balanced representation
        # This allows cells < 500 to keep without upsampling
        sub_adata = sample_top(
            self.adata,
            by = self.cell_type_idx,
            n_cells = self.n_cells_gene_filter,
            random_state = self.random_state
        )

        # EcoTyper uses base2 log
        sc.pp.log1p(sub_adata, base = 2)

        # Cell type specific
        # This dictionary will be populated
        self.adata.uns['ecotyper']['genes'] = defaultdict()

        if self.fractions == "cell_type_specific":

            sc.tl.rank_genes_groups(
                sub_adata,
                groupby = self.cell_type_idx,
                # all vs. rest
                method = 'wilcoxon',
                n_jobs = self.n_jobs
            )

            # Transfer DEG information
            self.adata.uns['rank_genes_groups'] = \
                sub_adata.uns['rank_genes_groups']

            for cell_type in tqdm(
                self.adata.obs[self.cell_type_idx].unique(),
                desc = "Computing cell type specific genes"
            ):

                df = sc.get.rank_genes_groups_df(self.adata, group = cell_type)
                # Filter for significance
                df = df[(df['logfoldchanges'] > 0) & (df['pvals_adj'] <= 0.05)]
                self.adata.uns['ecotyper']['genes'][cell_type] = df['names']

        else:
            raise ValueError("Not Implemented")


    def compute_distances(self):

        def _proc(cell_type, adata):

            genes = adata.uns['ecotyper']['genes'][cell_type]
            # Subset to cell type and genes
            cells = adata.obs[adata.obs[self.cell_type_idx] == cell_type].index
            data = adata[cells, genes].X.T

            return cell_type, pd.DataFrame.sparse.from_spmatrix(
                sparse_corrcoef(data),
                index = cells,
                columns = cells
            )


        sub_adata = sample_top(
            self.adata,
            by = self.cell_type_idx,
            n_cells = self.n_cells_distance,
            random_state = self.random_state
        )

        with Pool(processes = self.n_jobs) as pool:

            # This dictionary will be populated
            self.adata.uns['ecotyper']['distances'] = {

                cell_type: distances

                for cell_type, distances in tqdm(
                    pool.imap_unordered(
                        partial(
                            _proc,
                            adata = sub_adata
                        ),
                        self.adata.uns['ecotyper']['genes']
                    ),
                    total = len(self.adata.uns['ecotyper']['genes']),
                    desc = "Computing pairwise distances"
                )
            }


    def compute_state_marker_genes(
        self,
        adata: ad.AnnData,
        nmf_info: FastNMF.NMFInfo
    ):

        from sklearn.preprocessing import binarize


        has_gex = binarize(adata.X)
        gene_names = np.array(adata.var_names)
        state_assignment = np.argmax(nmf_info.H, axis = 0)

        adata.obs['cell_state'] = pd.Categorical(state_assignment)

        sc.pp.log1p(adata, base = 2)
        sc.tl.rank_genes_groups(
            adata,
            groupby = 'cell_state',
            # all vs. rest
            method = 'wilcoxon'
        )

        for cell_state in range(nmf_info.rank):

            df = sc.get.rank_genes_groups_df(adata, group = str(cell_state))
            df = df[(df['logfoldchanges'] > 0) & (df['pvals_adj'] <= 0.05)]

            # Filter significant genes by abundance >= 0.5
            abundance = has_gex[state_assignment == cell_state].mean(axis = 0)
            yield df[df['names'].isin(gene_names[(abundance >= 0.5).flat])]


    def filter_states_by_afi(
        self,
        nmf_info: FastNMF.NMFInfo,
        is_pos_only: bool = False
    ) -> list:

        # Index at which negative values start
        # If positives only, use the entire matrix
        pos_idx = nmf_info.W.shape[0] // (1 if is_pos_only else 2)

        afi = nmf_info.W[pos_idx:, :].sum(axis = 0) / \
                nmf_info.W[:pos_idx].sum(axis = 0)

        # np.where returns the indices where AFI < 1
        # We have not touched the W matrix, index == cell state
        return np.where(afi < 1)[0]


    # def filter_states_by_dropout(
    #     self,
    #     adata: ad.AnnData,
    #     nmf_info: FastNMF.NMFInfo
    # ) -> list:

    #     from scipy.stats import zscore

    #     valid_states = []

    #     dropout = np.zeros(nmf_info.rank)
    #     gex = np.zeros(nmf_info.rank)

    #     for cell_state, marker_genes in \
    #         enumerate(
    #             self.compute_state_marker_genes(
    #                 adata = adata,
    #                 nmf_info = nmf_info
    #             )
    #         ):

    #         # Skip if less than or equal to 10 marker genes
    #         if len(marker_genes) <= 10:
    #             continue

    #         # Subset to marker genes
    #         sub_adata = adata[:, marker_genes]
    #         valid_states.append(cell_state)

    #         # Dropout score is computed as the fraction of the highest occuring
    #         # count number per gene
    #         dropout[cell_state] = sparse_dropout(sub_adata.X).mean()

    #         # Compute average log2 expression
    #         sc.pp.log1p(sub_adata, base = 2)
    #         gex[cell_state] = sub_adata.X.mean()

    #     # Only consider valid states
    #     mean_z = zscore(
    #         zscore(dropout[valid_states]) + \
    #         zscore(gex[valid_states])
    #     )

    #     # np.where returns the states where mean z score <= 1.96 (p < 0.05)
    #     return np.array(valid_states)[np.where(mean_z <= 1.96)]

    def compute_nmf(self):

        self.max_clusters = max(2, self.max_clusters)
        self.nmf_restarts = max(1, self.nmf_restarts)

        sub_adata = sample_top(
            self.adata,
            by = self.cell_type_idx,
            n_cells = self.n_cells_distance,
            random_state = self.random_state
        )

        for cell_type, distances in tqdm(
            self.adata.uns['ecotyper']['distances'].items(),
            desc = "Computing NMF (may take a long time)"
        ):

            print(cell_type)

            nmf_info = FastNMF(
                X = PosNeg(distances.values).X,
                random_state = self.random_state
            ).estimate_rank(
                rank_range = range(2, self.max_clusters + 1),
                nmf_restarts = range(1, self.nmf_restarts + 1),
                cutoff  = self.cophenetic_cutoff
            )

            print(nmf_info)

            marker_genes = pd.concat(
                list(
                    self.compute_state_marker_genes(
                        adata = self.adata[distances.index],
                        nmf_info = nmf_info
                    )
                ),
                ignore_index = True
            ) \
                .groupby(by = 'names')['logfoldchanges'] \
                .max() \
                .sort_values(ascending = False) \
                .head(self.n_genes_top_nmf).index

            cell_type_adata = sub_adata[
                sub_adata.obs[self.cell_type_idx] == cell_type,
                marker_genes
            ]

            sc.pp.log1p(cell_type_adata, base = 2)
            sc.pp.scale(cell_type_adata)

            # Recompute NMF with marker genes
            exp_mat = PosNeg(cell_type_adata.X.T)

            nmf_info = FastNMF(
                X = exp_mat.X,
                random_state = self.random_state
            ).estimate_rank(
                rank_range = range(2, self.max_clusters + 1),
                nmf_restarts = range(1, self.nmf_restarts + 1),
                cutoff  = self.cophenetic_cutoff
            )

            print(nmf_info)

            # Filter for spurious cell states
            # 1. AFI filter
            # 2. Dropout score
            afi         = self.filter_states_by_afi(
                nmf_info = nmf_info,
                is_pos_only = exp_mat.is_pos_only()
            )

            dropouts    = self.filter_states_by_dropout(
                adata = cell_type_adata,
                nmf_info = nmf_info
            )

            valid_states = set(afi).intersection(set(dropouts))

            print(valid_states)


    def __unused__(self):
            """
            """


# Bootstrapping dataset instead of changing the NMF
# - choose random 2500
# GBMCare ecosystems
# Sarcoma
# Manually choose the rank after observing the elbow plot
# Force certain ranks instead of a range
# Dropping reccurrence
