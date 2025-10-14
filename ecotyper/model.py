import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os.path

from tqdm import tqdm
from collections import defaultdict
from multiprocess import Pool
from functools import partial
from typing import Optional
from pathlib import Path
from time import time

from .fast_nmf import FastNMF
from .utils import (
    PosNeg,
    sparse_corrcoef,
    sparse_dropout,
    idx_sample_top,
    # log2_deg,
    log2_norm
)

import warnings
warnings.filterwarnings("ignore")


class EcoTyper:

    def __init__(
        self,
        adata: ad.AnnData,
    ):

        self.adata = adata
        self.adata.uns['ecotyper'] = dict()


    def discovery(
        self,

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

        random_state: int = 0,

        figures: Optional[str] = None
    ):

        import pickle


        self.cell_type_idx = cell_type_idx
        self.sample_idx = sample_idx

        self.n_jobs = n_jobs

        self.fractions = fractions
        self.n_cells_gene_filter = n_cells_gene_filter
        self.n_cells_distance = n_cells_distance
        self.n_genes_top_nmf = n_genes_top_nmf
        self.cophenetic_cutoff = cophenetic_cutoff

        self.max_clusters = max(2, max_clusters)
        self.nmf_restarts = max(1, nmf_restarts)

        self.random_state = random_state
        self.figures = Path(figures)

        # Create figure folder
        if self.figures:
            os.makedirs(figures, exist_ok = True)

        if not os.path.isfile("saved.pkl"):

            # Step 1
            self.filter_cell_type_by_abundance()
            self.compute_cell_type_specific_genes()

            # Step 2 - 4
            self.compute_distances()
            self.compute_initial_nmf()

            with open("saved.pkl", "wb") as pkl:
                pickle.dump(self.adata, pkl)

        else:
            with open("saved.pkl", "rb") as pkl:
                self.adata = pickle.load(pkl)

        # Step 5
        self.extract_state_specific_genes()


    def __getitem__(self, key):

        f, s = key if isinstance(key, tuple) else (key, None)

        if s:
            return self.adata.uns['ecotyper'][f][s]

        return self.adata.uns['ecotyper'][f]


    def __setitem__(self, key, val):

        f, s = key if isinstance(key, tuple) else (key, None)

        if s:
            self.adata.uns['ecotyper'][f][s] = val
            return

        self.adata.uns['ecotyper'][f] = val


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

        self['cell_types'] = valid_cell_types
        self['cells'] = {

            cell_type: self.adata.obs[
                self.adata.obs[self.cell_type_idx] == cell_type
            ] \
                .index \
                .tolist()

            for cell_type in self['cell_types']
        }


    def log2_deg(self, adata):

        from scipy.stats import false_discovery_control
        from scanpy.tools._rank_genes_groups import _RankGenes

        # Scanpy's Wilcoxon is much faster
        # But we compute our own log foldchange
        rgg = _RankGenes(
            adata,
            groups = "all",
            groupby = self.cell_type_idx
        )

        # Log Foldchange
        def _proc(mask_obs):

            mat1 = g_X[mask_obs]
            mat2 = g_X[~mask_obs]

            return list(
                (
                    mat1.sum(axis = 0) / mat1.shape[0] -
                    mat2.sum(axis = 0) / mat2.shape[0]
                ).flat
            )

        # Large shared X
        def define_global(var):
            global g_X
            g_X = var

        with Pool(
            processes = self.n_jobs,
            initializer = define_global,
            initargs = (adata.X, )
        ) as pool:

            foldchange = list(
                tqdm(
                    pool.imap(
                        _proc,
                        rgg.groups_masks_obs
                    ),
                    total = len(rgg.groups_masks_obs),
                    desc = "Computing log foldchanges"
                )
            )

        wilcoxon = list(
            tqdm(
                rgg.wilcoxon(tie_correct = True),
                total = len(rgg.groups_masks_obs),
                desc = "Statistical test"
            )
        )

        for fc, w in zip(foldchange, wilcoxon):

            df = pd.DataFrame(
                {
                    'foldchange': fc,
                    'pvals': w[2]
                },
                index = adata.var_names.tolist()
            )

            df['pvals_adj'] = false_discovery_control(
                df['pvals'],
                method = 'bh'
            )

            yield rgg.groups_order[w[0]], df


    def significant_genes(self, df):

        df = df[(df['foldchange'] > 0) & (df['pvals_adj'] <= 0.05)]
        df = df.sort_values(by = 'foldchange', ascending = False)

        return df.index


    def compute_cell_type_specific_genes(self):

        # Randomly sample 500 cells for balanced representation
        # This allows cells < 500 to keep without upsampling
        sub_idx = idx_sample_top(
            self.adata,
            by = self.cell_type_idx,
            n_cells = self.n_cells_gene_filter,
            random_state = self.random_state
        )

        sub_adata = self.adata[sub_idx].copy()
        sc.pp.log1p(sub_adata, base = 2)
        sc.pp.scale(sub_adata)

        if self.fractions == "cell_type_specific":

            self['deg'] = dict(self.log2_deg(sub_adata))
            self['genes'] = dict(
                map(
                    lambda kv:
                        ( kv[0], self.significant_genes(kv[1]) ),
                    self['deg'].items()
                )
            )

        else:
            raise ValueError("Not Implemented")


    def compute_distances(self):

        def _proc(cell_type, cells, data):

            return cell_type, pd.DataFrame(
                sparse_corrcoef(data),
                index = cells,
                columns = cells
            )


        # Sample random 2500 (by default) cells per cell type
        sub_idx = idx_sample_top(
            self.adata,
            by = self.cell_type_idx,
            n_cells = self.n_cells_distance,
            random_state = self.random_state
        )

        with Pool(processes = self.n_jobs) as pool:

            # This dictionary will be populated
            self['distances'] = dict(
                tqdm(
                    pool.imap_unordered(
                        lambda args: _proc(*args),
                        [
                            (
                                cell_type,
                                cells,
                                log2_norm(
                                    self.adata[
                                        cells,
                                        self['genes', cell_type]
                                    ].X
                                ).T
                            )

                            for cell_type in self['cell_types']

                            # Cell type specific cells
                            if np.any(
                                cells := sub_idx.intersection(
                                    self['cells', cell_type]
                                )
                            )
                        ]
                    ),
                    total = len(self['cell_types']),
                    desc = "Computing pairwise distances"
                )
            )


    def compute_initial_nmf(self):

        def _proc(cell_type, data):

            return cell_type, \
                FastNMF(
                    # posneg transformation
                    X = PosNeg(data).X,
                    random_state = self.random_state
                ) \
                    .estimate_rank(
                        rank_range = range(2, self.max_clusters + 1),
                        nmf_restarts = range(1, self.nmf_restarts + 1),
                        cutoff = self.cophenetic_cutoff,
                        figures =
                            self.figures /
                            "rank_selection" /
                            "cross_corr" /
                            cell_type
                    )


        with Pool(processes = self.n_jobs) as pool:

            # This dictionary will be populated
            self.adata.uns['ecotyper']['initial_nmf'] = dict(
                tqdm(
                    pool.imap_unordered(
                        lambda args: _proc(*args),
                        [
                            (
                                cell_type,
                                self['distances', cell_type].values
                            )
                            for cell_type in self['cell_types']
                        ]
                    ),
                    total = len(self['cell_types']),
                    desc = "Computing initial NMF (may take a long time)"
                )
            )


    def extract_state_specific_genes(self):

        print(self.adata.uns['ecotyper']['initial_nmf'])


            # print(nmf_info)

            # marker_genes = pd.concat(
            #     list(
            #         self.compute_state_marker_genes(
            #             adata = self.adata[distances.index],
            #             nmf_info = nmf_info
            #         )
            #     ),
            #     ignore_index = True
            # ) \
            #     .groupby(by = 'names')['logfoldchanges'] \
            #     .max() \
            #     .sort_values(ascending = False) \
            #     .head(self.n_genes_top_nmf).index

            # cell_type_adata = sub_adata[
            #     sub_adata.obs[self.cell_type_idx] == cell_type,
            #     marker_genes
            # ]

            # sc.pp.log1p(cell_type_adata, base = 2)
            # sc.pp.scale(cell_type_adata)

            # # Recompute NMF with marker genes
            # exp_mat = PosNeg(cell_type_adata.X.T)

            # nmf_info = FastNMF(
            #     X = exp_mat.X,
            #     random_state = self.random_state
            # ).estimate_rank(
            #     rank_range = range(2, self.max_clusters + 1),
            #     nmf_restarts = range(1, self.nmf_restarts + 1),
            #     cutoff  = self.cophenetic_cutoff,
            #     figures =
            #         self.figures /
            #         "rank_selection" /
            #         "gene_exp" /
            #         cell_type
            # )

            # print(nmf_info)

            # # Filter for spurious cell states
            # # 1. AFI filter
            # # 2. Dropout score
            # afi = self.filter_states_by_afi(
            #     nmf_info = nmf_info,
            #     is_pos_only = exp_mat.is_pos_only()
            # )

            # dropouts = self.filter_states_by_dropout(
            #     adata = cell_type_adata,
            #     nmf_info = nmf_info
            # )

            # valid_states = set(afi).intersection(set(dropouts))

            # print(valid_states)


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
