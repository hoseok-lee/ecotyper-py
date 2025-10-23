import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import os.path
import scipy.sparse as sp

from tqdm import tqdm
from collections import defaultdict
from multiprocess import Pool
from functools import partial
from typing import Optional, Literal
from pathlib import Path
from time import time

from .fast_nmf import FastNMF
from .utils import (
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

        nmf_solver: Literal["mu", "cd"] = "mu",
        nmf_beta_loss: Literal[
            "kullback-leibler",
            "frobenius",
            "itakura-saito"
        ] = "kullback-leibler",

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
        self.nmf_solver = nmf_solver
        self.nmf_beta_loss = nmf_beta_loss

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

            # Step 5 - 7
            self.extract_state_specific_genes()
            self.recompute_nmf()
            self.filter_states()

            with open("saved.pkl", "wb") as pkl:
                pickle.dump(self.adata, pkl)

        else:
            with open("saved.pkl", "rb") as pkl:
                self.adata = pickle.load(pkl)

        # Step 7
        self.map_cell_states()
        # Step 8


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


    def log2_deg(
        self,
        groupby: str | list,
        obs_subset: pd.Index = slice(None),
        var_subset: pd.Index = slice(None)
    ):

        from scipy.stats import false_discovery_control
        from scanpy.tools._rank_genes_groups import _RankGenes

        # Subset and log2 + z-scale
        subset = self.adata[obs_subset, var_subset]
        sp_mat = log2_norm(subset.X)

        # Passed raw data as annotations
        if not isinstance(groupby, str):
            subset.obs['groupby'] = pd.Series(
                groupby,
                index = subset.obs_names,
                dtype = "category"
            )
            groupby = 'groupby'

        # Scanpy's Wilcoxon is much faster
        # But we compute our own log foldchange
        rgg = _RankGenes(
            subset,
            groups = "all",
            groupby = groupby
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


        def define_global(var):
            global g_X
            g_X = var

        with Pool(
            processes = self.n_jobs,
            initializer = define_global,
            initargs = (sp_mat, )
        ) as pool:

            foldchange = list(
                tqdm(
                    pool.map(
                        _proc,
                        rgg.groups_masks_obs
                    ),
                    total = len(rgg.groups_masks_obs),
                    desc = "Computing log foldchanges",
                    leave = False
                )
            )

        wilcoxon = list(
            tqdm(
                rgg.wilcoxon(tie_correct = True),
                total = len(rgg.groups_masks_obs),
                desc = "Statistical test",
                leave = False
            )
        )

        for fc, w in zip(foldchange, wilcoxon):

            df = pd.DataFrame(
                {
                    'foldchange': fc,
                    'pvals': w[2]
                },
                index = subset.var_names.rename("genes")
            )

            df['pvals_adj'] = false_discovery_control(
                df['pvals'],
                method = 'bh'
            )

            yield rgg.groups_order[w[0]], df


    def significant_genes(self, df: pd.DataFrame):

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

        if self.fractions == "cell_type_specific":

            self['deg'] = dict(
                tqdm(
                    self.log2_deg(
                        groupby = self.cell_type_idx,
                        obs_subset = sub_idx
                    ),
                    total = len(self['cell_types']),
                    desc = "Computing cell type specific genes",
                )
            )

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

            self['distances'] = dict(
                tqdm(
                    pool.imap_unordered(
                        lambda args: _proc(*args),
                        [
                            (
                                cell_type,
                                cells,
                                log2_norm(self.adata[cells, genes].X).T
                            )

                            for cell_type, genes in self['genes'].items()

                            # Cell type specific cells
                            if np.any(
                                cells := sub_idx.intersection(
                                    self['cells', cell_type]
                                )
                            )
                        ]
                    ),
                    total = len(self['genes']),
                    desc = "Computing pairwise distances"
                )
            )


    def compute_initial_nmf(self):

        def _proc(cell_type, data):

            return cell_type, \
                FastNMF(
                    # posneg transformation
                    X = data,
                    solver = self.nmf_solver,
                    beta_loss = self.nmf_beta_loss,
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

            self['initial_nmf'] = dict(
                tqdm(
                    pool.imap_unordered(
                        lambda args: _proc(*args),
                        [
                            (
                                cell_type,
                                distances.values
                            )
                            for cell_type, distances \
                                in self['distances'].items()
                        ]
                    ),
                    total = len(self['distances']),
                    desc = "Computing initial NMF on cross-correlation (may take a long time)"
                )
            )


    def compute_state_marker_genes(
        self,
        nmf_info: FastNMF.NMFInfo,
        obs_subset: pd.Index = slice(None),
        var_subset: pd.Index = slice(None)
    ):

        from sklearn.preprocessing import binarize


        has_gex = binarize(self.adata[obs_subset].X)
        gene_names = np.array(self.adata.var_names)
        state_assignment = FastNMF.assign_states(nmf_info).astype(str)

        results = defaultdict(pd.DataFrame)

        for cell_state, df in self.log2_deg(
            groupby = state_assignment,
            obs_subset = obs_subset,
            var_subset = var_subset
        ):

            df = df[(df['foldchange'] > 0) & (df['pvals_adj'] <= 0.05)]

            # Filter significant genes by abundance >= 0.5
            abundance = has_gex[state_assignment == cell_state].mean(axis = 0)

            results[cell_state] = \
                df[df.index.isin(gene_names[(abundance >= 0.5).flat])]

        return pd.concat(
            results.values(),
            axis = 0,
            keys = results,
            names = ['cell_state', 'genes']
        )


    def extract_state_specific_genes(self):

        self['state_genes'] = {
            cell_type: self.compute_state_marker_genes(
                nmf_info = nmf_info,
                obs_subset = self['distances',   cell_type].index,
                var_subset = self['genes',       cell_type]
            )

            for cell_type, nmf_info in tqdm(
                self['initial_nmf'].items(),
                desc = "Computing state specific genes"
            )
        }


    def recompute_nmf(self):

        def _proc(cell_type, data, rank):

            return cell_type, \
                FastNMF(
                    X = data,
                    solver = self.nmf_solver,
                    beta_loss = self.nmf_beta_loss,
                    random_state = self.random_state
                ) \
                    .estimate_rank(
                        rank_range = [ rank ],
                        nmf_restarts = range(1, self.nmf_restarts + 1)
                    )


        with Pool(processes = self.n_jobs) as pool:

            self['recomp_nmf'] = dict(
                tqdm(
                    pool.imap_unordered(
                        lambda args: _proc(*args),
                        [
                            (
                                cell_type,
                                self.adata[
                                    self['distances',   cell_type].index,
                                    # Get top genes out of all cell states
                                    # groupby --> sort --> head
                                    self['state_genes', cell_type] \
                                        .groupby(by = 'genes')['foldchange'] \
                                        .max() \
                                        .sort_values(ascending = False) \
                                        .head(self.n_genes_top_nmf).index
                                ].X.T,
                                nmf_info.rank
                            )
                            for cell_type, nmf_info \
                                in self['initial_nmf'].items()
                        ]
                    ),
                    total = len(self['initial_nmf']),
                    desc = "Re-computing NMF on gene expression"
                )
            )


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


    def filter_states_by_dropout(
        self,
        cell_type: str,
        nmf_info: FastNMF.NMFInfo
    ) -> list:

        from scipy.stats import zscore

        valid_states = []

        dropout = np.zeros(nmf_info.rank)
        gex = np.zeros(nmf_info.rank)

        for cell_state, marker_genes in \
            self['state_genes', cell_type].groupby(by = 'cell_state'):

            # Skip if less than or equal to 10 marker genes
            if len(marker_genes) <= 10:
                continue

            cell_state = int(cell_state)
            valid_states.append(cell_state)

            # Subset to marker genes
            sp_mat = self.adata[
                self['distances', cell_type].index,
                marker_genes \
                    .index \
                    .get_level_values('genes') \
                    .unique()
            ].X

            # Dropout score is computed as the fraction of the highest occuring
            # count number per gene
            dropout[cell_state] = sparse_dropout(sp_mat).mean()

            # Compute average log2 expression
            gex[cell_state] = sc.pp.log1p(sp_mat, base = 2).mean()

        # Only consider valid states
        mean_z = zscore(
            zscore(dropout[valid_states]) + \
            zscore(gex[valid_states])
        )

        # np.where returns the states where mean z score <= 1.96 (p < 0.05)
        return np.array(valid_states)[np.where(mean_z <= 1.96)]


    def filter_states(self):

        def _proc(cell_type, nmf_info):

            # Filter for spurious cell states
            # 1. AFI filter
            # 2. Dropout score
            afi = self.filter_states_by_afi(
                nmf_info = nmf_info,
                is_pos_only = nmf_info.pos_only
            )

            dropouts = self.filter_states_by_dropout(
                cell_type = cell_type,
                nmf_info = nmf_info
            )

            valid_states = set(afi).intersection(set(dropouts))

            return cell_type, list(valid_states)


        with Pool(processes = self.n_jobs) as pool:

            self['filt_states'] = dict(
                tqdm(
                    pool.imap_unordered(
                        lambda args: _proc(*args),
                        [
                            (
                                cell_type,
                                nmf_info
                            )
                            for cell_type, nmf_info \
                                in self['recomp_nmf'].items()
                        ]
                    ),
                    total = len(self['recomp_nmf']),
                    desc = "Filtering re-computed cell states"
                )
            )


    def map_cell_states(self):

        for cell_type, valid_states in self['filt_states'].items():

            valid_states = list(valid_states)

            if len(valid_states) == 0:
                continue

            cells_idx = np.isin(
                FastNMF.assign_states(self['recomp_nmf', cell_type]),
                list(valid_states)
            )

            valid_cells = self['distances', cell_type].index[cells_idx]

            valid_genes = self['state_genes', cell_type] \
                .loc[map(str, valid_states), :] \
                .index \
                .get_level_values('genes') \
                .unique()

            nmf_info = self['recomp_nmf', cell_type]

            subset = self.adata[valid_cells, valid_genes].copy()
            subset.X = log2_norm(subset.X)
            assigned_states = \
                FastNMF.assign_states(
                    nmf_info.H[
                        np.ix_(
                            valid_states,
                            cells_idx
                        )
                    ]
                )

            subset.obs['cell_state'] = pd.Series(
                assigned_states,
                index = subset.obs_names,
                dtype = "category"
            )

            sc.tl.dendrogram(
                subset,
                groupby             = 'cell_state',
                linkage_method      = "average",
                optimal_ordering    = True
            )

            sc.pl.heatmap(
                subset,
                var_names           = valid_genes,
                groupby             = 'cell_state',
                dendrogram          = True,
                swap_axes           = True,
                cmap                = "bwr",
                save                = cell_type
            )


    @staticmethod
    def read_text(
        data: str,
        annotation: str
    ):

        df = pd.read_csv(
            data,
            compression = 'gzip',
            sep = "\t",
            index_col = 0
        ).dropna()

        obs = pd.read_csv(
            annotation,
            sep = "\t",
            index_col = 0
        )

        adata = ad.AnnData(
            X = sp.csr_matrix(df.values).T,
            obs = obs
        )
        adata.var_names = df.index

        return EcoTyper(adata)
