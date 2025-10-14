from pathlib import Path
import pandas as pd
import pickle
import numpy as np

from collections import defaultdict
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform


path_to_ecotyper = Path("/mnt/data0/hoseoklee/workspace/ecotyper/EcoTyper/discovery_scRNA_CRC")
path_to_object = Path("/mnt/data0/hoseoklee/workspace/ecotyper-py/saved.pkl")

with open(path_to_object, "rb") as pkl:
    adata = pickle.load(pkl)


def cell_type_concordance():

    def compute_dcg(x, y):

        def find(
            arr, val
        ):

            idx = np.where(arr == val)[0]

            if len(idx) == 0:
                return 0

            return idx[0]

        x = np.array(list(x))
        y = np.array(list(y))

        return sum(
            # Logarithmic
            find(y, gene) / np.log2(rank + 2)
            for rank, gene in enumerate(x)
        )


    data = defaultdict(list)

    # DEG per cell type
    cell_types = adata.uns['ecotyper']['cell_types']

    for cell_type in cell_types:

        R_genes = pd.read_csv(
            path_to_ecotyper /
            "Cell_type_specific_genes" /
            "Analysis" /
            "Cell_type_specific_genes" /
            f"{cell_type}_cell_type_specific_genes.txt",
            sep = "\t",
            index_col = 0
        )

        py_genes = adata.uns['ecotyper']['genes'][cell_type]

        # Jaccard index
        R_set = set(R_genes.index)
        py_set = set(py_genes)
        jaccard_index = len(R_set & py_set) / len(R_set | py_set)

        # Normalized DCG
        ideal_dcg = compute_dcg(R_set, R_set)
        ndcg = compute_dcg(py_set, R_set) / ideal_dcg

        data[cell_type] = [jaccard_index, ndcg]

    print(
        pd.DataFrame(
            data,
            columns = cell_types,
            index = [ "Jaccard Index", "Normalized DCG" ]
        ).T
    )


def corrcoef_concordance():

    data = defaultdict(float)

    # Correlation per cell type
    cell_types = adata.uns['ecotyper']['cell_types']

    for cell_type in cell_types:

        R_corr = pd.read_csv(
            path_to_ecotyper /
            "Cell_type_specific_genes" /
            "Cell_States" /
            "discovery_cross_cor" /
            cell_type /
            "expression_top_genes_scaled.txt",
            sep = "\t",
            index_col = 0
        )

        py_corr = adata.uns['ecotyper']['distances'][cell_type]

        cells = R_corr.index.map(lambda x: x .replace('.', '-'))

        data[cell_type] = [
            pearsonr(
                R_corr.values,
                py_corr.loc[cells, cells].values,
                axis = None
            ).statistic
        ]

    print(
        pd.DataFrame(
            data,
            columns = cell_types,
            index = [ "Pearson Correlation" ]
        ).T
    )


cell_type_concordance()
corrcoef_concordance()