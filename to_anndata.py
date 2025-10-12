import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

data_path = ""
annotation_path = ""


data = pd.read_csv(
    data_path,
    compression = 'gzip',
    sep = "\t",
    index_col = 0
).dropna()

annotation = pd.read_csv(
    annotation_path,
    sep = "\t",
    index_col = 0
)

adata = ad.AnnData(
    X = csr_matrix(data.values).T,
    obs = annotation
)
adata.var_names = data.index