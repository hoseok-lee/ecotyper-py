from ecotyper.model import EcoTyper
import anndata as ad

data_path = "example_data/scRNA_CRC_adata.h5ad"
adata = ad.read_h5ad(data_path)

EcoTyper(
    adata,
    cell_type_idx = "CellType",
    sample_idx = "Sample",

    # max_clusters = 3,

    random_state = 1234
).run()