import anndata as ad

data_path = ""

cell_type_index = "CellType"
sample_index = "Sample"


adata = ad.read_h5ad(data_path)


############ Step 1
valid_cell_types = []
for cell_type, amount in adata.obs[cell_type_index].value_counts().items():

    if cell_type < 50:
        print(
            "Only {amount} single cells are available for cell type: "
            f"{cell_type}. At least 50 are required. Skipping this cell type!"
        )

    else:
        valid_cell_types.append(cell_type)


# Scale data
