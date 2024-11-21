import scanpy as sc
import pandas as pd


df = pd.read_csv("file")


cell_names = df.iloc[:, 0]
labels = df.iloc[:, 1]


data_matrix = df.iloc[:, 2:].values



adata = sc.AnnData(data_matrix, dtype=data_matrix.dtype)

adata.obs['cell_names'] = cell_names.values
adata.obs['labels'] = labels.values
adata.var_names = df.columns[2:]


sc.pp.log1p(adata)


sc.pp.filter_genes(adata, min_counts=1)



sc.pp.highly_variable_genes(adata, n_top_genes=1000)


adata_highly_variable = adata[:, adata.var['highly_variable']]

print(adata_highly_variable)
print(adata_highly_variable.var_names.tolist())

