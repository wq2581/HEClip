"""
The code for this work was developed based on the BLEEP model and we are grateful for their contribution.
Ref：
https://github.com/bowang-lab/BLEEP
https://proceedings.neurips.cc/paper_files/paper/2023/file/df656d6ed77b565e8dcdfbf568aead0a-Paper-Conference.pdf
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import harmonypy as hm


print(sc.__version__)


# filter expression matrices to only include HVGs shared across all datasets
def hvg_selection_and_pooling(exp_paths, n_top_genes=1000):
    # input n expression matrices paths, output n expression matrices with only the union of the HVGs

    # read adata and find hvgs
    hvg_bools = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        print(adata.shape)
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # save hvgs
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)

    # find union of hvgs
    hvg_union = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())

    # filter expression matrices
    filtered_exp_mtxs = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        # Preprocess the data and subset
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        filtered_exp_mtxs.append(adata[:, hvg_union].X)
    return filtered_exp_mtxs


exp_paths = ["~/GSE240429_data/data/filtered_expression_matrices/1/matrix.mtx",
             "~/GSE240429_data/data/filtered_expression_matrices/2/matrix.mtx",
             "~/GSE240429_data/data/filtered_expression_matrices/3/matrix.mtx",
             "~/GSE240429_data/data/filtered_expression_matrices/4/matrix.mtx"]

filtered_mtx = hvg_selection_and_pooling(exp_paths)

for i in range(len(filtered_mtx)):
    np.save("~/GSE240429_data/data/filtered_expression_matrices/" + str(i + 1) + "/hvg_matrix.npy", filtered_mtx[i].T)


################################
#! batch correct using harmony
#! Other batch correction methods can be used here in place of harmony. Furthermore, model can be trained using the hvg matrix and achieve comparable results if the datasets used are similar enough

d = np.load("~/GSE240429_data/data/filtered_expression_matrices/1/hvg_matrix.npy")
print(d.shape)

d2 = np.load("~/GSE240429_data/data/filtered_expression_matrices/2/hvg_matrix.npy")
print(d2.shape)

d3 = np.load("~/GSE240429_data/data/filtered_expression_matrices/3/hvg_matrix.npy")
print(d3.shape)

d4 = np.load("~/GSE240429_data/data/filtered_expression_matrices/4/hvg_matrix.npy")
print(d4.shape)

d = np.concatenate((d.T, d2.T, d3.T, d4.T), axis = 0)

data_sizes = [2378, 2349, 2277, 2265]
batch_labels = np.concatenate((np.zeros(2378), np.ones(2349), np.ones(2277)*2, np.ones(2265)*3))
batch_labels = batch_labels.astype(str)
df = pd.DataFrame(batch_labels, columns=["dataset"])

# # Run the Harmony integration algorithm
harmony = hm.run_harmony(d, meta_data=df, vars_use=["dataset"])
harmony_corrected = harmony.Z_corr.T

#split back into datasets
d1 = harmony_corrected[:data_sizes[0]]
d2 = harmony_corrected[data_sizes[0]:data_sizes[0]+data_sizes[1]]
d3 = harmony_corrected[data_sizes[0]+data_sizes[1]:data_sizes[0]+data_sizes[1]+data_sizes[2]]
d4 = harmony_corrected[data_sizes[0]+data_sizes[1]+data_sizes[2]:]

print(d1.shape, d2.shape, d3.shape, d4.shape)

#save
np.save("~/GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy", d1.T)
np.save("~/GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy", d2.T)
np.save("~/GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy", d3.T)
np.save("~/GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy", d4.T)  #saving gene x cell to be consistent with hvg_matrix.npy