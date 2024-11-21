import pandas as pd


df = pd.read_csv("file")


cell_names = df.iloc[:, 0]
labels = df.iloc[:, 1]


data_matrix = df.iloc[:, 2:]


gene_means = data_matrix.mean(axis=0)


n_top_genes = 1000


top_genes = gene_means.nlargest(n_top_genes).index.tolist()


heg_data = data_matrix[top_genes]


print("High Expression Genes (HEG):")
print(top_genes)
print("HEG Data:")
print(heg_data)
