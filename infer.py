"""
The code for this work was developed based on the BLEEP model and we are grateful for their contribution.
Ref：
https://github.com/bowang-lab/BLEEP
https://proceedings.neurips.cc/paper_files/paper/2023/file/df656d6ed77b565e8dcdfbf568aead0a-Paper-Conference.pdf
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from dt_load import CLIPDataset
from models import CLIPModel
from torch.utils.data import DataLoader
import os
import numpy as np



def build_loaders_inference():
    print("Building loaders")
    # (3467, 2378) (3467, 2349) (3467, 2277) (3467, 2265)
    dataset1 = CLIPDataset(image_path="../image/GEX_C73_A1_Merged.tiff",
                           spatial_pos_path="../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_1.csv",
                           reduced_mtx_path="../GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy",
                           barcode_path="../GSE240429_data/data/filtered_expression_matrices/1/barcodes.tsv")
    dataset2 = CLIPDataset(image_path="../image/GEX_C73_B1_Merged.tiff",
                           spatial_pos_path="../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                           reduced_mtx_path="../GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy",
                           barcode_path="../GSE240429_data/data/filtered_expression_matrices/2/barcodes.tsv")
    dataset3 = CLIPDataset(image_path="../image/GEX_C73_C1_Merged.tiff",
                           spatial_pos_path="../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_3.csv",
                           reduced_mtx_path="../GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy",
                           barcode_path="../GSE240429_data/data/filtered_expression_matrices/3/barcodes.tsv")
    dataset4 = CLIPDataset(image_path="../image/GEX_C73_D1_Merged.tiff",
                           spatial_pos_path="../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_4.csv",
                           reduced_mtx_path="../GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy",
                           barcode_path="../GSE240429_data/data/filtered_expression_matrices/4/barcodes.tsv")

    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4])
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    print("Finished building loaders")
    return test_loader

def get_image_embeddings(model_path, model):
    test_loader = build_loaders_inference()
    # model = CLIPModel().cuda()

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    # print(model)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

    return torch.cat(test_image_embeddings)


def get_spot_embeddings(model_path, model):
    test_loader = build_loaders_inference()
    # model = CLIPModel().cuda()

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            spot_embeddings.append(model.spot_projection(batch["reduced_expression"].cuda()))
    return torch.cat(spot_embeddings)


# 2265x256, 2277x256
def find_matches(spot_embeddings, query_embeddings, top_k):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T  # 2277x6992
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    cnt = 0
    for i in range(indices.cpu().numpy().shape[0]):
        if i in indices[i]:
            cnt=cnt+1

    return indices.cpu().numpy()   #for every sample retrieve top50 from reference set

#outputs:
#data sizes: (3467, 2378) (3467, 2349) (3467, 2277) (3467, 2265)

datasize = [2378, 2349, 2277, 2265]
model_path = './save/best.pt'
print(model_path)

save_path = "./result/embeddings/"
model = CLIPModel().cuda()

img_embeddings_all = get_image_embeddings(model_path, model)
img_embeddings_all = img_embeddings_all.cpu().numpy()

print(img_embeddings_all.shape)

if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(4):
    index_start = sum(datasize[:i])
    index_end = sum(datasize[:i+1])
    image_embeddings = img_embeddings_all[index_start:index_end]
    print(image_embeddings.shape)
    np.save(save_path + "img_embeddings_" + str(i+1) + ".npy", image_embeddings.T)


#infer spot embeddings and expression
spot_expression1 = np.load("../GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy")
spot_expression2 = np.load("../GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy")
spot_expression3 = np.load("../GSE240429_data/data/filtered_expression_matrices/3/harmony_matrix.npy")
spot_expression4 = np.load("../GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy")

save_path = "./result/embeddings/"
image_embeddings3 = np.load(save_path + "img_embeddings_3.npy")
image_embeddings1 = np.load(save_path + "img_embeddings_1.npy")
image_embeddings2 = np.load(save_path + "img_embeddings_2.npy")
image_embeddings4 = np.load(save_path + "img_embeddings_4.npy")

#query
image_query = image_embeddings3
expression_gt = spot_expression3
#reference
reference = np.concatenate([image_embeddings1, image_embeddings2, image_embeddings4], axis = 1)    # for retrival
expression_key = np.concatenate([spot_expression1, spot_expression2, spot_expression4], axis = 1)  # for exp generation


save_path = " "
if image_query.shape[1] != 256:
    image_query = image_query.T
    print("image query shape: ", image_query.shape)
if expression_gt.shape[0] != image_query.shape[0]:
    expression_gt = expression_gt.T
    print("expression_gt shape: ", expression_gt.shape)
if reference.shape[1] != 256:
    reference = reference.T
    print("reference shape: ", reference.shape)
if expression_key.shape[0] != reference.shape[0]:
    expression_key = expression_key.T
    print("expression_key shape: ", expression_key.shape)


print("finding matches, using average of top 50 expressions")
indices = find_matches(reference, image_query, top_k=200)
matched_spot_embeddings_pred = np.zeros((indices.shape[0], reference.shape[1]))  # top embedding
matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))  # top exp
for i in range(indices.shape[0]):
    matched_spot_embeddings_pred[i, :] = np.average(reference[indices[i, :], :], axis=0)    # average
    matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0)

print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)


def Hit_at_k(pred, true, k):
    # - pred: ndarray，(2277, 3467)
    # - true: ndarray，(2277, 3467)
    # - k: top k index

    num_samples = pred.shape[0]
    
    # find index
    pred_top_k_indices = np.argsort(pred, axis=1)[:, -k:]
    true_top_k_indices = np.argsort(true, axis=1)[:, -k:]

    # calculate Hit@K
    correct_predictions = 0
    
    for i in range(num_samples):
        # find intersections
        pred_set = set(pred_top_k_indices[i])
        true_set = set(true_top_k_indices[i])
        
        if pred_set & true_set:  # If there is an intersection, it is considered a correct prediction.
            correct_predictions += 1

    return correct_predictions / num_samples




true = expression_gt
pred = matched_spot_expression_pred


print(pred.shape)   #(2277, 3467)
print(true.shape)   #(2277, 3467)

hit_at_k = Hit_at_k(pred, true, 5)
print(f"Hit@{5} : {hit_at_k:.4f}")
hit_at_k = Hit_at_k(pred, true, 4)
print(f"Hit@{4} : {hit_at_k:.4f}")
hit_at_k = Hit_at_k(pred, true, 3)
print(f"Hit@{3} : {hit_at_k:.4f}")
hit_at_k = Hit_at_k(pred, true, 2)
print(f"Hit@{2} : {hit_at_k:.4f}")
hit_at_k = Hit_at_k(pred, true, 1)
print(f"Hit@{1} : {hit_at_k:.4f}")



# genewise correlation
corr = np.zeros(pred.shape[0])
for i in range(pred.shape[0]):
    corr[i] = np.corrcoef(pred[i, :], true[i, :], )[0, 1]
corr = corr[~np.isnan(corr)]

# row for cell
print("MeanCor across cells: ", np.mean(corr))

corr = np.zeros(pred.shape[1])
for i in range(pred.shape[1]):
    corr[i] = np.corrcoef(pred[:, i], true[:, i], )[0, 1]
corr = corr[~np.isnan(corr)]

# column for gene
print("MeanCor: ", np.mean(corr))
print("MaxCor: ", np.max(corr))

ind = np.argsort(np.sum(true, axis=0))[-10:]
print("HEG-10: ", np.mean(corr[ind]))
ind = np.argsort(np.var(true, axis=0))[-10:]
print("HVG-10: ", np.mean(corr[ind]))

ind = np.argsort(np.sum(true, axis=0))[-50:]
print("HEG-50: ", np.mean(corr[ind]))
ind = np.argsort(np.var(true, axis=0))[-50:]
print("HVG-50: ", np.mean(corr[ind]))

ind = np.argsort(np.sum(true, axis=0))[-100:]
print("HEG-100: ", np.mean(corr[ind]))
ind = np.argsort(np.var(true, axis=0))[-100:]
print("HVG-100: ", np.mean(corr[ind]))

# marker genes
marker_gene_list = ["HAL", "CYP3A4", "VWF", "SOX9", "KRT7", "ANXA4", "ACTA2", "DCN"]
gene_names = pd.read_csv("../GSE240429_data/data/filtered_expression_matrices/3/features.tsv", header=None,
                         sep="\t").iloc[:, 1].values
hvg_b = np.load("../GSE240429_data/data/filtered_expression_matrices/hvg_union.npy")
marker_gene_ind = np.zeros(len(marker_gene_list))
for i in range(len(marker_gene_list)):
    marker_gene_ind[i] = np.where(gene_names[hvg_b] == marker_gene_list[i])[0]
print("mean correlation marker genes: ", np.mean(corr[marker_gene_ind.astype(int)]))


