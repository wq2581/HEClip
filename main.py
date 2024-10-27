"""
The code for this work was developed based on the BLEEP model and we are grateful for their contribution.
Ref：
https://github.com/bowang-lab/BLEEP
https://proceedings.neurips.cc/paper_files/paper/2023/file/df656d6ed77b565e8dcdfbf568aead0a-Paper-Conference.pdf
"""

import os
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.utils.data.distributed

from dt_load import CLIPDataset
from models import CLIPModel
from utils import AvgMeter
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='DDP for CLIP')

parser.add_argument('--save_path', type=str, default='./save', help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_epochs', type=int, default=15, help='')
parser.add_argument('--dataset', type=str, default='bleep_data', help='[bleep_data,all_data,aug_data]')
parser.add_argument('--max_seq_len', default=512, type=int, help='')
parser.add_argument('--order', default=True, type=bool, help='')
parser.add_argument('--exp_adaption', default=False, type=bool, help='')
parser.add_argument('--use_hvg', default=True, type=bool, help='')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')


def build_loaders(args):
    print("Building loaders")
    if args.dataset=='bleep_data':
        dataset = CLIPDataset(image_path = "../image/GEX_C73_A1_Merged.tiff",
                   spatial_pos_path = "../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_1.csv",
                   reduced_mtx_path = "../GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy",
                   barcode_path = "../GSE240429_data/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset2 = CLIPDataset(image_path = "../image/GEX_C73_B1_Merged.tiff",
                    spatial_pos_path = "../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                    reduced_mtx_path = "../GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy",
                    barcode_path = "../GSE240429_data/data/filtered_expression_matrices/2/barcodes.tsv")
        dataset4 = CLIPDataset(image_path = "../image/GEX_C73_D1_Merged.tiff",
                    spatial_pos_path = "../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_4.csv",
                    reduced_mtx_path = "../GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy",
                    barcode_path = "../GSE240429_data/data/filtered_expression_matrices/4/barcodes.tsv")
        # repeat for data augmentation
        dataset5 = CLIPDataset(image_path = "../image/GEX_C73_A1_Merged.tiff",
                   spatial_pos_path = "../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_1.csv",
                   reduced_mtx_path = "../GSE240429_data/data/filtered_expression_matrices/1/harmony_matrix.npy",
                   barcode_path = "../GSE240429_data/data/filtered_expression_matrices/1/barcodes.tsv")
        dataset6 = CLIPDataset(image_path = "../image/GEX_C73_B1_Merged.tiff",
                    spatial_pos_path = "../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                    reduced_mtx_path = "../GSE240429_data/data/filtered_expression_matrices/2/harmony_matrix.npy",
                    barcode_path = "../GSE240429_data/data/filtered_expression_matrices/2/barcodes.tsv")
        dataset7 = CLIPDataset(image_path = "../image/GEX_C73_D1_Merged.tiff",
                    spatial_pos_path = "../GSE240429_data/data/tissue_pos_matrices/tissue_positions_list_4.csv",
                    reduced_mtx_path = "../GSE240429_data/data/filtered_expression_matrices/4/harmony_matrix.npy",
                    barcode_path = "../GSE240429_data/data/filtered_expression_matrices/4/barcodes.tsv")

    dataset = torch.utils.data.ConcatDataset([dataset, dataset2, dataset4, dataset5, dataset6, dataset7])
    # dataset = torch.utils.data.ConcatDataset([dataset, dataset2, dataset4])

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))
    print('train_dataset:', len(train_dataset), 'test_dataset:', len(test_dataset))
    print("train/test split completed")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    print("Finished building loaders")
    return train_loader, test_loader



def train_epoch(model, train_loader, optimizer):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    return loss_meter


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    print("Starting...")
    args = parser.parse_args()

    model = CLIPModel().to(args.device)

    train_loader, test_loader = build_loaders(args)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.001
    )

    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        # Train
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer)

        print('train_loss:', train_loss)
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)

        if test_loss.avg < best_loss:
            if not os.path.exists(str(args.save_path)):
                os.mkdir(str(args.save_path))
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), str(args.save_path) + "/best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))


if __name__ == "__main__":
    main()