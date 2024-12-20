"""
The code for this work was developed based on the BLEEP model and we are grateful for their contribution.
Ref：
https://github.com/bowang-lab/BLEEP
https://proceedings.neurips.cc/paper_files/paper/2023/file/df656d6ed77b565e8dcdfbf568aead0a-Paper-Conference.pdf
"""
import cv2
import pandas as pd
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image


class HECLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path):
        # image_path is the path of an entire slice of visium h&e stained image (~2.5GB)

        # spatial_pos_csv
        # barcode name
        # detected tissue boolean
        # x spot index
        # y spot index
        # x spot position (px)
        # y spot position (px)

        # expression_mtx
        # feature x spot (alphabetical barcode order)

        # barcode_tsv
        # spot barcodes - alphabetical order

        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(reduced_mtx_path).T  # cell x features

        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0]
        image = self.whole_image[(v1 - 128):(v1 + 128), (v2 - 128):(v2 + 128)]
        image = self.transform(image)

        item['image'] = torch.tensor(image).permute(2, 0, 1).float()  # color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float()  # cell x features (3467)


        return item

    def __len__(self):
        return len(self.barcode_tsv)
