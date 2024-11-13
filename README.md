
# HEClip


This code is prepared for **"HEClip: An Advanced CLIP-like  model for Gene Expression Prediction from Histology Images"**.

## Overview

### Abstract
HEClip is an innovative CLIP-based model designed to predict gene expression directly from histological images, addressing the challenges of accuracy and complexity faced by traditional methods. Histological images play a crucial role in medical diagnosis and research, but matching them with corresponding gene expression profiles is often time-consuming and costly. While various machine learning approaches have been proposed to tackle this issue, they often struggle with accuracy and involve complex workflows, limiting their effectiveness in predicting gene expression from image data. HEClip leverages contrastive learning and single-modality-centered loss functions to optimize the image encoder, enhancing the predictive power of the image modality while reducing reliance on the gene modality. Unlike traditional methods, HEClip employs image-based data augmentation strategies and achieves state-of-the-art performance across multiple benchmarks. Evaluations on the GSE240429 liver dataset demonstrate HEClip's strong performance in predicting highly expressed genes, achieving high correlation, hit rates, and stable cross-cell predictions.


![The flowchart.](./heclip_00.png)

## Installation
Download HEClip:
```git clone https://github.com/QSong-github/BrainCLIP```

Install Environment:
```pip install -r requirements.txt``` or ```conda env create -f environment.yaml```


## Running

### Train the HEClip.

   
   (1) download the iamge data.
   ```bash
   $ cd /path/to/AD_43SNP.zip
   $ unzip AD_43SNP.zip

   $ cd /path/to/reukbb.zip
   $ unzip reukbb.zip
   ```
   
   (2) Build the dataset.
   ```bash
   $ cd /path/to/Uni
   $ python dataset_making.py
   ```
   (3) Train the model.
   ```bash
   $ cd /path/to/Uni
   $ python main.py
   ```
   
### Inference   

   (4) Inference.
   ```bash
   $ cd /path/to/Uni
   $ python infer.py
   ```

### Train the BrainClip with SNP or Label.

We also provide training code using only SNPs or labels. If you want to try it, please visit the [SNP](https://github.com/QSong-github/BrainCLIP/tree/main/SNP) folder or [Label](https://github.com/QSong-github/BrainCLIP/tree/main/Label) folder.


## Quick start

If you want to use our model, you can download the pre-trained BrainClip model from [here](https://github.com/QSong-github/BrainCLIP/tree/main/save) and quickly try it by the [tutorial](https://github.com/QSong-github/BrainCLIP/blob/main/tutorial.ipynb).

   
