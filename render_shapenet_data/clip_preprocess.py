# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import pathlib
import argparse

import torch

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE 


import clip

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess rendered images into clip image features.')
    parser.add_argument(
        '--dataset_folder', type=str, default='./tmp',
        help='path for downloaded 3d dataset folder')
    return parser.parse_args()

synset_name_list = [
    'car',
    'chair',
    'motorbike'
]

synset_list = [
    '02958343',  # Car
    '03001627',  # Chair
    '03790512'  # Motorbike
]

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.img_dirs = []
        
        for synset_name, synset in zip(synset_name_list, synset_list):
            class_dir = self.root / synset_name / 'img' / synset
            self.img_dirs += [d for d in class_dir.iterdir() if d.is_dir()]

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        img_dir = self.img_dirs[idx]
        img_paths = sorted(img_dir.glob("*.png"))
        imgs = [Image.open(img_path) for img_path in img_paths]
        
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs)
        
        return imgs, str(img_dir)

def plot(X, Y):
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

    # Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    plt.figure(figsize=(16, 16))
    
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(Y[i]), color=plt.cm.prism(Y[i]), 
                fontdict={'weight': 'bold', 'size': 9})
    
    plt.savefig('tmp.png')


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = ImageDataset(args.dataset_folder, transform=preprocess)

    for images, path in tqdm(dataset):
        with torch.no_grad():
            features = model.encode_image(images.to(device))
            features = features.cpu().numpy()
            
            output_path = pathlib.Path(path) / 'clip_feature.npy'
            np.save(output_path, features)
            

