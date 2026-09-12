
import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
sys.path.append(os.path.dirname(os.path.dirname(DOSSIER_PARENT)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DOSSIER_PARENT))))

from torch.utils.data import DataLoader
from utils.TSNE.dataset import words
from basic.models import FCN_Encoder
import math
import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
import random
from PIL import Image
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image
import warnings
#from torch._six import inf
from typing import Union, Iterable
from tqdm import tqdm
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

params = {
    "model":{
        "name" : "IAM",
        "preprocessor":[
                {
                    "type": "to_RGB",
                },
            ],
        "config": {
            "full_compatibility": True,  # for backward compatibility
            "load_in_memory": True,  # Load all images in CPU memory
            "worker_per_gpu": 8,  # Num of parallel processes per gpu for data loading
            "width_divisor": 8,  # Image width will be divided by 8
            "height_divisor": 32,  # Image height will be divided by 32
            "padding_value": 0,  # Image padding value
            "padding_token": 1000,  # Label padding value (None: default value is chosen)
            "padding_mode": "br",  # Padding at bottom and right
            "charset_mode": "CTC",  # add blank token
            "constraints": ["CTC_line", ],  # Padding for CTC requirements if necessary
            "normalize": True,  # Normalize with mean and variance of training dataset
            "training_samples": None,  # Number of training samples to use (None to use all samples)"
            "padding": {
                "min_height": "max",  # Pad to reach max height of training samples
                "min_width": "max",  # Pad to reach max width of training samples
                "min_pad": None,
                "max_pad": None,
                "mode": "br",  # Padding at bottom and right
                "train_only": False,  # Add padding at training time and evaluation time
            },
            "preprocessings": [
                {
                    "type": "to_RGB",
                    # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                },
            ],
        },
        #"augmentations_page": default_doc_augment,
        #"augmentations_line_1": aug_view_a,
        #"augmentations_line_2": aug_view_b,
        "page_size": (640, 896),
        "line_size": (1232, 64),
        "input_channels": 3,  # number of channels of input image
        "dropout": 0.1,  # dropout rate for encoder
        "encoder_type": FCN_Encoder,
        #"name": SimCLRDoc,
        "feat_channels": 256,
        "proj_dim": 128,
        "tau": 0.25,
    },
    "training_params":{
        "batch_size_line": 40,
        "batch_size_page": 8,
        "num_epochs": 100,
#    "training":{
#        "batch_size_line": 16,
#        "batch_size_page": 8,
#        "num_epochs": 100,
#    "training_params":{
#        "batch_size_line": 128,
#        "batch_size_page": 6,
#        "num_epochs": 205,
        "lr": 1e-3,
        "use_amp": True, # automatic mixed precision
        "output_dir":"outputs/IAM_contrastive_new_aug__",
        "tensorboard_dir":"outputs/IAM_contrastive_new_aug__/results",
        "save_images": None, # "outputs/IAM_contrastive_new_aug/images",
        "optimizers": {
            "all": {
                "class": torch.optim.AdamW,
                "args": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "amsgrad": False,
                }
            },
        },
        "training_mode":{
            "schedule": [
                {"max_epoch": 50, "prob_line": 1.1},
                {"max_epoch": 100, "prob_line": 1.1},
                {"max_epoch": 200, "prob_line": 1.1},
                {"max_epoch": 300, "prob_line": 1.1},
            ]
        },
    },
    "paths":{
        "page_files":"/home/michel/dev/python/formatted/IAM_page",
        "line_files":"/home/michel/dev/python/formatted/IAM_non_syn_line",
    },
}

if __name__ == "__main__":
    imagesPath = "/home/michel/dev/python/raw/IAM/words"
    metadata = "/home/michel/dev/python/DAN/utils/TSNE/words.txt"
    top_n = 10
    samples_per_n = 10
    standard = True
    min_length = 1

    encoder = FCN_Encoder({
        "input_channels": 3,
        "dropout": 0.5,
    })
    init = True
    if init:
        print("init:")
        checkpoint = torch.load(
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_noscale/simclr_epoch1.pth", #simclr_epoch200.pth",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_noscale/simclr_epoch200.pth",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR10/best.pt",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_new_aug/best.pt",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR11/best.pt",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR12/best.pt",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR13/best.pt",
            #"/home/michel/dev/python/DAN/outputs/FCN_IAM_line_non_syn/checkpoints/best.pt",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR15/best.pt",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR17/best.pt",
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_new_aug/best.pt",
            "/home/michel/dev/python/SparK/IAM_line_25perc_random/best.pt",
            map_location="cpu",
            weights_only=False
        )
        if "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        else:
            encoder.load_state_dict(checkpoint)
    w = next(encoder.parameters()).detach().cpu()
    print(f"mean is {w.mean()}, std={w.std()}")

    dataset = words(params["model"], imagesPath, metadata, min_length = min_length)
    print(f"dataset size is {len(dataset)}")
    most_used = dataset.most_used_descriptions(dataset, top_n)
    most_used_capped = dataset.cap_samples_per_label(most_used, labels_subset=set(item["label"] for item in most_used),
                                  n=samples_per_n,
                                  seed=42)
    max_width = max(item["img"].shape[2] for item in most_used_capped)
    max_height = max(item["img"].shape[1] for item in most_used_capped)
    print(f"Max image size for most used words: {max_width}x{max_height}")
    min_width = min(item["img"].shape[2] for item in most_used_capped)
    min_height = min(item["img"].shape[1] for item in most_used_capped)
    print(f"Min image size for most used words: {min_width}x{min_height}")

    encoder.eval()
    features = []
    labels = []
    texts = []

    with torch.no_grad():
        for item in tqdm(most_used_capped):
            img = item["img"].unsqueeze(0)  # (1, C, H, W)
            label = item["label"]

            if standard:
                img = (img - img.mean()) / (img.std() + 1e-6)

            z = encoder(img)  # shape: (1, D) or (1, C, H, W)
            
            # If encoder outputs feature maps, pool to a vector
            if z.ndim == 4:
                z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1))
                z = z.view(z.size(0), -1)

            features.append(z.squeeze(0).cpu().numpy())
            labels.append(label)
            texts.append(label)

    X = np.stack(features)  # (N, D)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_    

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    tsne = TSNE(
        n_components=2,
        perplexity=15,        # you have ~top_n * samples_per_n points
        learning_rate=200,
        n_iter_without_progress=1500,
        init="pca",
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_pca)    
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y,
        cmap="tab10",
        s=25,
        alpha=0.8
    )

    handles, _ = scatter.legend_elements()
    # determine the average x,y per catergory and draw and elipse around all points of that category in the correct color.
    # All points of a category should be in the elipse.

    for i, class_name in enumerate(class_names):
        class_points = X_tsne[y == i]
        if len(class_points) > 1:
            x_mean = np.mean(class_points[:, 0])
            y_mean = np.mean(class_points[:, 1])
            ellipse_width = np.max(class_points[:, 0])-np.min(class_points[:, 0])
            ellipse_height = np.max(class_points[:, 1])-np.min(class_points[:, 1])
            ellipse = plt.matplotlib.patches.Ellipse(
                (x_mean, y_mean),
                width=ellipse_width,
                height=ellipse_height,
                edgecolor=scatter.cmap(i / len(class_names)),
                facecolor='none',
                alpha=0.5
            )
            plt.gca().add_patch(ellipse)
    plt.legend(handles, class_names, title="Words", loc="best", fontsize=9)

    plt.title("t-SNE of IAM Word Embeddings (FCN Encoder)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()
