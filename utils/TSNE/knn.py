
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


def knn_predict(z_query, Z, Y, k=5, num_classes=10):
    z_query = z_query / z_query.norm()

    sims = torch.matmul(Z, z_query)      # [N]
    topk = torch.topk(sims, k=k)

    scores = torch.zeros(num_classes)
    for sim, idx in zip(topk.values, topk.indices):
        scores[Y[idx]] += sim

    return scores.argmax().item()


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
        "page_size": (640, 896),
        "line_size": (1232, 64),
        "input_channels": 3,  # number of channels of input image
        "dropout": 0.1,  # dropout rate for encoder
        "checkpoint_path": None,
        "encoder_type": FCN_Encoder,
        "feat_channels": 256,
        "proj_dim": 128,
        "tau": 0.25,
    },
    "training_params":{
        "batch_size_line": 40,
        "batch_size_page": 8,
        "num_epochs": 100,
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
    samples_per_n = 200
    standard = True

    encoder = FCN_Encoder({
        "input_channels": 3,
        "dropout": 0.5,
    })
    init = True
    if init:
        checkpoint = torch.load(
            #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_noscale/simclr_epoch1.pth", #simclr_epoch200.pth",
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
            print("Loaded encoder state dict from checkpoint ")
        else:
            encoder.load_state_dict(checkpoint)
            print("Loaded encoder state dict from checkpoint ")
    w = next(encoder.parameters()).detach().cpu()
    print(f"mean is {w.mean()}, std={w.std()}")

    dataset = words(params["model"], imagesPath, metadata)
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
    print(f"Total number of samples {len(dataset)}")
    encoder.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)

    Z = []      # embeddings
    Y = []      # labels (word strings or indices)

    with torch.no_grad():
        for item in tqdm(most_used_capped):
            x = item["img"]           # tensor [C,H,W]
            y = item["label"]

            x = x.unsqueeze(0).to(device)   # [1,C,H,W]

            #z = encoder(x)                 # [1,C,H',W']
            #z = z.mean(dim=[2, 3])         # [1,C]  ✅ GLOBAL POOLING
            #z = z.squeeze(0)               # [C]
            #z = z / z.norm()               # L2 normalize

            if standard:
                z = encoder(x)                 # [1, C, H, W]
                z = z.mean(dim=[2, 3])        # [1, C]
                z = z.squeeze(0)              # [C]
                z = z / z.norm()        
            else:
                z = encoder(x)        # [1, C, H, W]
                z = z.mean(dim=2)     # [1, C, W]   ← keep sequence
                z = z.flatten(1)      # [1, C·W]
                z = z.squeeze(0)
                z = z / z.norm()


            Z.append(z.cpu())
            Y.append(y)

    Z = torch.stack(Z)   # [1000, D]

    label2idx = {label: i for i, label in enumerate(sorted(set(Y)))}
    idx2label = {i: l for l, i in label2idx.items()}

    Y_idx = torch.tensor([label2idx[y] for y in Y])
    num_classes = len(label2idx)

    encoder.eval()

    correct = 0
    result = []

    for k in range(1, 45, 5):
        correct = 0
        with torch.no_grad():
            for item in tqdm(most_used_capped):
                x = item["img"].unsqueeze(0).to(device)
                y = label2idx[item["label"]]

                
                if standard:
                    zq = encoder(x)                 # [1, C, H, W]
                    zq = zq.mean(dim=[2, 3])        # [1, C]
                    zq = zq.squeeze(0)              # [C]
                    zq = zq / zq.norm()        
                else:
                    zq = encoder(x)        # [1, C, H, W]
                    zq = zq.mean(dim=2)     # [1, C, W]   ← keep sequence
                    zq = zq.flatten(1)      # [1, C·W]
                    zq = zq.squeeze(0)
                    zq = zq / zq.norm()

                pred = knn_predict(zq.cpu(), Z, Y_idx, k=k, num_classes=num_classes)
                correct += int(pred == y)

        acc = correct / len(most_used_capped)
        print(f"kNN accuracy (k={k}) = {acc:.4f}")
        result.append((k, acc))
        # write results to csv file
        with open("seqclr.csv", "w") as f:
            f.write("k,accuracy\n")
            for k_val, acc_val in result:
                f.write(f"{k_val},{acc_val}\n")

