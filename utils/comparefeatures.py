import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
#sys.path.append(os.path.dirname(os.path.dirname(DOSSIER_PARENT)))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DOSSIER_PARENT))))



from torch.utils.data import DataLoader
from OCR.document_OCR.Contrastive.Dataset import TwoViewWrapper, default_doc_augment, DocFolder, aug_lines
#from Dataset import TwoViewWrapper, default_doc_augment, DocFolder
from basic.models import FCN_Encoder
import math
import torch
from torch.amp.autocast_mode import autocast #, GradScaler
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
import matplotlib
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image
from Dataset import DocFolder


dataset = DocFolder(
    root_dir="/home/michel/dev/python/formatted/IAM_non_syn_line/train",
    transform=None,
    augmentations=None,
    line_size=(1232, 64),
    page_size=None,
    input_channels=3,
)




#img=Image.open("/home/michel/Labour.jpeg").convert("RGB")
#transform = torchvision.transforms.Compose([
#    torchvision.transforms.ToTensor(),
#])
#image = transform(img)  # shape: [C, H, W], values in [
#image = image.unsqueeze(0)  # add batch dimension, shape: [1, C, H, W]

images = []
for i in range(10):
    index = random.randint(0, len(dataset)-1)
    img, _ = dataset[index]
    images.append(img)
    




ssl_encoder = FCN_Encoder({
    "input_channels": 3,
    "dropout": 0.5,
})
supervised_encoder = FCN_Encoder({
    "input_channels": 3,
    "dropout": 0.5,
})


checkpoint_ssl = torch.load(
        #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_noscale/simclr_epoch1.pth", #simclr_epoch200.pth",
        #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_newloss_bs128_newaug_tau025/simclr_epoch200.pth",
        #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_noscale/simclr_epoch200.pth",
        #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR/best_90.pt",
        "/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR6/best_176.pt",
        map_location="cpu",
        weights_only=False
    )

checkpoint_supervised = torch.load(
        #"/home/michel/dev/python/DAN/outputs/FCN_IAM_line_non_syn/checkpoints/best.pt",
        "/home/michel/dev/python/DAN/outputs/FCN_IAM_line_10PercSamples/checkpoints/best_96.pt",
        map_location="cpu",
        weights_only=False
    )

ssl_encoder.load_state_dict(checkpoint_ssl["encoder_state_dict"])
supervised_encoder.load_state_dict(checkpoint_supervised["encoder_state_dict"])

ssl_encoder.eval()
supervised_encoder.eval()

with torch.no_grad():
    feats_ssl = ssl_encoder(image)               # before projection head!
    feats_sup = supervised_encoder(image)
print(feats_ssl.norm(dim=1).mean())
print(feats_sup.norm(dim=1).mean())
print("SSL mean:", feats_ssl.mean().item(), "var:", feats_ssl.var().item())
print("SUP mean:", feats_sup.mean().item(), "var:", feats_sup.var().item())
sim = torch.nn.functional.cosine_similarity(
    feats_ssl.flatten(1),
    feats_sup.flatten(1),
    dim=1
)
print("Average similarity:", sim.mean())
