import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
#sys.path.append(os.path.dirname(os.path.dirname(DOSSIER_PARENT)))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DOSSIER_PARENT))))
from torchviz import make_dot


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


matplotlib.use("TkAgg")

activations = {}

'''
load image.
Image is converted to RGB if needed, then to tensor and normalized to [0,1]
'''
#img=Image.open("/home/michel/dev/python/formatted/IAM_non_syn_line/train/train_0.jpeg").convert("RGB")
img=Image.open("/home/michel/Labour.jpeg").convert("RGB")
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
image = transform(img)  # shape: [C, H, W], values in [
image = image.unsqueeze(0)  # add batch dimension, shape: [1, C, H, W]
model = FCN_Encoder({
    "input_channels": 3,
    "dropout": 0.5,
})

#y = model(image)
#make_dot(y, params=dict(model.named_parameters())).render("fcn_encoder", format="png")
#torch.onnx.export(model, image, "fcn_encoder.onnx")
from torchinfo import summary
C=3
H=32
W=1232
summary(model, input_size=(1, C, H, W))
from torchviz import make_dot
y = model(image)
make_dot(y).render("fcn_encoder_graph", format="png")

if True:
    checkpoint = torch.load(
        "/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_noscale/simclr_epoch1.pth", #simclr_epoch200.pth",
        map_location="cpu",
        weights_only=False
    )
else:
    checkpoint = torch.load(
        "/home/michel/dev/python/DAN/outputs/FCN_IAM_line_non_syn/checkpoints/best.pt",
        map_location="cpu",
        weights_only=False
    )

model.load_state_dict(checkpoint["encoder_state_dict"])


def hook_fn(module, input, output):
    activations["feat"] = output.detach()

# Example target layer
target_layer = model.init_blocks[0].conv1
target_layer.register_forward_hook(hook_fn)

model.eval()
with torch.no_grad():
    _ = model(image)  # [1, C, H, W]

'''
feat = activations["feat"][0]  # shape: [C, H, W]

# Visualize feature maps
num_feature_maps = feat.shape[0]
n_cols = 8
n_rows = math.ceil(num_feature_maps / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))      
'''







feat = activations["feat"][0]   # [C, H, W]

num_feature_maps = feat.shape[0]
n_cols = 8
n_rows = math.ceil(num_feature_maps / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
axes = axes.flatten()

for i in range(num_feature_maps):
    axes[i].imshow(feat[i].cpu(), cmap="gray")
    axes[i].axis("off")

# Hide unused subplots
for i in range(num_feature_maps, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()
