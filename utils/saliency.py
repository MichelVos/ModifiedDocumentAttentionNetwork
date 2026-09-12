import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from OCR.document_OCR.Contrastive.Dataset import TwoViewWrapper, default_doc_augment, DocFolder, aug_lines
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
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # forward hook (store conv activations)
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        # backward hook (store gradients wrt conv activations)
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, target_index=None):
        # forward pass
        output = self.model(x)

        # choose target index (for CTC you pick max logit)
        if target_index is None:
            target_index = output.mean()   # safe default for encoders

        # backward
        self.model.zero_grad()
        target_index.backward(retain_graph=True)

        # gradients: (B, C, H, W)
        grads = self.gradients
        acts = self.activations

        # GAP the gradients to obtain channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # build the CAM
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # normalize to 0–1
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)

        # upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        return cam

img=Image.open("/home/michel/Labour.jpeg").convert("RGB")
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
image = transform(img)  # shape: [C, H, W], values in [
image = image.unsqueeze(0)

model = FCN_Encoder({
    "input_channels": 3,
    "dropout": 0.5,
})
if True:
    checkpoint = torch.load(
        #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_noscale/simclr_epoch1.pth", #simclr_epoch200.pth",
        #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly_newloss_bs128_newaug_tau025/simclr_epoch200.pth",
        #"/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR/best_90.pt",
        "/home/michel/dev/python/DAN/outputs/IAM_contrastive_seqCLR2/best_99.pt",
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


# assume model is FCN_Encoder(...)
model.eval()

# attach Grad-CAM to init_block[0].conv3
target_layer = model.init_blocks[0].conv3
cam = GradCAM(model, target_layer)

# your input: (1, 1, H, W)
heatmap = cam(image)

cam_np = heatmap.squeeze().cpu().numpy()      # (H, W)
cam_np = (cam_np * 255).astype(np.uint8)

cv2.imwrite("heatmap_gray.png", cam_np)