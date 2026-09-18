"""
Layer-wise CKA Analysis for DAN Encoders
========================================

This script compares the internal representations learned by two
FCN_Encoder models from the Document Attention Network (DAN) using
Centered Kernel Alignment (CKA).

The purpose is to measure how similar the representations are at
different layers of the encoder. This can be used, for example, to
compare:

    - a self-supervised pretrained encoder,
    - a supervised trained encoder,
    - a randomly initialized encoder,
    - or encoders at different stages of training.

Forward hooks are attached to selected encoder layers to capture their
activations for the same input batch. The activations are flattened and
converted to centered Gram matrices, after which linear CKA is computed.

CKA scores range approximately from 0 to 1:

    1.0  -> highly similar representations
    0.0  -> little representational similarity

This analysis can be used to investigate whether representations learned
during self-supervised pretraining are preserved, transformed, or
overwritten during subsequent supervised CTC training.

Expected input:
    A batch of document/line images with shape [N, C, H, W].

Output:
    A dictionary containing the CKA similarity score for each selected
    encoder layer.

"""

import torch, torch.nn.functional as F
import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
sys.path.append(os.path.dirname(os.path.dirname(DOSSIER_PARENT)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DOSSIER_PARENT))))

from torch.utils.data import DataLoader
from OCR.line_OCR.Contrastive.Dataset import TwoViewWrapper, default_doc_augment, DocFolder
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
import matplotlib.pyplot as plt


def hookify(model, layer_names):
    feats = {n: None for n in layer_names}
    handles = []
    for n in layer_names:
        m = dict(model.named_modules())[n]
        handles.append(m.register_forward_hook(lambda nm: (lambda _, __, o: feats.__setitem__(nm, o.detach()))(n)))
    return feats, handles

def gram_linear(x):  # x: [N, D] or [N, C, H, W]
    if x.dim() > 2:
        x = x.flatten(1)
    x = x - x.mean(0, keepdim=True)
    return x @ x.t()

def cka(x, y):
    K = gram_linear(x); L = gram_linear(y)
    hsic = (K*L).sum()
    norm = torch.linalg.matrix_norm(K) * torch.linalg.matrix_norm(L)
    return (hsic / (norm + 1e-12)).item()

@torch.no_grad()
def layerwise_cka(model_a, model_b, layer_names, batch):
    fa, ha = hookify(model_a, layer_names)
    fb, hb = hookify(model_b, layer_names)
    model_a.eval(); model_b.eval()
    _ = model_a(batch); _ = model_b(batch)
    out = {}
    for n in layer_names:
        xa, xb = fa[n], fb[n]
        out[n] = cka(xa.flatten(1), xb.flatten(1))
    for h in ha+hb: h.remove()
    return out



# Example usage:
# layers = ["init_blocks.0", "init_blocks.1", "blocks.0", "blocks.1", "blocks.2", "blocks.3"]
# cka_scores = layerwise_cka(model_ssl, model_sup, layers, batch)  # batch: [N,C,H,W]
# print(cka_scores)
model_ssl = FCN_Encoder(params={"input_channels":3, "dropout":0.1})
model_sup = FCN_Encoder(params={"input_channels":3, "dropout":0.1})
ssl_path = f"{Path.home()}/tmp/IAM_contrastive/simclr_epoch101.pth"
sup_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_syn/checkpoints/best_202.pt"
model_ssl.load_state_dict(torch.load(ssl_path, map_location="cpu")["encoder_state_dict"])
model_sup.load_state_dict(torch.load(sup_path, map_location="cpu", weights_only=False)["encoder_state_dict"])


layers = ["init_blocks.0", "init_blocks.1", "init_blocks.2", "init_blocks.3", "init_blocks.4", "init_blocks.5", "blocks.0","blocks.1", "blocks.2", "blocks.3",] 
cka_scores = layerwise_cka(model_ssl, model_sup, layers, batch)  # batch: [N,C,H,W]
print(cka_scores)