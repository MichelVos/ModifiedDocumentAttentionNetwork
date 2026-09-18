"""
compare.py

Compare the learned weights of a self-supervised encoder and a supervised
encoder by measuring the similarity between their corresponding parameters.
This utility loads two checkpoint files, extracts their state dictionaries when
available, and iterates over shared tensor keys to compute cosine similarity and
L2 distance. The resulting statistics provide a lightweight indication of how
closely the latent representations align across different training paradigms.

Typical use:
    - compare SSL and supervised encoder checkpoints
    - analyze weight similarity across training strategies
    - inspect the relationship between self-supervised and supervised feature learning

This script is a diagnostic and analysis utility rather than a core training
or evaluation component.
"""

import torch
import os
from pathlib import Path

ssl = torch.load(f"{Path.home()}/tmp/IAM_contrastive/simclr_epoch101.pth", map_location="cpu")
sup = torch.load(f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_syn/checkpoints/best_202.pt", map_location="cpu", weights_only=False)

# If they have a "state_dict"
ssl_sd = ssl["encoder_state_dict"] if "encoder_state_dict" in ssl else ssl
sup_sd = sup["encoder_state_dict"] if "encoder_state_dict" in sup else sup

common_keys = ssl_sd.keys() & sup_sd.keys()

for k in common_keys:
    w1 = ssl_sd[k].flatten()
    w2 = sup_sd[k].flatten()
    cos = torch.nn.functional.cosine_similarity(w1, w2, dim=0)
    l2  = torch.norm(w1 - w2).item()
    print(f"{k}: cosine={cos:.4f}, l2={l2:.4f}")
