import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
sys.path.append(os.path.dirname(os.path.dirname(DOSSIER_PARENT)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DOSSIER_PARENT))))

from torch.utils.data import DataLoader
from OCR.document_OCR.Contrastive.Dataset import TwoViewWrapper, default_doc_augment, DocFolder, aug_lines, aug_lines_weak, aug_view_a, aug_view_b
#from Dataset import TwoViewWrapper, default_doc_augment, DocFolder
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
import torch
#from torch._six import inf
from typing import Union, Iterable
import cProfile
import pstats

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def randint(low, high):
    """
    call torch.randint to preserve random among dataloader workers
    """
    return int(torch.randint(low, high, (1, )))

def pad_image(image, padding_value, new_height=None, new_width=None, pad_width=None, pad_height=None, padding_mode="br", return_position=False):
    """
    data: list of numpy array
    mode: "br"/"tl"/"random" (bottom-right, top-left, random)
    """
    if pad_width is not None and new_width is not None:
        raise NotImplementedError("pad_with and new_width are not compatible")
    if pad_height is not None and new_height is not None:
        raise NotImplementedError("pad_height and new_height are not compatible")

    h, w, c = image.shape
    #print(f"pad_with: {pad_width}, pad_height: {pad_height}, new_width: {new_width}, new_height: {new_height}")
    pad_width = pad_width if pad_width is not None else max(0, new_width - w) if new_width is not None else 0
    pad_height = pad_height if pad_height is not None else max(0, new_height - h) if new_height is not None else 0

    if not (pad_width == 0 and pad_height == 0):
        padded_image = np.ones((h+pad_height, w+pad_width, c)) * padding_value
        if padding_mode == "br":
            hi, wi = 0, 0
        elif padding_mode == "tl":
            hi, wi = pad_height, pad_width
        elif padding_mode == "random":
            hi = randint(0, pad_height) if pad_height >= 1 else 0
            wi = randint(0, pad_width) if pad_width >= 1 else 0
        else:
            raise NotImplementedError("Undefined padding mode: {}".format(padding_mode))
        padded_image[hi:hi + h, wi:wi + w, ...] = np.asarray(image)
        output = padded_image
    else:
        hi, wi = 0, 0
        output = image

    if return_position:
        return output, [[hi, hi+h], [wi, wi+w]]
    return output


image_counter = 0
def collate_two_views(batch: List[Dict], mean_std, multiple_of: Tuple[int,int]=(32,8), pad_value: float=0.0):
    global image_counter
    def stack_and_pad(key: str):
        padding_value = 0.0
        hs = [b[key].shape[0] for b in batch] # H, W, C
        ws = [b[key].shape[1] for b in batch]
        Hmax, Wmax = max(hs), max(ws)
        mh, mw = multiple_of
        xs, ms = [], []
        
        i = 0
        for b in batch:
            if isinstance(b[key], np.ndarray):
                x = torch.from_numpy(b[key]).float()
            else:
                x = b[key].clone().float()   # correct way to copy a tensor

            H_orig, W_orig = x.shape[:2]
            x = pad_image(x, (0.0-mean_std[0])/mean_std[1], Hmax, Wmax)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            x = x.permute(2, 0, 1).float()   # C, H, W
            # correct mask
            m = torch.zeros(1, Hmax, Wmax, dtype=torch.float32)
            m[:, :H_orig, :W_orig] = 1.0

            xs.append(x)
            ms.append(m)
            i = i + 1
        return torch.stack(xs,0), torch.stack(ms,0)
    v1,m1 = stack_and_pad("view1")
    v2,m2 = stack_and_pad("view2")
    return {"view1": v1, "mask1": m1, "view2": v2, "mask2": m2}

def masked_global_avg_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # feat: (B,C,H,W), mask: (B,1,H,W) where 1=valid
    num = (feat * mask).sum(dim=(2,3))
    den = mask.sum(dim=(2,3)).clamp_min(1.0)
    return num / den

class ProjectionMLP_old(nn.Module):
    def __init__(self, in_dim, hidden=2048, out_dim=256, use_bn=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden) if use_bn else nn.Identity()
        self.fc2 = nn.Linear(hidden, out_dim)


        '''
        projector = nn.Sequential(
        nn.Linear(h_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, proj_dim)
        )
        '''

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return F.normalize(x, dim=1)
    
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden=2048, out_dim=256, use_bn=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=1)
        
'''
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float=0.2) -> torch.Tensor:
    # z1,z2: (B,D) unit-norm
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # (2B,D)
    sim = (z @ z.t()) / tau                        # cosine similarities
    self_mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # (2B,)
    denom = torch.exp(sim)[~self_mask].view(2*B, -1).sum(dim=1)
    return -(pos - torch.log(denom)).mean()
'''
def nt_xent_optimized(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.t()) / tau         # cosine similarities

    # Create ground truth labels for cross-entropy
    # The first B samples (z1) have positives at indices B to 2B-1
    # The next B samples (z2) have positives at indices 0 to B-1
   # labels = torch.arange(2 * B, device=z.device, dtype=torch.long)
   # labels[::2] += 1  # For even indices, positive is next (in z2 half)
   # labels[1::2] -= 1  # For odd indices, positive is previous (in z1 half)
    # A simpler way to get targets:
    # targets = torch.cat([torch.arange(B, device=z.device) + B, torch.arange(B, device=z.device)], dim=0)

    # We need to mask out the self-similarities (diagonal elements) before the softmax,
    # as they should not contribute to the denominator.
    # The standard F.cross_entropy expects raw logits (sim in this case) and handles
    # the log-softmax calculation internally in a numerically stable way.
    # However, F.cross_entropy does not handle masking of the diagonal for us.
    # We can use the implementation trick from SimCLR paper which handles this implicitly
    # by shaping the inputs and targets correctly, or just use the mask trick with the manual calculation.

    # Reverting to the efficient manual way that avoids issues with F.cross_entropy's label expectations
    # when you have a 2B x 2B similarity matrix:

    # Mask out self-similarities
    self_mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim_no_self = sim.masked_fill(self_mask, float('-inf'))

    # Calculate log-probabilities using logsumexp for stability
    # log( exp(pos) / sum(exp(all_negatives + pos)) ) = pos - logsumexp(all_negatives + pos)
    # The numerator is the positive pair similarity (pos)
    pos_mask = self_mask.roll(shifts=B, dims=0) | self_mask.roll(shifts=-B, dims=0)
    pos_sim = sim[pos_mask].view(2 * B, 1) # Shape (2B, 1)

    # Use logsumexp for the denominator calculation for numerical stability
    # F.log_softmax does log(exp(x)/sum(exp(x))) along a dim
    log_probs = F.log_softmax(sim_no_self, dim=1)

    # The loss is the negative of the log probability of the positive pair
    loss = -log_probs[pos_mask].view(2 * B, 1)
    
    return loss.mean()

class SimCLRDoc(nn.Module):
    def __init__(self, encoder: nn.Module, feat_channels: int=256, proj_dim: int=128, tau: float=0.2):
        super().__init__()
        self.encoder = encoder
        self.proj = ProjectionMLP(in_dim=feat_channels, hidden=512, out_dim=proj_dim, use_bn=True)
        self.tau = tau
        #self.scale = nn.Parameter(torch.tensor(2.5))


    @torch.no_grad()
    def _compute_details(self, z1, z2, topk: int = 5):
        """
        z1, z2 are (B, D) and should be L2-normalized.
        Returns a dict of cheap monitoring stats (all detached).
        """
        # Positive cosine similarities (pairwise)
        pos_sim = (z1 * z2).sum(dim=1)  # (B,)

        # Build similarity blocks
        # sims11 and sims22 include self-sim; we'll mask the diagonal.
        sims11 = z1 @ z1.T  # (B,B)
        sims22 = z2 @ z2.T
        sims12 = z1 @ z2.T
        sims21 = z2 @ z1.T

        B = z1.size(0)
        eye = torch.eye(B, device=z1.device, dtype=torch.bool)

        # Collect negatives for each anchor as a single matrix per anchor set
        # For anchors in z1, negatives are (z1 vs z1, excluding diag) and (z1 vs z2, excluding the matching pair)

        fill = torch.finfo(sims11.dtype).min   # for fp16 this is -65504
        negs_from_z1 = torch.cat([sims11.masked_fill(eye, fill), sims12], dim=1)
        #negs_from_z1 = torch.cat([sims11.masked_fill(eye, -9e15), sims12], dim=1)  # (B, 2B)
        # For anchors in z2 similarly:
        fill = torch.finfo(sims22.dtype).min
        negs_from_z2 = torch.cat([sims22.masked_fill(eye, fill), sims21], dim=1)
        #negs_from_z2 = torch.cat([sims22.masked_fill(eye, -9e15), sims21], dim=1)  # (B, 2B)

        # Hard-negative proxy: mean of top-k negatives for each anchor (avoid +∞ by clamping)
        k = min(topk, negs_from_z1.size(1))
        hard_z1 = torch.topk(negs_from_z1, k=k, dim=1).values.mean(dim=1)  # (B,)
        hard_z2 = torch.topk(negs_from_z2, k=k, dim=1).values.mean(dim=1)  # (B,)
        neg_sim_mean = torch.stack([hard_z1, hard_z2], dim=0).mean(dim=0)  # (B,)

        # Embedding norm BEFORE L2 norm (if your ProjectionMLP returns unnormalized too)
        # If your ProjectionMLP already L2-normalizes, skip this; otherwise compute from raw projection.
        # Here we treat z1/z2 as already normalized; compute their pre-norm magnitude from arccos trick not needed.
        # Instead, report post-norm (should be ~1).
        emb_norm = torch.cat([z1.norm(dim=1), z2.norm(dim=1)], dim=0)  # (2B,)

        # Collapse check: eigenvalues of covariance of normalized features
        z = torch.cat([z1, z2], dim=0)                    # (2B, D)
        zc = F.normalize(z, dim=1)                        # ensure unit-length
        cov = (zc.T @ zc) / zc.size(0)                    # (D, D)
        # numerical safety
        cov_f = cov.float()                 # fp32
        eigvals = torch.linalg.eigvalsh(cov_f)
        eigvals = eigvals.clamp_min(1e-12)

        # eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)

        # Alignment & uniformity (Wang & Isola) on the batch
        align = ((z1 - z2).pow(2).sum(dim=1)).mean()
        # Uniformity expects normalized embeddings
        with autocast('cuda', enabled=params["training_params"]["use_amp"]):
            # compute in fp32 for stability
            z_float = zc.float()
            dists = torch.cdist(z_float, z_float, p=2)
            unif = torch.log(torch.exp(-2 * dists.pow(2)).mean())

        return {
            "z1": z1.detach(),
            "z2": z2.detach(),
            "pos_sim": pos_sim.detach(),
            "neg_sim_mean": neg_sim_mean.detach(),
            "emb_norm": emb_norm.detach(),
            "eigvals": eigvals.detach(),
            "alignment": align.detach(),
            "uniformity": unif.detach(),
        }
    def forward(self, v1, m1, v2, m2, return_details: bool = False):
        """
        Returns:
          - if return_details=False (default): loss
          - if return_details=True: (loss, details_dict)
        """
        # Encode
        #scale = nn.Parameter(torch.tensor(2.5))
        f1 = self.encoder(v1) #* scale  # (B,Cf,Hf,Wf)
        #print("SSL mean/std/max:", f1.mean().item(), f1.std().item(), f1.abs().max().item())
        f2 = self.encoder(v2) #* scale  

        # Resize masks to feature map size
        m1f = F.interpolate(m1, size=f1.shape[-2:], mode='nearest')
        m2f = F.interpolate(m2, size=f2.shape[-2:], mode='nearest')

        # Masked global pooling -> (B,Cf)
        g1 = masked_global_avg_pool(f1, m1f)
        g2 = masked_global_avg_pool(f2, m2f)

        # Projection head -> (B,D)
        z1 = self.proj(g1)  # returns L2-normalized vectors
        #print("z std:", z1.std(dim=0).mean().item())
        z2 = self.proj(g2)

        #pos_sim = (z1 * z2).sum(dim=1).mean()
        #neg_sim = (z1 @ z2.T).mean()

        #print("pos:", pos_sim.item(), "neg:", neg_sim.item())
        # If your ProjectionMLP does NOT L2-normalize, do it here:
        #z1 = F.normalize(z1, dim=1)
        #z2 = F.normalize(z2, dim=1)

        #loss = nt_xent(z1, z2, tau=self.tau)
        loss = nt_xent_optimized(z1, z2, tau=self.tau)

        if not return_details:
            return loss

        # Compute lightweight monitoring stats without tracking gradients
        with torch.no_grad():
            details = self._compute_details(z1, z2)

        return loss, details
    
    @torch.no_grad()
    def encode(self, x, m=None):
        """Convenience: get pooled & projected features for evaluation."""
        f = self.encoder(x)
        if m is not None:
            m = F.interpolate(m, size=f.shape[-2:], mode='nearest')
            g = masked_global_avg_pool(f, m)
        else:
            g = f.mean(dim=(2,3))
        z = self.proj(g)
        return F.normalize(z, dim=1)


def load_latest_checkpoint(model, optimizer=None, ckpt_dir="outputs/IAM_contrastive"):
    """
    Load the latest checkpoint from a directory.
    
    Args:
        model (nn.Module): Your model instance.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore state.
        ckpt_dir (str): Directory containing checkpoints.
        
    Returns:
        model, optimizer, start_epoch
    """
    ckpts = glob.glob(os.path.join(ckpt_dir, "simclr_epoch*.pth"))
    if not ckpts:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0
    
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"Resuming from checkpoint: {latest_ckpt}")
    
    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        # Full checkpoint with optimizer + epoch
        model.load_state_dict(checkpoint["model"])
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
    else:
        # Old style: only state_dict
        model.load_state_dict(checkpoint)
        start_epoch = 0
    
    return model, optimizer, start_epoch

def determine_line_probability(epoch: int, schedule: List[Dict]) -> float:
    for entry in schedule:
        if epoch < entry["max_epoch"]:
            return entry["prob_line"]
    return schedule[-1]["prob_line"]

def create_directories(params):
    output_dir = params["training_params"]["output_dir"]
    tensorboard_dir = params["training_params"]["tensorboard_dir"]
    images_dir = params["training_params"]["save_images"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    if images_dir is not None:
        os.makedirs(images_dir, exist_ok=True)

def detect_image_format(t):
    t = t.float()
    mn, mx = t.min().item(), t.max().item()

    if mx > 200:                           # likely 0–255
        return "0-255 (unnormalized)"
    if 0 <= mn >= -1 and mx <= 1:
        # check if distribution suggests mean/std normalization
        if mn < 0 or mx > 1:
            return "Normalized (mean/std)"
        return "0-1 scaled"
    if mn < 0 and mx < 5:
        return "Normalized (mean/std)"

    return "Unknown format"

params = {
    "model":{
        "preprocessor_page":[
                {
                    "type": "to_RGB",
                },
            ],
        "preprocessor_line":[
                {
                    "type": "to_RGB",
                    "type": "resize_when_bigger", "keep_ratio": True, "max_height": None, "max_width": 1232,
                },
            ],
        "config": {
            "divisor": 1, # divides image in subimages
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
            #"max_size":{
            #    "max_height": 32,
            #    "max_width": 1232,
            #},
            #"padding": {
            #    "min_height": "max",  # Pad to reach max height of training samples
            #    "min_width": "max",  # Pad to reach max width of training samples
            #    "min_pad": None,
            #    "max_pad": None,
            #    "mode": "br",  # Padding at bottom and right
            #    "train_only": False,  # Add padding at training time and evaluation time
            #},
            "preprocessings": [
                {
                    "type": "to_RGB",
                    # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                },
            ],
        },
        "augmentations_page": default_doc_augment,
        "augmentations_line_1": aug_view_a,
        "augmentations_line_2": aug_view_a,
        "page_size": (640, 896),
        "line_size": (1232, 32),
        "input_channels": 3,  # number of channels of input image
        "dropout": 0.1,  # dropout rate for encoder
        "encoder_type": FCN_Encoder,
        "name": SimCLRDoc,
        "feat_channels": 256,
        "proj_dim": 128,
        "tau": 0.2,
    },
    #"preload": "/home/michel/dev/python/SparK/IAM_pretrain_L2_224_25perc_patches_set_encoder_A1/best.pt",
    "preload": "/home/michel/dev/python/SparK/READ_pretrain_L2_224_40perc_random_full_set_encoder_A1/best.pt",
    "training_params":{
        "batch_size_line": 64,
        "batch_size_page": 8,
        "num_epochs": 75,
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
        "output_dir_mask": "outputs_seed/READ_MIM_Contrastive_seed_{seed}", # directory to save checkpoints, with mask and batch size info
        "output_dir":"outputs_seed/READ_MIM_Contrastive_seed{seed}", # directory to save checkpoints
        "tensorboard_dir":"outputs_seed/READ_MIM_Contrastive_seed{seed}/results",
        #"tensorboard_dir":f"{params['training_params']['output_dir']}/results",
        "save_images": None, # "outputs/READ_contrastive_bs256_div4/images",
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
        "page_files":"/home/michel/dev/python/formatted/READ_2016_page",
        "line_files":"/home/michel/dev/python/formatted/READ_2016_non_syn_line",
    },
}

def clip_grad_norm_(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be inf for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf') :
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic behavior (optional but recommended for experiments)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Train():
    for seed in [1, 2, 3]:
        set_seed(seed)
        params["training_params"]["output_dir"] = params["training_params"]["output_dir_mask"].format(seed=seed)
        params["training_params"]["tensorboard_dir"] = params["training_params"]["output_dir_mask"].format(seed=seed) + "/results"

        dataset_name = "IAM"


        create_directories(params)
        #train_base_page = DocFolder(params=params["model"], root=params["paths"]["page_files"] , size=params["model"]["page_size"], custom_name=dataset_name, train=True, rgb=False, 
        #                        preprocessings=params["model"]["preprocessor_page"], load_in_memory = True) 
        train_base_line = DocFolder(params=params["model"], root= params["paths"]["line_files"], size=params["model"]["line_size"], custom_name=dataset_name, train=True, rgb=True, 
                                preprocessings=params["model"]["preprocessor_line"], load_in_memory=True) 
        #train_ds_page = TwoViewWrapper(train_base_page, augA=params["model"]["augmentations_page"], augB=params["model"]["augmentations_page"])
        train_ds_line = TwoViewWrapper(train_base_line, augA=params["model"]["augmentations_line_1"], augB=params["model"]["augmentations_line_2"])
        #train_loader_page = DataLoader(
        #    train_ds_page, batch_size=params["training_params"]["batch_size_page"], shuffle=True, num_workers=4, pin_memory=True,
        #    collate_fn=lambda b: collate_two_views(b, mean_std=train_ds_page.get_mean_std(), multiple_of=(32,8)), drop_last=True
        #)
        train_loader_page=None
        train_loader_line = DataLoader(
            train_ds_line, 
            batch_size=params["training_params"]["batch_size_line"], 
            shuffle=True, 
            num_workers=8, 
            pin_memory=False,
            collate_fn=lambda b: collate_two_views(b, mean_std=train_base_line.get_mean_std(), 
            multiple_of=(32,8)), 
            drop_last=True, 
            worker_init_fn=seed_worker,
            #persistent_workers=True,
        )
        encoder = params["model"]["encoder_type"](params["model"])
        if params["preload"] is not None:
            checkpoint = torch.load(params["preload"], 
                map_location="cpu",
                weights_only=False)
            incompatibleKeys = []
            if "encoder_state_dict" in checkpoint:
                incompatibleKeys = encoder.load_state_dict(checkpoint["encoder_state_dict"])
                print("Loaded encoder state dict from checkpoint ")
            else:
                incompatibleKeys = encoder.load_state_dict(checkpoint)
                print("Loaded encoder state dict from checkpoint ")
            print(f"Incompatible keys when loading checkpoint: {incompatibleKeys}")
        model = params["model"]["name"](
                encoder=encoder, 
                feat_channels=params["model"]["feat_channels"], 
                proj_dim=params["model"]["proj_dim"], 
                tau=params["model"]["tau"])

        opt = torch.optim.AdamW([
            {"params": encoder.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
            {"params": model.proj.parameters(), "lr": 1e-3, "weight_decay": 1e-3},
            ])

        '''
        opt = params["training_params"]["optimizers"]["all"]["class"](model.parameters(), 
                lr=params["training_params"]["optimizers"]["all"]["args"]["lr"], 
                weight_decay=params["training_params"]["optimizers"]["all"]["args"]["weight_decay"])
        '''        
        writer = SummaryWriter(params["training_params"]["tensorboard_dir"])
        model.train()
        model, opt, start_epoch = load_latest_checkpoint(model, opt, params["training_params"]["output_dir"])
        epoch = start_epoch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if opt is not None:
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        scaler = GradScaler()

        while epoch < params["training_params"]["num_epochs"]:
            t0 = time.time()
            loss_vals = []
            pos_sims, neg_sims = [], []
            emb_norms = []
            grad_norms = []
            var_mins, var_means, var_maxs = [], [], []
            count = 0
            p_line = determine_line_probability(epoch, params["training_params"]["training_mode"]["schedule"])
            line = 1 if random.random() < p_line else 0
            
            for batch in train_loader_line if line == 1 else train_loader_page:
                print(f"\rProgress: {count}", end="", flush=True)
                count = count + 1
                v1 = batch["view1"].to(device, non_blocking=True)
                v2 = batch["view2"].to(device, non_blocking=True)
                m1 = batch["mask1"].to(device, non_blocking=True)
                m2 = batch["mask2"].to(device, non_blocking=True)
                # print(detect_image_format(v1))
                if params["training_params"]["save_images"] is not None:
                    images = batch["view1"].cpu()
                    for i in range(images.size(0)):
                        if line ==1:    
                            train_base_line.convertToImage(images[i]).save(f"{params["training_params"]["save_images"]}/SSL_PIL_image_{i}1.png")
                        else:
                            train_base_page.convertToImage(images[i]).save(f"{params["training_params"]["save_images"]}/SSL_PIL_image_{i}1.png")
                    images = batch["view2"].cpu()
                    for i in range(images.size(0)):
                        if line == 1:
                            train_base_line.convertToImage(images[i]).save(f"{params["training_params"]["save_images"]}/SSL_PIL_image_{i}2.png")
                        else:
                            train_base_page.convertToImage(images[i]).save(f"{params["training_params"]["save_images"]}/SSL_PIL_image_{i}2.png")
                
                opt.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=params["training_params"]["use_amp"]):
                    loss, out = model(v1, m1, v2, m2, return_details=True)
                #with torch.no_grad():
                                #img = sample_batch.to(device)
                                #f_ssl = ssl_encoder(img)
                                #f_base = base_encoder(img)

                
                            #print("BASE mean/std/max:", x.mean().item(), x.std().item(), x.abs().max().item())


                #loss.backward()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                #model.parameters().isinf().any()
                #grad_norm = torch.nn.utils.clip_grad_norm_(
                #    model.parameters(), max_norm=1e9
                #).item()
                #grad_norm = clip_grad_norm_(
                #    model.parameters(), max_norm=1e9
                #).item()
                #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9).item()
                #opt.step()
                scaler.step(opt)
                scaler.update()
                loss_vals.append(loss.item())
                #grad_norms.append(grad_norm)

                z1, z2 = out["z1"].detach(), out["z2"].detach()  # (B, D)
                if "pos_sim" in out: pos_sims.append(out["pos_sim"].detach().mean().item())
                if "neg_sim_mean" in out: neg_sims.append(out["neg_sim_mean"].detach().mean().item())
                emb = torch.cat([z1, z2], dim=0)
                emb_norms.append(emb.norm(dim=1).mean().item())

                zn = torch.nn.functional.normalize(emb, dim=1)
                cov = (zn.T @ zn) / zn.size(0)
                '''
                eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
                var_mins.append(eigvals.min().item())
                var_means.append(eigvals.mean().item())
                var_maxs.append(eigvals.max().item())
                '''
            epoch_time = time.time() - t0
            step = epoch + 1
            print(f"\rEpoch {epoch+1} done in {epoch_time:.1f}s, loss {sum(loss_vals)/len(loss_vals):.4f}")
            writer.add_scalar("train/loss_mean", sum(loss_vals)/len(loss_vals), step)
            writer.add_scalar("train/loss_std", torch.tensor(loss_vals).std().item(), step)
            writer.add_scalar("opt/lr", opt.param_groups[0]["lr"], step)
            writer.add_scalar("time/epoch_seconds", epoch_time, step)
            #writer.add_scalar("grad/global_norm_mean", sum(grad_norms)/len(grad_norms), step)

            if pos_sims:
                writer.add_scalar("repr/pos_cosine_mean", sum(pos_sims)/len(pos_sims), step)
            if neg_sims:
                writer.add_scalar("repr/neg_cosine_mean", sum(neg_sims)/len(neg_sims), step)
            writer.add_scalar("repr/emb_norm_mean", sum(emb_norms)/len(emb_norms), step)
            #writer.add_scalar("collapse/eig_min_mean", sum(var_mins)/len(var_mins), step)
            #writer.add_scalar("collapse/eig_mean_mean", sum(var_means)/len(var_means), step)
            #writer.add_scalar("collapse/eig_max_mean", sum(var_maxs)/len(var_maxs), step)

            # histogram snapshots once in a while
            #if step % 5 == 0:
            #    writer.add_histogram("repr/eigvals", eigvals.cpu(), step)

            torch.save({
                "epoch": epoch,
                "encoder_state_dict": model.encoder.state_dict(),
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
            }, f"{params["training_params"]["output_dir"]}/simclr_epoch{epoch+1}.pth")
            epoch = epoch + 1
            line = 1-line


cProfile.run("Train()", "stats")

p = pstats.Stats("stats")
p.sort_stats("cumtime").print_stats(20)
