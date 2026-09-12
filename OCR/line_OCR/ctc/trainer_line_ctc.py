#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from basic.metric_manager import MetricManager
from OCR.ocr_manager import OCRManager
from OCR.ocr_utils import LM_ind_to_str
import torch
from torch.amp import autocast
from torch.nn import CTCLoss
import re
import time
import os
import torchvision.utils as vutils



def inspect_sample(ds, idx=0, device="cuda"):
    # 1) get raw sample
    raw = ds[idx]  # adapt: (img, target) or dict
    if isinstance(raw, dict):
        img = raw["image"]
    elif isinstance(raw, (list, tuple)):
        img = raw[0]
    else:
        img = raw

    print("RAW:", type(img))
    try:
        import numpy as np
        if isinstance(img, np.ndarray):
            print(" raw ndarray:", img.dtype, img.shape,
                  "min/max:", img.min(), img.max())
    except Exception: pass

    # 2) apply pipeline (the same transforms your DataLoader uses)
    # If your dataset already returns a tensor, skip this.
    import torchvision.transforms.functional as TF
    if not torch.is_tensor(img):
        # convert to tensor the same way you do in Dataset
        # (adjust: grayscale/RGB, ToTensor, etc.)
        # If using PIL, TF.to_tensor(img) scales to [0,1]
        try:
            t = TF.to_tensor(img)
        except Exception:
            # If already numpy HxWxC or HxW
            import numpy as np
            arr = img.astype("float32")
            if arr.max() > 1.5:  # e.g., uint8/uint16
                arr /= (255.0 if arr.max() <= 255 else 65535.0)
            if arr.ndim == 2:
                arr = arr[None, ...]
            t = torch.from_numpy(arr)
    else:
        t = img

    print("TENSOR pre-norm:", t.dtype, tuple(t.shape),
          "min/max:", float(t.min()), float(t.max()))
    assert torch.isfinite(t).all(), "NaN/Inf in raw tensor!"

    # 3) robust normalization (match your pretraining if you can!)
    # If you used fixed mean/std during pretraining, use those instead.
    x = t.float()
    # auto-scale to [0,1] if needed
    mx = x.amax(dim=(1,2), keepdim=True) if x.dim()==3 else x.max()
    if x.max() > 1.5:
        base = 255.0 if x.max() <= 255 else (65535.0 if x.max() <= 65535 else float(x.max()))
        x = x / max(base, 1.0)
    # per-sample standardization with std clamp
    mu  = x.mean(dim=(1,2), keepdim=True)
    std = x.std(dim=(1,2), keepdim=True).clamp_(min=1e-3)
    x = (x - mu) / std

    print("TENSOR post-norm:", x.dtype, tuple(x.shape),
          "min/max:", float(x.min()), float(x.max()))
    assert torch.isfinite(x).all(), "NaN/Inf after norm!"

    return x

def detect_image_format(t):
    t = t.float()
    mn, mx = t.min().item(), t.max().item()

    if mx > 200:                           # likely 0–255
        return "0–255 (unnormalized)"
    if 0 <= mn >= -1 and mx <= 1:
        # check if distribution suggests mean/std normalization
        if mn < 0 or mx > 1:
            return "Normalized (mean/std)"
        return "0–1 scaled"
    if mn < 0 and mx < 5:
        return "Normalized (mean/std)"

    return "Unknown format"

class TrainerLineCTC(OCRManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)
        self.batchCount = 0
    '''
    def train_batch(self, batch_data, metric_names):
        """
        Forward and backward pass for training
        """
        x = batch_data["imgs"].to(self.device)
        # x = inspect_sample(x, idx=0, device=self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"])
        self.zero_optimizers()

        with autocast('cuda', enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)
            # Right before loss:
            #print('global_pred shape:', tuple(global_pred.shape))  # (T, N, C) after permute
            #print('global_pred dtype:', global_pred.dtype)
            #print('global_pred range:', global_pred.min().item(), global_pred.max().item())

            #print("targets dtype:", y.dtype)
            #print("targets min:", y.min().item())
            #print("targets max:", y.max().item())
            #print("Blank index:", 79)
        B, Lmax = y.shape
        # mask positions < true length
        y_len = torch.tensor(y_len, device=y.device)

        mask = torch.arange(Lmax, device=y.device).unsqueeze(0) < y_len.unsqueeze(1)
        y_flat = y[mask]                     # shape (sum(y_len),)
        y_flat = y_flat.to(torch.long)
        log_probs = global_pred  # (T, N, C) as you had
        x_reduced_len = torch.tensor(x_reduced_len, device=y.device)
        # Ensure dtypes are correct
        input_lengths  = x_reduced_len.to(torch.long)   # (N,)
        target_lengths = y_len.to(torch.long)           # (N,)
        #print(f"{y_flat.max().item()} vs {self.params['model_params']['vocab_size']}")
        #assert y_flat.max().item() < (self.params['model_params']["vocab_size"]+1), "Target out of range"
        #assert y_flat.min().item() >= 0, "Negative target index"
        #print(f"log_probs shape: {log_probs.shape[0]}, input_lengths: {input_lengths.max().item()}")
        #assert log_probs.shape[0] >= input_lengths.max().item()
        #assert y_flat.numel() == int(target_lengths.sum().item())
        #print("log_probs.shape (actually passed):", log_probs.shape)
        #print("input_lengths.max():", input_lengths.max().item())
        #print("target_lengths.max():", target_lengths.max().item())
        # (Optional but recommended) compute loss in fp32 for stability
        #log_probs = log_probs.permute(1, 2, 0) #if False else log_probs  # T, classes, batch 

        loss = loss_ctc(log_probs.float(), y_flat, input_lengths, target_lengths)        
        #loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)

        self.backward_loss(loss)

        self.step_optimizers()
        # original: pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        pred = torch.argmax(global_pred, dim=2) 
        pred = pred.T

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss_ctc": loss.item(),
            "str_x": self.pred_to_str(pred, x_reduced_len),
            "str_y": batch_data["raw_labels"]
        }

        return values
    '''
    def train_batch(self, batch_data, metric_names):
        """
        Forward and backward pass for training
        """
        x = batch_data["imgs"].to(self.device)
        # print(detect_image_format(x))
        y = batch_data["labels"].to(self.device)
        # print(detect_image_format(x))
        
        if False: # show images
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)

            # Move tensor to CPU and save grid or individual images
            # Example: save first 8 images as a grid
            vutils.save_image(
                x[:8].cpu(), 
                os.path.join(debug_dir, "batch_sample_grid.png"),
                nrow=4,  # 4 images per row
                normalize=True,  # scales pixel values to [0,1]
                value_range=(0, 1)
            )

            # Optional: save each image individually
            for i in range(min(8, x.size(0))):
                vutils.save_image(
                    x[i].cpu(),
                    os.path.join(debug_dir, f"img_{i}.png"),
                    normalize=True,
                    value_range=(0, 1)
            )
        


        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], zero_infinity=True)
        self.zero_optimizers()

        with autocast('cuda', enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)

            #with torch.no_grad():
                #img = sample_batch.to(device)
                #f_ssl = ssl_encoder(img)
                #f_base = base_encoder(img)

            #print("SSL mean/std/max:", f_ssl.mean().item(), f_ssl.std().item(), f_ssl.abs().max().item())
            #print("BASE mean/std/max:", x.mean().item(), x.std().item(), x.abs().max().item())

            global_pred = self.models["decoder"](x)


        #with torch.no_grad():
        #    print("encoder out shape:", tuple(x.shape), "dtype:", x.dtype)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)

        self.backward_loss(loss)

        self.step_optimizers()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        #predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
        #predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
        '''
            if any(99 in tokens for tokens in predicted_tokens):
                print("99 is in the list")
            else:
                print("99 is not in the list")
        '''
        str_x = self.pred_to_str(pred, x_reduced_len)
        if self.params["training_params"]["showGroundTruthAndPrediction"]:
            self.batchCount += 1
            if self.batchCount % 1000 == 0:
                #print(f"Batch {self.batchCount} - Loss: {sum_loss.item()} - Error rate: {error_rate if 'error_rate' in locals() else 'N/A'}")
                print(f"{batch_data["raw_labels"]} =>  {str_x}")

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss_ctc": loss.item(),
            "str_x": str_x,
            "str_y": batch_data["raw_labels"]
        }

        return values


    def evaluate_batch(self, batch_data, metric_names):
        """
        Forward pass only for validation and test
        """
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], zero_infinity=True)

        start_time = time.time()
        '''
        with torch.no_grad():
            out = self.models["encoder"](x)
            print("out finite:", torch.isfinite(out).all().item())
            print("out min/max:", out.min().item(), out.max().item())

            log_probs = torch.nn.functional.log_softmax(out, dim=1)
            print("log_probs finite:", torch.isfinite(log_probs).all().item())
            print("log_probs min/max:", log_probs.min().item(), log_probs.max().item())
        '''
        with autocast('cuda', enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)
        # expects T, B, C  receives  B  C  T


        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        str_x = self.pred_to_str(pred, x_reduced_len)

        process_time =time.time() - start_time

        if self.params["training_params"]["showGroundTruthAndPrediction"]:
            print(f"{batch_data['raw_labels']} =>  {str_x}")
        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss_ctc": loss.item(),
            "str_x": str_x,
            "str_y": batch_data["raw_labels"],
            "time": process_time
        }
        return values

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def pred_to_str(self, pred, pred_len):
        """
        convert prediction tokens to string
        """
        ind_x = [pred[i][:pred_len[i]] for i in range(pred.shape[0])]
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]
        return str_x
