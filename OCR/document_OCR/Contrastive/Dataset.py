import math, random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import cv2
import io
from basic.utils import pad_image

# expects your DocFolder that returns {"image": CxHxW, "path", "orig_hw"}

class DocFolder(Dataset):
    def __init__(self, params, root, train, custom_name, size, preprocessings, rgb=True, load_in_memory = False):
        self.img_path = os.path.join(root, 'train' if train else 'valid')
        # self.paths = [str(p) for p in Path(self.img_path).rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}]
        self.set_name = "train" if train else "valid"
        self.params = params
        self.divisor = params["config"]["divisor"] if "divisor" in params["config"] else 1
        self.name = custom_name
        self.size = size
        self.rgb = rgb
        self.mean = None
        self.std = None
        self.preprocessings = preprocessings
        self.load_in_memory = load_in_memory
        self.samples = self.load_samples(self.img_path, self.load_in_memory)
        self.mean, self.std = self.compute_std_mean()
        self.apply_specific_treatment_after_dataset_loading(self)
        self.padding_value = params["config"]["padding_value"]
        

    def get_mean_std(self):
        return self.mean, self.std

    def load_image(self, path):
        with Image.open(path) as pil_img:
            img = np.array(pil_img)
            ## grayscale images
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
        return img


    def load_samples(self, image_path, load_in_memory=True):
        """
        Load images
        """
        print(f"Loading images from {image_path}, load_in_memory={load_in_memory}")
        samples = list()
        for p in Path(self.img_path).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                img = self.load_image(str(p))
                # divide into subimages if divisor > 1
                if self.divisor > 1:
                    h, w = img.shape[:2]
                    sub_w = w // self.divisor
                    for i in range(self.divisor):
                        x0 = i * sub_w
                        sub_image = img[:, x0:x0+sub_w]
                        samples.append({
                            "path": str(p),
                            "img": self.apply_preprocessing(sub_image, self.preprocessings) if load_in_memory else None
                        })
                else:
                    samples.append({
                        "path": str(p),
                        "img": self.apply_preprocessing(img, self.preprocessings) if load_in_memory else None
                    })
        print(f"Found {len(samples)} images")
        return samples


    def compute_std_mean(self):
        """
        Compute cumulated variance and mean of whole dataset
        """
        print("Computing mean and std...")
        max_width = max_height = 0
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
        if not self.load_in_memory:
            sample = self.samples[0].copy()
            img = self.apply_preprocessing(self.load_image(sample["path"]), self.preprocessings)
        else:
            img = self.samples[0]["img"]
        _, _, c = img.shape
        sum = np.zeros((c,))
        nb_pixels = 0

        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                img = self.apply_preprocessing(self.load_image(self.samples[i]["path"]), self.preprocessings)
            else:
                img = self.samples[i]["img"]
            h, w = img.shape[:2]
            max_width = max(max_width, w)
            max_height = max(max_height, h)
            sum += np.sum(img, axis=(0, 1))
            nb_pixels += np.prod(img.shape[:2])
        mean = sum / nb_pixels
        diff = np.zeros((c,))
        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                img = self.apply_preprocessing(self.load_image(self.samples[i]["path"]), self.preprocessings)
            else:
                img = self.samples[i]["img"]
            diff += [np.sum((img[:, :, k] - mean[k]) ** 2) for k in range(c)]
        std = np.sqrt(diff / nb_pixels)

        self.mean = mean
        self.std = std
        print("Mean:", mean, "Std:", std)
        print(f"Max image size: {max_width}x{max_height}")
        return mean, std

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, i):
        sample = self.samples[i]
        if self.load_in_memory:
            img = sample["img"]
            if img is None:
                img = self.load_image(sample["path"])
                img = self.apply_preprocessing(img, self.preprocessings)
        else:
            img = self.load_image(sample["path"])
            img = self.apply_preprocessing(img, self.preprocessings)
        
        h, w = img.shape[:2]
        fullcompatibility = self.params["config"].get("full_compatibility", False)
        if fullcompatibility:
        
        
            if "max_size" in self.params["config"] and self.params["config"]["max_size"]:
                max_ratio = max(img.shape[0] / self.params["config"]["max_size"]["max_height"], sample["img"].shape[1] / self.params["config"]["max_size"]["max_width"])
                if max_ratio > 1:
                    new_h, new_w = int(np.ceil(sample["img"].shape[0] / max_ratio)), int(np.ceil(sample["img"].shape[1] / max_ratio))
                    img = cv2.resize(img, (new_w, new_h))        
            x = (img - self.mean) / self.std
            img_position = [0, 0, w, h]  # x0, y0, w, h
            if "padding" in self.params["config"] and self.params["config"]["padding"]:
                if self.set_name == "train" or not self.params["config"]["padding"]["train_only"]:
                    min_pad = self.params["config"]["padding"]["min_pad"]
                    max_pad = self.params["config"]["padding"]["max_pad"]
                    pad_width = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None
                    pad_height = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None

                    x, img_position = pad_image(x, padding_value=self.padding_value,
                                            new_width=self.params["config"]["padding"]["min_width"],
                                            new_height=self.params["config"]["padding"]["min_height"],
                                            pad_width=pad_width,
                                            pad_height=pad_height,
                                            padding_mode=self.params["config"]["padding"]["mode"],
                                            return_position=True)
        else:
    
            if w > self.size[0]:
                img = cv2.resize(img, self.size)  # width, height
            else:
                # pad to the right
                pad_width = self.size[0] - w
                left = pad_width // 2
                right = pad_width - left
                img = np.pad(img, ((0,0),(left,right),(0, 0)), mode='constant', constant_values=255)
                img = cv2.resize(img, self.size)  # width, height
        #img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        x = (img - self.mean) / self.std

        return {"image": x, "path": sample["path"]}
   
    def convertToImage(self, tensor):
        """
        Convert a torch tensor to a PIL image
        """
        if self.rgb:
            arr = tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        else:
            arr = tensor.squeeze(0).cpu().numpy()        # (H, W)
        mean = np.array(self.mean).reshape(1, 1, 3)
        std  = np.array(self.std).reshape(1, 1, 3)

        arr = (arr * std + mean)
        arr = arr.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return img  

    def convertNumPyToImage(self, arr):
        """
        Convert a torch tensor to a PIL image
        """
        arr = (arr * self.std + self.mean)
        arr = arr.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return img  
    
    def apply_preprocessing(self, img, preprocessings):
        """
        Apply preprocessings on each sample
        """
        resize_ratio = [1, 1]
        # img = sample["img"]
        for preprocessing in preprocessings:

            if preprocessing["type"] == "dpi":
                ratio = preprocessing["target"] / preprocessing["source"]
                temp_img = img
                h, w, c = temp_img.shape
                temp_img = cv2.resize(temp_img, (int(np.ceil(w * ratio)), int(np.ceil(h * ratio))))
                if len(temp_img.shape) == 2:
                    temp_img = np.expand_dims(temp_img, axis=2)
                img = temp_img

                resize_ratio = [ratio, ratio]

            if preprocessing["type"] == "to_grayscaled":
                temp_img = img
                h, w, c = temp_img.shape
                if c == 3:
                    img = np.expand_dims(
                        0.2125 * temp_img[:, :, 0] + 0.7154 * temp_img[:, :, 1] + 0.0721 * temp_img[:, :, 2],
                        axis=2).astype(np.uint8)

            if preprocessing["type"] == "to_RGB":
                temp_img = img
                h, w, c = temp_img.shape
                if c == 1:
                    img = np.concatenate([temp_img, temp_img, temp_img], axis=2)

            if preprocessing["type"] == "resize":
                keep_ratio = preprocessing["keep_ratio"]
                max_h, max_w = preprocessing["max_height"], preprocessing["max_width"]
                temp_img = img
                h, w, c = temp_img.shape

                ratio_h = max_h / h if max_h else 1
                ratio_w = max_w / w if max_w else 1
                if keep_ratio:
                    ratio_h = ratio_w = min(ratio_w, ratio_h)
                #new_h = min(max_h, int(h * ratio_h))
                #new_w = min(max_w, int(w * ratio_w))
                new_h = int(h * ratio_h) if max_h is None else min(max_h, int(h * ratio_h))
                new_w = int(w * ratio_w) if max_w is None else min(max_w, int(w * ratio_w))                
                temp_img = cv2.resize(temp_img, (new_w, new_h))
                if len(temp_img.shape) == 2:
                    temp_img = np.expand_dims(temp_img, axis=2)

                img = temp_img
                resize_ratio = [ratio_h, ratio_w]

            if preprocessing["type"] == "resize_when_bigger":
                keep_ratio = preprocessing["keep_ratio"]
                max_h, max_w = preprocessing["max_height"], preprocessing["max_width"]

                temp_img = img
                h, w, c = temp_img.shape

                if (not max_h is None and h > max_h) or (not max_w is None and w > max_w):  
                    ratio_h = max_h / h if max_h else 1
                    ratio_w = max_w / w if max_w else 1
                    if keep_ratio:
                        ratio_h = ratio_w = min(ratio_w, ratio_h)
                    #new_h = min(max_h, int(h * ratio_h))
                    #new_w = min(max_w, int(w * ratio_w))
                    new_h = int(h * ratio_h) if max_h is None else min(max_h, int(h * ratio_h))
                    new_w = int(w * ratio_w) if max_w is None else min(max_w, int(w * ratio_w))                
                    temp_img = cv2.resize(temp_img, (new_w, new_h))
                    if len(temp_img.shape) == 2:
                        temp_img = np.expand_dims(temp_img, axis=2)

                    img = temp_img
                    resize_ratio = [ratio_h, ratio_w]


            if preprocessing["type"] == "fixed_height":
                new_h = preprocessing["height"]
                temp_img = img
                h, w, c = temp_img.shape
                ratio = new_h / h
                temp_img = cv2.resize(temp_img, (int(w*ratio), new_h))
                if len(temp_img.shape) == 2:
                    temp_img = np.expand_dims(temp_img, axis=2)
                img = temp_img
                resize_ratio = [ratio, ratio]
        '''
        if resize_ratio != [1, 1] and "raw_line_seg_label" in sample:
            for li in range(len(sample["raw_line_seg_label"])):
                for side, ratio in zip((["bottom", "top"], ["right", "left"]), resize_ratio):
                    for s in side:
                        sample["raw_line_seg_label"][li][s] = sample["raw_line_seg_label"][li][s] * ratio
        '''
        #sample["img"] = img
        #sample["resize_ratio"] = resize_ratio
        return img

    def apply_specific_treatment_after_dataset_loading(self, dataset):
        #dataset.charset = self.charset
        #dataset.tokens = self.tokens
        #dataset.convert_labels()
        if "READ_2016" in dataset.name and "augmentation" in dataset.params["config"] and dataset.params["config"]["augmentation"]:
            dataset.params["config"]["augmentation"]["fill_value"] = tuple([int(i) for i in dataset.mean])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_height"] == "max":
            #dataset.params["config"]["padding"]["min_height"] = max([s["img"].shape[0] for s in self.train_dataset.samples])
            dataset.params["config"]["padding"]["min_height"] = max([s["img"].shape[0] for s in dataset.samples])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_width"] == "max":
            #dataset.params["config"]["padding"]["min_width"] = max([s["img"].shape[1] for s in self.train_dataset.samples])
            dataset.params["config"]["padding"]["min_width"] = max([s["img"].shape[1] for s in dataset.samples])

class TwoViewWrapper(torch.utils.data.Dataset):
    def __init__(self, base, augA=None, augB=None):
        self.base = base
        self.augA = augA or default_doc_augment
        self.augB = augB or default_doc_augment
        self.mean, self.std = self.base.get_mean_std()
        print("TwoViewWrapper mean:", self.mean, "std:", self.std)
    def __len__(self): return len(self.base)
    def get_mean_std(self):
        return self.mean, self.std
    def __getitem__(self, i):
        base_item = self.base[i]
        x = base_item["image"]

        path = base_item["path"]
        v1 = self.augA(x, self.mean, self.std, i, 1)
        v2 = self.augB(x, self.mean, self.std, i, 2)
        
        if False:
            img0 = self.base.convertNumPyToImage(x)
            img1 = self.base.convertNumPyToImage(v1)
            img2 = self.base.convertNumPyToImage(v2)
            img0.save(f"augments/view0_{i}.png")
            img1.save(f"augments/view1_{i}.png")
            img2.save(f"augments/view2_{i}.png")
        
        return {"view1": v1, "view2": v2, "path": path}
#imgcount = 0

def random_affine(img, degrees, translate, scale, shear):
    """
    img: np.ndarray of shape (H, W, C), values in [0,1]
    degrees: tuple (min, max)
    translate: tuple (max_dx, max_dy) as fraction of image size
    scale: tuple (min, max)
    shear: tuple (min, max)
    returns: np.ndarray of same shape
    """ 
    H, W, C = img.shape
    angle = random.uniform(degrees[0], degrees[1])
    max_dx = translate[0] * W
    max_dy = translate[1] * H
    tx = random.uniform(-max_dx, max_dx)
    ty = random.uniform(-max_dy, max_dy)
    sc = random.uniform(scale[0], scale[1])
    sh = random.uniform(shear[0], shear[1])

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, sc)
    M[0, 2] += tx
    M[1, 2] += ty

    # Shear
    M_shear = np.array([[1, math.tan(math.radians(sh)), 0],
                        [0, 1, 0]], dtype=np.float32)
    M = M_shear @ np.vstack([M, [0, 0, 1]])
    M = M[0:2, :]
    img_out = cv2.warpAffine(
        img,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return img_out


def thin_random_erasing(img, max_height_frac=0.2, max_width_frac=0.05):
    """
    img: np.ndarray of shape (H, W, C), values in [0,1]
    max_height_frac: maximum height of erased rectangle as fraction of image height
    max_width_frac: maximum width of erased rectangle as fraction of image width
    returns: np.ndarray of same shape
    """
    H, W, C = img.shape
    h = random.randint(1, int(H * max_height_frac))
    w = random.randint(1, int(W * max_width_frac))
    x0 = random.randint(0, W - w)
    y0 = random.randint(0, H - h)
    img[y0:y0 + h, x0:x0 + w, :] = 0.0
    return img

def aug_lines(x: np.ndarray, tmean, std, i, j) -> np.ndarray:
    """
    x: np.ndarray of shape (H, W, C), normalized using (x*std+tmean)/255
    tmean, std: normalization stats
    i, j: ids for debugging (kept for compatibility)

    Returns: np.ndarray (normalized again).
    """

    # --- Unnormalize to [0,1] ---
    x_raw = (x * std + tmean) / 255.0 # [0,1] range

    x_raw = np.clip(x_raw, 0.0, 1.0) # just in case

    H, W, C = x_raw.shape


    # brightness/contrast
    if random.random() < 0.8:
        b = random.uniform(0.9, 1.1)
        c = random.uniform(0.9, 1.1)
        mean = np.mean(x_raw, axis=(0,1), keepdims=True)
        x_raw = (x_raw - mean) * c + mean
        x_raw = np.clip(x_raw * b, 0.0, 1.0)

    # small gaussian noise
    if random.random() < 0.5:
        sigma = random.uniform(0.1, 0.5)   # safe range for lines
        x_raw = cv2.GaussianBlur(x_raw, (3, 3), sigmaX=sigma)
    # very small random erasing along stroke direction
    '''
    if random.random() < 0.3:
        x_raw = thin_random_erasing(x_raw, max_height_frac=0.2, max_width_frac=0.05)
    '''


    if random.random() < 0.8:
        x_raw = random_affine(
                x_raw,
                degrees=(-3, 3),      # tiny rotation
                translate=(0.02, 0.02),
                scale=(0.9, 1.1),
                shear=(-3, 3)
                )

    '''
    # ----------------------
    # 1. Mild brightness/contrast jitter
    # ----------------------
    if random.random() < 0.8:
        b = random.uniform(0.9, 1.1)
        c = random.uniform(0.9, 1.1)
        mean = np.mean(x_raw, axis=(0,1), keepdims=True)
        x_raw = (x_raw - mean) * c + mean
        x_raw = np.clip(x_raw * b, 0.0, 1.0)

    # ----------------------
    # 2. Mild gaussian blur
    # ----------------------
    if random.random() < 0.3:
        sigma = random.uniform(0.1, 0.5)   # safe range for lines
        x_raw = cv2.GaussianBlur(x_raw, (3, 3), sigmaX=sigma)

    # ----------------------
    # 3. Mild gaussian noise
    # ----------------------
    if random.random() < 0.2:
        noise = np.random.normal(0, 0.01, x_raw.shape).astype(np.float32)
        x_raw = np.clip(x_raw + noise, 0.0, 1.0)

    # ----------------------
    # 4. Mild JPEG compression
    # ----------------------
    if random.random() < 0.2:
        q = random.randint(70, 95)
        img_uint8 = (x_raw * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8.squeeze())
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q)
        x_raw = np.array(Image.open(io.BytesIO(buf.getvalue()))).astype(np.float32) / 255.0
        if x_raw.ndim == 2:
            x_raw = np.expand_dims(x_raw, axis=2)

    # ----------------------
    # 5. Small rotation (safe)
    # ----------------------
    angle = random.uniform(-2.0, 2.0)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    x_raw = cv2.warpAffine(
        x_raw,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if x_raw.ndim == 2:
        x_raw = np.expand_dims(x_raw, axis=2)

    if random.random() < 0.5:
        sx = random.uniform(0.8, 1.2)
        M = np.array([[sx, 0, 0],
                    [0, 1.0, 0]], dtype=np.float32)
        x_raw = cv2.warpAffine(
            x_raw, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if random.random() < 0.3:
        sy = random.uniform(0.9, 1.1)
        M = np.array([[1.0, 0, 0],
                    [0, sy, 0]], dtype=np.float32)

    if random.random() < 0.5:
        mask_w = random.randint(W // 20, W // 8)
        x0 = random.randint(0, W - mask_w)
        x_raw[:, x0:x0 + mask_w, :] = 0.0

    '''
    # ----------------------
    # Renormalize and return
    # ----------------------
    x_out = ((x_raw * 255.0) - tmean) / std # Back to normalized
    return x_out.astype(np.float32)

import numpy as np

def brightness_contrast(x, b=(0.9,1.1), c=(0.9,1.1)):
    """
    x: float image in [0,1], shape (H,W,C)
    """
    b_val = np.random.uniform(*b)
    c_val = np.random.uniform(*c)

    mean = np.mean(x, axis=(0,1), keepdims=True)
    x = (x - mean) * c_val + mean     # contrast
    x = x * b_val                    # brightness
    return np.clip(x, 0.0, 1.0)

import cv2
import random

def gaussian_blur(x, k=3, sigma=(0.1,0.5)):
    """
    x: float image in [0,1]
    """
    s = random.uniform(*sigma)
    x_blur = cv2.GaussianBlur(x, (k, k), sigmaX=s)
    return np.clip(x_blur, 0.0, 1.0)

def gaussian_noise(x, std=(0.0,0.02)):
    """
    x: float image in [0,1]
    """
    s = np.random.uniform(*std)
    noise = np.random.normal(0.0, s, x.shape)
    x_noisy = x + noise
    return np.clip(x_noisy, 0.0, 1.0)



def mild_perspective(x, max_warp=0.02):
    """
    x: float image in [0,1]
    max_warp: fraction of width/height for corner displacement
    """
    H, W = x.shape[:2]

    dx = max_warp * W
    dy = max_warp * H

    # original corners
    src = np.float32([
        [0, 0],
        [W-1, 0],
        [W-1, H-1],
        [0, H-1]
    ])

    # random, *small* displacement
    dst = src.copy()
    for i in range(4):
        dst[i,0] += random.uniform(-dx, dx)
        dst[i,1] += random.uniform(-dy, dy)

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        x, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return np.clip(warped, 0.0, 1.0)

def aug_view_a_org(x: np.ndarray, tmean, std, i, j):
    x = (x * std + tmean) / 255.0 # [0,1] range

    x = np.clip(x, 0.0, 1.0) # just in case


    # Photometric
    x = brightness_contrast(x, b=(0.6,1.4), c=(0.6,1.4))
    x = gaussian_blur(x, k=3, sigma=(0.1,1))
    x = gaussian_noise(x, std=(0.01, 0.05))

    # Very small geometry
    x = random_affine(
        x,
        degrees=(-3,3),
        translate=(0.02,0.03),
        scale=(0.9,1.1),
        shear=(-3,3)
    )
    x_out = ((x * 255.0) - tmean) / std # Back to normalized
    return x_out.astype(np.float32)
    

def aug_view_a(x: np.ndarray, tmean, std, i, j):
    # ---- back to [0,1] ----
    x = (x * std + tmean) / 255.0
    x = np.clip(x, 0.0, 1.0)

    H, W, C = x.shape

    # ---- NEW: horizontal crop + resize jitter ----
    if np.random.rand() < 0.8:
        crop_ratio = np.random.uniform(0.7, 1.0)  # keep 70–100%
        new_W = int(W * crop_ratio)

        if new_W < W:
            start = np.random.randint(0, W - new_W + 1)
            x = x[:, start:start + new_W, :]

            # resize back to original width
            x = cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)

    # ---- Photometric (slightly tuned) ----
    x = brightness_contrast(x, b=(0.7, 1.3), c=(0.7, 1.3))
    x = gaussian_blur(x, k=3, sigma=(0.1, 1.2))
    x = gaussian_noise(x, std=(0.01, 0.06))

    # ---- NEW: slight gamma variation (very effective, safe) ----
    if np.random.rand() < 0.5:
        gamma = np.random.uniform(0.8, 1.2)
        x = np.power(x, gamma)

    # ---- Geometry (keep small, but slightly less symmetric) ----
    x = random_affine(
        x,
        degrees=(-3, 3),
        translate=(0.02, 0.04),
        scale=(0.9, 1.1),
        shear=(-3, 3)
    )

    # ---- back to normalized ----
    x_out = ((x * 255.0) - tmean) / std
    return x_out.astype(np.float32)

def aug_view_b(x: np.ndarray, tmean, std, i, j):
    x = (x * std + tmean) / 255.0 # [0,1] range

    x = np.clip(x, 0.0, 1.0) # just in case

    # Photometric (stronger)
    x = brightness_contrast(x, b=(0.85,1.15), c=(0.85,1.15))
    x = gaussian_blur(x, k=3, sigma=(0.2,0.6))

    # Geometry (stronger but safe)
    x = random_affine(
        x,
        degrees=(-4,4),
        translate=(0.03,0.03),
        scale=(0.9,1.1),
        shear=(-4,4)
    )

    # Optional mild perspective
    if random.random() < 0.3:
        x = mild_perspective(x, max_warp=0.02)

    x_out = ((x * 255.0) - tmean) / std # Back to normalized
    return x_out.astype(np.float32)


def aug_lines_weak(x: np.ndarray, tmean, std, i, j) -> np.ndarray:
    """
    x: np.ndarray of shape (H, W, C), normalized using (x*std+tmean)/255
    tmean, std: normalization stats
    i, j: ids for debugging (kept for compatibility)

    Returns: np.ndarray (normalized again).
    """

    # --- Unnormalize to [0,1] ---
    x_raw = (x * std + tmean) / 255.0 # [0,1] range

    x_raw = np.clip(x_raw, 0.0, 1.0) # just in case

    H, W, C = x_raw.shape


    # 3) brightness/contrast
    if random.random() < 0.8:
        # x_raw = random_brightness_contrast(x_raw, brightness=0.2, contrast=0.2)
        b = random.uniform(0.9, 1.1)
        c = random.uniform(0.9, 1.1)
        mean = np.mean(x_raw, axis=(0,1), keepdims=True)
        x_raw = (x_raw - mean) * c + mean
        x_raw = np.clip(x_raw * b, 0.0, 1.0)

    if random.random() < 0.8:
        x_raw = random_affine(
                x_raw,
                degrees=(-1, 1),      # tiny rotation
                translate=(0.01, 0.01),
                scale=(0.95, 1.05),
                shear=(-2, 2)
                )


    x_out = ((x_raw * 255.0) - tmean) / std # Back to normalized
    return x_out.astype(np.float32)


def default_doc_augment(x: np.ndarray, tmean, std, i, j) -> np.ndarray:
    """
    x: np.ndarray of shape (H, W, C), values in [0,1]
    returns: np.ndarray of same shape
    """
    #global imgcount
    # x = (img - self.mean) / self.std
    proc = ''
    x_raw = (x * std + tmean) / 255 # unnormalize to [0,1]
    write_image(i, x_raw, "initial", j)

    H, W, C = x_raw.shape
    #arr = (x_raw * 255).astype(np.uint8)  # scale to [0,255] and cast to uint8
    #img = Image.fromarray(arr)
    #img.save(f"{imgcount}_original.png")
    # Random resized crop
    
    scale = (0.5, 1.0)
    ratio = (0.9, 1.1)
    for _ in range(10):
        area = H * W
        target_area = random.uniform(*scale) * area
        aspect = math.sqrt(random.uniform(*ratio))
        h = int(round(math.sqrt(target_area / aspect)))
        w = int(round(math.sqrt(target_area * aspect)))
        if 1 <= h <= H and 1 <= w <= W:
            i = random.randint(0, H - h)
            j = random.randint(0, W - w)
            x_raw = x_raw[i:i+h, j:j+w, :]
            break
    
    write_image(i, x_raw, "After first loop", j)
    
    #print("x.min:", x_raw.min(), "x.max:", x_raw.max())
    #print("x.mean:", x_raw.mean())
    #print("per-channel mean:", x_raw.mean(axis=(0,1)))

    # Brightness / contrast jitter
    if random.random() < 0.8 and True:
        proc = proc + 'bc '
        b = random.uniform(0.8, 1.2)  # brightness
        c = random.uniform(0.8, 1.2)  # contrast
        mean = np.mean(x_raw, axis=(0,1), keepdims=True)
        #print("per-channel mean:", mean)
        x_raw = (x_raw - mean) * c + mean
        x_raw = np.clip(x_raw * b, 0, 1)
        write_image(i, x_raw, "brightness contrast", j)

    # Gaussian blur
    if random.random() < 0.5 and True:
        proc = proc + 'blur '
        sigma = random.uniform(0.1, 1.5)
        ksize = 3
        x_raw = cv2.GaussianBlur(x_raw, (ksize, ksize), sigmaX=sigma)
        write_image(i, x_raw, "blur", j)

    # Light JPEG noise (simulated)
    if random.random() < 0.3 and True:
        proc = proc + 'jpeg '
        q = random.randint(40, 90)
        x_uint8 = (x_raw * 255).astype(np.uint8)
        pil_img = Image.fromarray(x_uint8.squeeze() if x_uint8.shape[2]==1 else x_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q)
        x_raw = np.array(Image.open(io.BytesIO(buf.getvalue())), dtype=np.uint8).astype(np.float32) / 255.0
        if x_raw.ndim == 2:  # grayscale JPEG reload
            x_raw = np.expand_dims(x, axis=2)
        write_image(i, x_raw, "jpeg", j)

    if random.random() < 0.5 and True:
        proc = proc + 'noise '
        mean = 0.0
        stddev = 0.05  # noise strength (5% of range)
        gauss = np.random.normal(mean, stddev, x_raw.shape).astype(np.float32)
        noisy = x_raw + gauss
        x_raw = np.clip(noisy, 0.0, 1.0)  # keep in [0,1]
        write_image(i, x_raw, "noise", j)


    if random.random() < 0.5 and True:
        proc = proc + 'saltpepper '
        s_vs_p = 0.5
        amount = 0.01  # fraction of pixels
        noisy = x_raw.copy()

        # Salt (set some pixels to white = 1.0)
        num_salt = np.ceil(amount * x_raw.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in x_raw.shape])
        noisy[coords] = 1.0

        # Pepper (set some pixels to black = 0.0)
        num_pepper = np.ceil(amount * x_raw.size * (1. - s_vs_p))
        coords = tuple([np.random.randint(0, i - 1, int(num_pepper)) for i in x_raw.shape])
        noisy[coords] = 0.0

        x_raw = noisy
        write_image(i, x_raw, "saltpepper", j)

    # Tiny rotation (±3°), bilinear, fill=0
    if True:
        proc = proc + 'rot '
        angle = random.uniform(-3.0, 3.0)
        M = cv2.getRotationMatrix2D((x.shape[1] / 2, x.shape[0] / 2), angle, 1.0)
        x_raw = cv2.warpAffine(x_raw, M, (x.shape[1], x.shape[0]),
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        write_image(i, x_raw, "rot", j)


    #arr = (x_raw * 255).astype(np.uint8)  # scale to [0,255] and cast to uint8
    #img = Image.fromarray(arr)
    #img.save(f"{imgcount}_augmented.png")
    #with open(f"{i}_proc.txt", "w", encoding="utf-8") as f:
    #    f.write(proc)
    #imgcount += 1
    x1 = ((x_raw * 255) - tmean) / std # normalize again
    return x1.astype(np.float32)

def write_image(index, img, remark, j):
    '''
    arr = (img * 255).astype(np.uint8)  # scale to [0,255] and cast to uint8
    img = Image.fromarray(arr)
    img.save(f"augments/{index}_{j}_{remark}.png")
    '''
    return
