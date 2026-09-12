import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
sys.path.append(os.path.dirname(os.path.dirname(DOSSIER_PARENT)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(DOSSIER_PARENT))))
import pickle





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
#from basic.utils import pad_image
# expects your DocFolder that returns {"image": CxHxW, "path", "orig_hw"}
import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class DocFolder(Dataset):
    def __init__(self, params, preprocessings = [ {"type": "to_RGB",},], rgb=True, load_in_memory = True, size=(1232, 64)):
        self.img_path = os.path.join(params["line_dataset_params"]["path"], 'train' )
        self.pkl_path = os.path.join(params["line_dataset_params"]["path"], 'labels.pkl' )
        self.set_name = "train" 
        self.params = params
        self.size = size
        self.rgb = rgb
        self.mean = None
        self.std = None
        self.preprocessings = preprocessings
        self.load_in_memory = load_in_memory
        self.samples = self.load_samples(self.img_path, self.load_in_memory)
        self.mean, self.std = self.compute_std_mean()
        #self.apply_specific_treatment_after_dataset_loading(self)
        self.padding_value = 0 # params["config"]["padding_value"]

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
        samples = list()
        print(f"Loading images from {image_path}, load_in_memory={load_in_memory}")
        with open(self.pkl_path, "rb") as f:
            info = pickle.load(f)
        gt = info["ground_truth"]['train']
        for filename in natural_sort(gt.keys()):
            name = os.path.join(image_path, filename)
            full_path = os.path.join(image_path, filename)
            if isinstance(gt[filename], dict) and "text" in gt[filename]:
                label = gt[filename]["text"]
            else:
                label = gt[filename]
            img = self.load_image(full_path)
            if not "bottom" in gt[filename] or not "right" in gt[filename] or not "top" in gt[filename] or not "left" in gt[filename]:
                if "raw_resize" in self.params["config"] and self.params["config"]["raw_resize"]:
                    height, width = self.params["config"]["raw_resize"]
                    if img.shape[0] > height or img.shape[1] > width:
                        # I don't want to resize to height,width but scale the image so the larges dimension falls within height,width
                        ratio_h = height / img.shape[0]
                        ratio_w = width / img.shape[1]
                        ratio = min(ratio_h, ratio_w)
                        new_size = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))  # (width, height)
                        img = cv2.resize(img, new_size)
                samples.append({
                    "path": full_path,
                    "label": label,
                    "img": self.apply_preprocessing(img, self.preprocessings) if load_in_memory else None,  
                    "top": 0,
                    "bottom": img.shape[0],
                    "left": 0,
                    "right": img.shape[1],
                })
            else:
                samples.append({
                    "path": full_path,
                    "label": label,
                    "img": self.apply_preprocessing(img, self.preprocessings) if load_in_memory else None,
                    "top": 0,
                    "bottom": gt[filename]["bottom"] - gt[filename]["top"],
                    "left": 0,
                    "right": gt[filename]["right"] - gt[filename]["left"],
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
        ten = len(self.samples) // 10
        for i in range(len(self.samples)):
            if i % ten == 0:
                print(".", end = "")
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
        print("\rMean:", mean, "Std:", std)
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

        return {"image": x, "path": sample["path"], "label": sample["label"]}
   
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
                new_h = min(max_h, int(h * ratio_h))
                new_w = min(max_w, int(w * ratio_w))
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
        if resize_ratio != [1, 1] and "raw_line_seg_label" in sample:
            for li in range(len(sample["raw_line_seg_label"])):
                for side, ratio in zip((["bottom", "top"], ["right", "left"]), resize_ratio):
                    for s in side:
                        sample["raw_line_seg_label"][li][s] = sample["raw_line_seg_label"][li][s] * ratio

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




params = {
    "dataset_params": {
        #"dataset_level": dataset_level,
        #"dataset_manager": OCRDatasetManager,
        #"dataset_class": OCRDataset,
        #"datasets": {
        #    dataset_name: "../formatted/{}_{}{}".format(dataset_name, dataset_level, dataset_variant),
        #},
        #"train": {
        #    "name": "{}-train".format(dataset_name),
        #    "datasets": [(dataset_name, "train"), ],
        #},
        #"valid": {
        #    "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
        #},
        "config": {
            "save_image": None, # "debug_images_dan_IAM_page_real",  # set to None to disable image saving
            "load_in_memory": True,  # Load all images in CPU memory
            "worker_per_gpu": 4,  # Num of parallel processes per gpu for data loading
            "width_divisor": 8,  # Image width will be divided by 8
            "height_divisor": 32,  # Image height will be divided by 32
            "padding_value": 0,  # Image padding value
            "padding_token": None,  # Label padding value
            "charset_mode": "seq2seq",  # add end-of-transcription ans start-of-transcription tokens to charset
            "constraints": ["add_eot", "add_sot"],  # add end-of-transcription ans start-of-transcription tokens in labels
            "normalize": True,  # Normalize with mean and variance of training dataset
            "preprocessings": [
                {
                    "type": "to_RGB",
                    # if grayscaled image, produce RGB one (3 channels with same value) otherwise do nothing
                },
            ],
            #"augmentation": aug_config_real(0.9, 0.1),
            # "synthetic_data": None,
            "non_synthetic_data": {
                "init_proba": 0.9,  # begin proba to generate synthetic document
                "end_proba": 0.2,  # end proba to generate synthetic document
                "num_steps_proba": 200000,  # linearly decrease the percent of synthetic document from 90% to 20% through 200000 samples
                #"proba_scheduler_function": linear_scheduler,  # decrease proba rate linearly
                "start_scheduler_at_max_line": True,  # start decreasing proba only after curriculum reach max number of lines
                #"dataset_level": dataset_level,
                "curriculum": True,  # use curriculum learning (slowly increase number of lines per synthetic samples)
                "crop_curriculum": True,  # during curriculum learning, crop images under the last text line
                "curr_start": 0,  # start curriculum at iteration
                "curr_step": 10000,  # interval to increase the number of lines for curriculum learning
                "min_nb_lines": 1,  # initial number of lines for curriculum learning
                #"max_nb_lines": max_nb_lines[dataset_name],  # maximum number of lines for curriculum learning
                "padding_value": 255,                    
                "page": True, # Page and line behave a little different.
            },
            "synthetic_data": {
                "init_proba": 0.9,  # begin proba to generate synthetic document
                "end_proba": 0.2,  # end proba to generate synthetic document
                "num_steps_proba": 200000,  # linearly decrease the percent of synthetic document from 90% to 20% through 200000 samples
                #"proba_scheduler_function": linear_scheduler,  # decrease proba rate linearly
                "start_scheduler_at_max_line": True,  # start decreasing proba only after curriculum reach max number of lines
                #"dataset_level": dataset_level,
                "curriculum": True,  # use curriculum learning (slowly increase number of lines per synthetic samples)
                "crop_curriculum": True,  # during curriculum learning, crop images under the last text line
                "curr_start": 0,  # start curriculum at iteration
                "curr_step": 10000,  # interval to increase the number of lines for curriculum learning was: 10000
                "min_nb_lines": 1,  # initial number of lines for curriculum learning
                #"max_nb_lines": max_nb_lines[dataset_name],  # maximum number of lines for curriculum learning
                "padding_value": 255,
                # config for synthetic line generation
                "config": {
                    "background_color_default": (255, 255, 255),
                    "background_color_eps": 15,
                    "text_color_default": (0, 0, 0),
                    "text_color_eps": 15,
                    "font_size_min": 35,
                    "font_size_max": 45,
                    "color_mode": "RGB",
                    "padding_left_ratio_min": 0.00,
                    "padding_left_ratio_max": 0.05,
                    "padding_right_ratio_min": 0.02,
                    "padding_right_ratio_max": 0.2,
                    "padding_top_ratio_min": 0.02,
                    "padding_top_ratio_max": 0.1,
                    "padding_bottom_ratio_min": 0.02,
                    "padding_bottom_ratio_max": 0.1,
                    "not_synthetic": True,
                },
            }
        }
    },

    "line_dataset_params": {
        "path": "/home/michel/dev/python/formatted/IAM_non_syn_line",
    },

    "model_params": {
        "models": {
            #"encoder": FCN_Encoder,
            #"decoder": GlobalHTADecoder,
        },
        # "transfer_learning": None,
        "transfer_learning": {
            # model_name: [state_dict_name, checkpoint_path, learnable, strict]
            #"encoder": ["encoder", "../../line_OCR/ctc/outputs/FCN_read_2016_line_syn/checkpoints/best.pt", True, True],
            #"decoder": ["decoder", "../../line_OCR/ctc/outputs/FCN_read_2016_line_syn/checkpoints/best.pt", True, False],
            #"encoder": ["encoder", "outputs/FCN_read_2016_line_syn/checkpoints/best.pt", True, True],
            #"decoder": ["decoder", "outputs/FCN_read_2016_line_syn/checkpoints/best.pt", True, False],
            #"encoder": ["encoder", "outputs/FCN_IAM_line_syn/checkpoints/best.pt", True, True],
            #"decoder": ["decoder", "outputs/FCN_IAM_line_syn/checkpoints/best.pt", True, False],
            #"encoder": ["encoder", "outputs/FCN_RIMES_line_syn/checkpoints/best.pt", True, True],
            #"decoder": ["decoder", "outputs/FCN_RIMES_line_syn/checkpoints/best.pt", True, False],
            "encoder": ["encoder", "outputs/FCN_IAM_line200/checkpoints/best.pt", True, True],
            "decoder": ["decoder", "outputs/FCN_IAM_line200/checkpoints/best.pt", True, False],
            #"encoder": ["encoder", "outputs/FCN_IAM_line/checkpoints/best.pt", True, True],
            #"decoder": ["decoder", "outputs/FCN_IAM_line/checkpoints/best.pt", True, False],
            #"encoder": ["encoder", "outputs/FCN_RIMES_line_syn/checkpoints/best.pt", True, True],
            #"decoder": ["decoder", "outputs/FCN_RIMES_line_syn/checkpoints/best.pt", True, False],
        },
        "transfered_charset": True,  # Transfer learning of the decision layer based on charset of the line HTR model
        "additional_tokens": 1,  # for decision layer = [<eot>, ], only for transfered charset

        "input_channels": 3,  # number of channels of input image
        "dropout": 0.5,  # dropout rate for encoder
        "enc_dim": 256,  # dimension of extracted features
        "nb_layers": 5,  # encoder
        "h_max": 500,  # maximum height for encoder output (for 2D positional embedding)
        "w_max": 1000,  # maximum width for encoder output (for 2D positional embedding)
        "l_max": 15000,  # max predicted sequence (for 1D positional embedding)
        "dec_num_layers": 8,  # number of transformer decoder layers
        "dec_num_heads": 4,  # number of heads in transformer decoder layers
        "dec_res_dropout": 0.1,  # dropout in transformer decoder layers
        "dec_pred_dropout": 0.1,  # dropout rate before decision layer
        "dec_att_dropout": 0.1,  # dropout rate in multi head attention
        "dec_dim_feedforward": 256,  # number of dimension for feedforward layer in transformer decoder layers
        "use_2d_pe": True,  # use 2D positional embedding
        "use_1d_pe": True,  # use 1D positional embedding
        "use_lstm": False,
        "attention_win": 100,  # length of attention window
        # Curriculum dropout
        #"dropout_scheduler": {
        #    "function": exponential_dropout_scheduler,
        #    "T": 5e4,
        #}

    },

    "training_params": {
        "output_folder": "dan_IAM_page_real_20251201",  # folder name for checkpoint and results dan_IAM_page, dan_read_page, dan_rimes_page
        "max_nb_epochs": 350,  # maximum number of epochs before to stop
        "max_training_time": 8*3600, #3600 * 24 * 1.9,  # maximum time before to stop (in seconds)
        "load_epoch": "last",  # ["best", "last"]: last to continue training, best to evaluate
        "interval_save_weights": 1,  # None: keep best and last only
        "batch_size": 4,  # mini-batch size for training
        "valid_batch_size": 4,  # mini-batch size for validation
        "use_ddp": False,  # Use DistributedDataParallel
        "ddp_port": "20027",
        "use_amp": True,  # Enable automatic mix-precision
        #"nb_gpu": torch.cuda.device_count(),
        #"optimizers": {
        #    "all": {
        #        "class": Adam,
        #        "args": {
        #            "lr": 0.0001,
        #            "amsgrad": False,
        #        }
        ##    },
        #}#,
        "showGroundTruthAndPrediction": False,  # Print ground truth and prediction
        "log_values": True,  # Log values in log file
        "early_stop_evaluation": True,  # Early stop evaluation on validation set if cer > 0.1
        "lr_schedulers": None,  # Learning rate schedulers
        "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
        "eval_on_valid_interval": 5,  # Interval (in epochs) to evaluate during training
        "focus_metric": "cer",  # Metrics to focus on to determine best epoch
        "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
        #"set_name_focus_metric": "{}-valid".format(dataset_name),  # Which dataset to focus on to select best weights
        "train_metrics": ["loss_ce", "cer", "wer", "syn_max_lines"],  # Metrics name for training
        "eval_metrics": ["cer", "wer", "map_cer"],  # Metrics name for evaluation on validation set during training
        "force_cpu": False,  # True for debug purposes
        "max_char_prediction": 2000,  # max number of token prediction 3000, 2000:IAM
        # Keep teacher forcing rate to 20% during whole training
        "teacher_forcing_scheduler": {
            "min_error_rate": 0.2,
            "max_error_rate": 0.2,
            "total_num_steps": 5e4
        },
    },
}

if __name__ == "__main__":
    DocFolder(params)

