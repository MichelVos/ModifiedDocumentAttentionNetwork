import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter, defaultdict
import random
import numpy as np
import torch
import math
import cv2

class words(Dataset):
    def __init__(self, params,  imagesPath, metadata, min_length = None):
        self.load_in_memory = True
        self.name = params["name"]
        self.params = params
        self.preprocessings = params.get("preprocessor", [])
        self.mean = None
        self.std = None
        self.rgb = True
        self.samples = self.load_data(imagesPath, metadata, min_length)
        self.mean, self.std = self.compute_std_mean()
        self.apply_specific_treatment_after_dataset_loading(self)
        self.padding_value = params["config"]["padding_value"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        x = sample["img"]
        x = self.resize_pad_word(x, target_h=64, target_w=256)
        # save the image for debugging
        #cv2.imwrite(f"debug_{idx}.png", x)
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).float()     # (H, W, C)
        x = x.permute(2, 0, 1)         # (C, H, W)
        sample["img"] = x
        return sample

    def resize_pad_word(self, img, target_h=32, target_w=128):
        # image is now HWC
        h, w = img.shape[:2]

        # scale to target height
        scale = target_h / h
        new_w = int(w * scale)

        img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # pad width centered

        pad_w = max(0, target_w - new_w)
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        img = np.pad(
            img,
            ((0, 0), (pad_l, pad_r), (0, 0)) if img.ndim == 3 else ((0, 0), (pad_l, pad_r)),
            mode="constant",
            constant_values=255  # white background
        )

        # clip if too wide
        img = img[:, :target_w]

        return img

    def load_data(self, imagesPath, metadata, min_length):
        data = []
        words = self.read_words(metadata)
        for word_id, transcription in words:
            accept = True
            if not min_length is None:
                if len(transcription) < min_length:
                    accept = False

            if accept:
                image_path = self.get_image_path(imagesPath, word_id)
                # image = self.load_image(image_path)
                data.append({
                    "img": self.apply_preprocessing(self.load_image(image_path), self.preprocessings), 
                    "label": transcription
                    })
        return data

    def most_used_descriptions_old(self, data, n = 10):
        desc_counter = Counter(desc for _, desc in data)
        top_descriptions = {desc for desc, _ in desc_counter.most_common(n)}
        result = [(img, desc) for img, desc in data if desc in top_descriptions]
        return result

    def most_used_descriptions(self, data, n=10):
        desc_counter = Counter(item["label"] for item in data)

        top_descriptions = {
            desc for desc, _ in desc_counter.most_common(n)
        }

        return [
                item
                for item in data
                if item["label"] in top_descriptions
            ]
    
    def read_words(self, filepath):
        """
        Read words.txt file, ignoring lines starting with # and extracting
        only the first and last field from each data line.
        
        Args:
            filepath (str): Path to the words.txt file
            
        Returns:
            list of tuples: Each tuple contains (word_id, transcription)
        """
        words = []
        
        with open(filepath, 'r') as f:
            for line in f:
                # Strip whitespace
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Split by whitespace
                fields = line.split()

                
                if len(fields) >= 2 and fields[1] != 'err':
                    # First field is word ID, last field is transcription
                    word_id = fields[0]
                    transcription = fields[-1]
                    if len(transcription) > 1:
                        words.append((word_id, transcription))
        
        return words
    
    def get_image_path(self, base_path, word_id):
        elements = word_id.split('-')
        first_dir = elements[0]
        second_dir = elements[0] + '-' + elements[1]
        
        image_path = os.path.join(base_path, first_dir, second_dir, f"{word_id}.png")
        
        return image_path

    def compute_std_mean(self):
        """
        Compute cumulated variance and mean of whole dataset
        """
        print("Computing mean and std...")
        max_width = max_height = 0
        min_width = min_height = math.inf
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
            min_width = min(min_width, w)
            min_height = min(min_height, h)
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
        print(f"Min image size: {min_width}x{min_height}")
        return mean, std

    def cap_samples_per_label(self, data, labels_subset, n=500, seed=None):
        """
        data: list of dicts {"img": ..., "label": ...}
        labels_subset: iterable of labels to keep (e.g. top-N most frequent)
        n: maximum samples per label
        seed: optional, for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Group samples by label
        by_label = defaultdict(list)
        for item in data:
            label = item["label"]
            if label in labels_subset:
                by_label[label].append(item)

        # Cap each label to at most n samples
        result = []
        for label, items in by_label.items():
            if len(items) > n:
                result.extend(random.sample(items, n))
            else:
                result.extend(items)

        return result

    def load_image(self, path):
        with Image.open(path) as pil_img:
            img = np.array(pil_img)
            ## grayscale images
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
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
    

if __name__ == "__main__":
    imagesPath = "/home/michel/dev/python/raw/IAM/words"
    metadata = "/home/michel/dev/python/DAN/utils/TSNE/words.txt"
    dataset = words(None, imagesPath, metadata)

    for i in range(10):
        image, transcription = dataset[i]["img"], dataset[i]["label"]
        print(f"Item {i}: Transcription = {transcription}, Image Size = {image.size}")

    print(f"\nTotal items in dataset: {len(dataset)}")
    n_most_used = 10
    most_used = dataset.most_used_descriptions(dataset.samples, n=n_most_used)
    most_used = dataset.cap_samples_per_label(most_used, 
                                  labels_subset=set(item["label"] for item in most_used),
                                  n=500,
                                  seed=42)
    print(f"\nTotal items with most used descriptions: {len(most_used)}")
    
    desc_counter = Counter(item["label"] for item in most_used)
    for item in desc_counter.most_common(n_most_used):
        print(f"Description: {item[0]}, Count: {item[1]}")


