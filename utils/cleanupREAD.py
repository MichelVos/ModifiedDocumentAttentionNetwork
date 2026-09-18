"""
cleanupREAD.py

Preprocess and enhance handwritten document images for READ dataset cleaning.
This utility reads images from a folder, applies contrast and background
normalization steps to improve the visibility of text strokes, and saves the
enhanced versions to a target directory. The script is intended for dataset
cleanup and preparation prior to model training or evaluation.

The preprocessing pipeline includes:
    - grayscale conversion
    - background estimation and removal
    - contrast normalization
    - local enhancement using CLAHE-like operations
    - optional noise reduction

Typical use:
    - clean a raw dataset folder
    - improve text readability for OCR or document recognition pipelines
    - generate enhanced images for downstream training

This script is a data-preprocessing utility rather than a core training module.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def enhance_soft(path):
    img = cv2.imread(path)

    # --- grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- remove background (illumination) ---
    background = cv2.GaussianBlur(gray, (71, 71), 0)
    norm = cv2.divide(gray, background, scale=255)

    # --- CLAHE (local contrast enhancement) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    return enhanced

def enhance_image(path, save_path=None, show=False):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # --- 1. Convert to grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 2. Estimate background (large blur) ---
    # increase ksize if background is very uneven
    background = cv2.medianBlur(gray, 51)

    # --- 3. Normalize (remove background) ---
    # makes text brighter relative to background
    norm = cv2.subtract(background, gray)

    # --- 4. Contrast stretching ---
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    # invert gray image
    norm = cv2.bitwise_not(norm)
    # --- 5. Adaptive threshold ---
    '''
    bin_img = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )
    ''' 
    # --- 6. Invert: text black, background white ---
    #bin_img = cv2.bitwise_not(bin_img)

    # --- 7. Optional: remove small noise ---
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    # --- Save / show ---
    if save_path:
        cv2.imwrite(save_path, norm)

    if show:
        cv2.imshow("result", norm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return norm


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            in_path = os.path.join(input_dir, fname)
            name, _ = os.path.splitext(fname)
            out_path = os.path.join(output_dir, name + "_enhanced.png")

            enhance_image(in_path, out_path, show=False)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    # --- Example usage ---
    # single image:
    # enhance_image("input.jpg", "output.png", show=True)

    # folder:
    #process_folder(f"{Path.home()}/dev/python/formatted/READ_2016_non_syn_line_cleaned2/train", f"{Path.home()}/dev/python/formatted/READ_2016_non_syn_line_cleaned3/train")
    process_folder(f"{Path.home()}/dev/python/formatted/READ_2016_page/test", f"{Path.home()}/dev/python/formatted/READ_2016_page_cleaned3/test")