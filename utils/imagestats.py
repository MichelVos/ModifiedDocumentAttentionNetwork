"""
imagestats.py

Compute basic statistics on image dimensions within a dataset directory.
This utility scans images in a folder, reads their width and height, and reports
the average, minimum, and maximum dimensions across the collection. The script is
intended to help inspect dataset homogeneity and determine whether image
resizing or normalization strategies are appropriate for model training.

Typical use:
    - summarize dataset image sizes
    - check resolution consistency across folders
    - support preprocessing and model input design decisions

This script is a data-analysis utility rather than a core training or
evaluation component.
"""

import os
from PIL import Image   
from pathlib import Path

def calculate_average_image_dimensions(directory):
    total_width = 0
    total_height = 0
    max_width = 0
    max_height = 0
    min_width = float('inf')
    min_height = float('inf')
    count = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                total_width += width
                total_height += height
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height
                if width < min_width:
                    min_width = width
                if height < min_height:
                    min_height = height
                count += 1

    if count > 0:
        avg_width = total_width / count
        avg_height = total_height / count
        print(f"Average width: {avg_width}, Average height: {avg_height}")
    else:
        print("No images found in the directory.")
    print(f"Max width: {max_width}, Max height: {max_height}")
    print(f"Min width: {min_width}, Min height: {min_height}")

# Example usage:
calculate_average_image_dimensions(f'{Path.home()}/dev/python/formatted/IAM_non_syn_line/train')
