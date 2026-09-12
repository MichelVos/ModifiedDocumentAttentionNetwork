'''
I want a program that receives a directory that contains images and scans all images and calculates the average width and height of the images in that directory. The program should print out the average width and height once it has processed all images.import os
from PIL import Image.
'''
import os
from PIL import Image   

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
calculate_average_image_dimensions('/home/michel/dev/python/formatted/IAM_non_syn_line/train')   
