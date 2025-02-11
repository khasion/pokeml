#!/usr/bin/env python3
import os
from PIL import Image

def remove_icc_profile(image_path, output_path):
    """
    Opens an image using Pillow, strips the embedded ICC profile by creating a new image
    with the same data, and saves it to output_path.
    """
    try:
        img = Image.open(image_path)
        data = list(img.getdata())
        new_img = Image.new(img.mode, img.size)
        new_img.putdata(data)
        new_img.save(output_path)
        print(f"Processed: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_directory(directory):
    """
    Recursively walks through the given directory and processes all image files.
    Only files ending with common image extensions (png, jpg, jpeg) are processed.
    The processed image is saved temporarily and then replaces the original file.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                tmp_output = image_path
                remove_icc_profile(image_path, tmp_output)
                # Replace the original file with the processed image
                os.replace(tmp_output, image_path)

def main():
    directories = ['pokemon-data', 'no-pokemon']
    for directory in directories:
        if os.path.isdir(directory):
            print(f"Processing directory: {directory}")
            process_directory(directory)
        else:
            print(f"Directory not found: {directory}")

if __name__ == '__main__':
    main()
