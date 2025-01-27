import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import argparse


def load_and_preprocess_images(input_folder, output_folder, target_size=(128, 128), grayscale=False):

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = []
    labels = []
    # Get subfolder names (class names)
    class_names = os.listdir(input_folder)

    for class_name in class_names:
        
        class_input_folder = os.path.join(input_folder, class_name)
        class_output_folder = os.path.join(output_folder, class_name)

        # Create output subfolder for each class
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)
        # Skip if it's not a folder
        if not os.path.isdir(class_input_folder):
            continue  

        # Initialize image counter for this class
        image_counter = 0

        for filename in os.listdir(class_input_folder):
            img_path = os.path.join(class_input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is not None:

                # Resize image
                img = cv2.resize(img, target_size)
                # Normalize pixel values to [0, 1]
                img = img / 255.0
                # Convert back to 8-bit format for saving
                img = (img * 255).astype(np.uint8)
                # Save the preprocessed image in PNG format
                output_filename = f"{class_name}_{image_counter}.png"
                output_path = os.path.join(class_output_folder, output_filename)
                cv2.imwrite(output_path, img)

                image_counter += 1
                
                # Append image and label for further use
                images.append(img)
                labels.append(class_name)
    
    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Save the preprocessed data and labels as NumPy files
    np.save(os.path.join(output_folder, 'preprocessed_images.npy'), images)
    np.save(os.path.join(output_folder, 'preprocessed_labels.npy'), labels)

    return images, labels, class_names

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Preprocess images from a folder and save them in a specified output folder.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing the input images.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where preprocessed images will be saved.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[128, 128], help="Target size for resizing images (height, width).")


    # Parse arguments
    args = parser.parse_args()
    
    # Call the preprocessing function
    images, labels, class_names = load_and_preprocess_images(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        target_size=tuple(args.target_size)
    )

    # Print summary
    print(f"Preprocessing complete!")
    print(f"Number of images processed: {len(images)}")
    print(f"Classes found: {class_names}")
    print(f"Preprocessed images saved in: {args.output_folder}")

if __name__ == "__main__":
    main()
