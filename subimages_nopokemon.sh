#!/bin/bash

# Set the target directory
BASE_DIR="no-pokemon"

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
  echo "Error: ImageMagick is not installed. Install it and try again."
  exit 1
fi

# Function to split images
split_image() {
  local image_path="$1"
  local image_dir=$(dirname "$image_path")
  local image_name=$(basename "$image_path" .png)

  # Create a temporary directory for output to avoid clutter
  temp_dir="${image_dir}/${image_name}_split"

  mkdir -p "$temp_dir"

  # Use ImageMagick's convert command to split the image
  convert "$image_path" -crop 128x128 +repage "${temp_dir}/${image_name}_%d.png"

  # Move the cropped files back to the original directory and clean up the temp folder
  mv "${temp_dir}"/* "$image_dir"
  rmdir "$temp_dir"
  echo "Split $image_path into 128x128 subimages in $image_dir"
}

# Export function for use with find
export -f split_image

# Find all .png images in the directory and apply the split_image function
find "$BASE_DIR" -type f -name "*.png" -exec bash -c 'split_image "$0"' {} \;

echo "All images have been processed."
