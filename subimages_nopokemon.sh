#!/bin/bash

# Directory containing the images
SOURCE_DIR="no-pokemon"

# Output directory for subimages
OUTPUT_DIR="subimages"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each .png file in the directory and its subdirectories
find "$SOURCE_DIR" -type f -name "*.png" | while read -r IMAGE_PATH; do
  # Get the base filename without extension and the directory
  BASENAME=$(basename "$IMAGE_PATH" .png)
  IMAGE_DIR=$(dirname "$IMAGE_PATH")
  
  # Create a subdirectory for this image's subimages in the output directory
  IMAGE_OUTPUT_DIR="$OUTPUT_DIR/$BASENAME"
  mkdir -p "$IMAGE_OUTPUT_DIR"

  echo "Processing: $IMAGE_PATH"
  
  # Use ImageMagick to crop the image into 128x128 subimages
  convert "$IMAGE_PATH" -crop 128x128 +repage "$IMAGE_OUTPUT_DIR/${BASENAME}_%d.png"
done

echo "All images have been split into 128x128 subimages and saved to $OUTPUT_DIR."
