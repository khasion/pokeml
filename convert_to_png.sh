#!/bin/bash

# Function to convert non-PNG images to PNG
do_convert() {
  local folder="$1"

  # Loop through all image files in the folder
  for image in "$folder"/*.{jpg,jpeg,JPG,JPEG,svg,bmp}; do
    # Check if the file exists (in case no matches)
    if [ -e "$image" ]; then
      # Extract the base name without the extension
      base_name="$(basename "$image" | sed 's/\.[^.]*$//')"

      # Convert to PNG format
      convert "$image" "$folder/$base_name.png"

      # Optionally, remove the original file after conversion
      rm "$image"

      echo "Converted $image to $folder/$base_name.png"
    fi
  done
}

# Define the folders to process
folders=(
  "kaggle-pokemon/*/"
  "huggingface-pokemon/*/"
)

# Process each folder
for folder_pattern in "${folders[@]}"; do
  for folder in $folder_pattern; do
    # Check if it's a valid directory
    if [ -d "$folder" ]; then
      echo "Processing folder: $folder"
      do_convert "$folder"
    fi
  done

done

echo "Image conversion complete."