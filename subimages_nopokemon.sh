#!/bin/bash

# Set the target directory (update if needed)
BASE_DIR="no-pokemon"

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
  echo "Error: ImageMagick is not installed. Install it and try again."
  exit 1
fi

# Function to split images into 256x256 full tiles only
split_image() {
  local image_path="$1"
  local image_dir
  image_dir=$(dirname "$image_path")
  local image_name
  image_name=$(basename "$image_path" .png)

  # Create a temporary directory for the output tiles
  local temp_dir="${image_dir}/${image_name}_split"
  mkdir -p "$temp_dir"

  # Get the original image dimensions (width and height)
  read orig_width orig_height < <(identify -format "%w %h" "$image_path")

  # Set the desired tile size
  local tile_size=256

  # Calculate the number of full tiles in each dimension (using floor division)
  local full_tiles_x=$(( orig_width / tile_size ))
  local full_tiles_y=$(( orig_height / tile_size ))

  # If no full tile can be produced, skip the image
  if [ "$full_tiles_x" -eq 0 ] || [ "$full_tiles_y" -eq 0 ]; then
    echo "Image $image_path is smaller than ${tile_size}x${tile_size}, skipping."
    rm -r "$temp_dir"
    return
  fi

  local tile_index=0

  # Loop over each full tile's row and column
  for (( row=0; row<full_tiles_y; row++ )); do
    for (( col=0; col<full_tiles_x; col++ )); do
      local offset_x=$(( col * tile_size ))
      local offset_y=$(( row * tile_size ))
      convert "$image_path" -crop "${tile_size}x${tile_size}+${offset_x}+${offset_y}" +repage "${temp_dir}/${image_name}_${tile_index}.png"
      tile_index=$(( tile_index + 1 ))
    done
  done

  # Move the cropped tiles back to the original directory and remove the temporary folder
  mv "${temp_dir}"/* "$image_dir"
  rmdir "$temp_dir"
  echo "Split $image_path into ${tile_index} ${tile_size}x${tile_size} subimages in $image_dir"
}

# Export the function for use with find
export -f split_image

# Find all .png images in the target directory and process them with the split_image function
find "$BASE_DIR" -type f -name "*.png" -exec bash -c 'split_image "$0"' {} \;

echo "All images have been processed."
