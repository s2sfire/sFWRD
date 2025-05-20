#!/bin/bash

# Define the source and destination directories
src_dir="../../../database_files/UFS_S2S/MPM"
dest_dir="../../../UFS_Downscaled/Statistical/12km/MPM"

# Use find to recursively search for .nc files in the source directory
find "$src_dir" -type f -name "t2*.nc" -o -name "vpd*.nc" -o -name "ws*.nc" | while read -r file; do
    # Get the relative path of the file within the source directory
    rel_path="${file#$src_dir/}"
    
    # Construct the destination file path
    dest_file="$dest_dir/$rel_path"
    
    # Ensure the destination directory for this file exists
    dest_file_dir=$(dirname "$dest_file")
    mkdir -p "$dest_file_dir"
    
    # Check if the output file already exists
    if [ ! -f "$dest_file" ]; then
        # The output file does not exist, so run the cdo command
        cdo remapbil,grid_files/UFS_target_grid_12km.txt "$file" "$dest_file"
        echo "Processed: $file -> $dest_file"
    else
        # The output file exists, so skip this file
        echo "Output file $dest_file already exists, skipping..."
    fi
done
