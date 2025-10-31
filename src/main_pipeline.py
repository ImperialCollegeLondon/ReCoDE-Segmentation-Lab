#!/usr/bin/env python3
"""Execution of image processing pipeline.

main_pipeline.py
Author:         David Buchner, Imperial College London
Created:        08/10/2025
Last Modified:  29/10/2025
Description:
   Script to execute the image segmentation pipeline.
   The pipline consists of:
   1. Loading and preprocessing data
   2. Threshholding and watershed segmentation
   3. Segmentation labelling
   4. Visualization segmentation

Usage:
   python main_pipeline.py /path/to/data/

Requirements:
   - Python 3.x
   - Required libraries:
       * numpy

"""

# -----------------------------
# Load Python packages
import numpy as np

from image_processing import (
    chamfer_distance_3d_optimized,
    chamfer_distance_3d_structured,
    otsu_threshold,
)
from load_dataset import load_tif_sequence

# -----------------------------
# 1. Load the 3D image/dataset onto a numpy array
# i.e. into a format that allows efficient manipulation in the later steps
image3d = load_tif_sequence("../imaging_data/spherical_particles", start=100, end=199)

# -----------------------------
# 2. Crop the image to a cubical section.
# Get the shape of the original 3D image (e.g., (depth, height, width))
size_of_original_image3d = np.shape(image3d)

# Determine the size of the cube: use the smallest dimension of the original image
size_cubecrop = np.min(size_of_original_image3d)

# Crop the 3D image to a centered cube
image3d_cubecrop = image3d[
    tuple(
        slice((s - size_cubecrop) // 2, (s - size_cubecrop) // 2 + size_cubecrop)
        for s in image3d.shape
    )
]

# Print a statement showing the crop
print(
    f"3D image cropped from size {size_of_original_image3d} to {image3d_cubecrop.shape}"
)

# Create orthogonal slice visualization
# plot_3d_orthogonal_planes(image3d_cubecrop, snapshot_view=(30, 300))

# -----------------------------
# 3. Calculate Otsu threshold and binary image
# Compute Otsu's threshold
thresh_value = otsu_threshold(image3d_cubecrop)

# Apply threshold
binary_cubecrop = image3d_cubecrop > thresh_value  # returns a boolean array

# -----------------------------
# 4. Apply distance transform and watershed segmentation
# Compute distance transform
#distance_cubecrop = chamfer_distance_3d(binary_cubecrop)
distance_cubecrop_structured = chamfer_distance_3d_structured(binary_cubecrop)
#distance_cubecrop_argwhere = chamfer_distance_3d_argwhere(binary_cubecrop)
distance_cubecrop_optimized = chamfer_distance_3d_optimized(binary_cubecrop)

# Create orthogonal slice visualization
#plot_3d_orthogonal_planes(distance_cubecrop_optimized, snapshot_view=(30, 300))

