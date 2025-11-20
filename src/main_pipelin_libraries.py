#!/usr/bin/env python3
"""Execution of image processing pipeline using libraries.

Script Name:    main_pipeline_libraries.py
Author:         David Buchner, Imperial College London
Created:        10/10/2025
Last Modified:  22/10/2025
Description:
   Script to execute the image segmentation pipeline.
   It makes use of the image processing library scikit-image
   The pipline consists of:
   1. Loading and preprocessing data
   2. Threshholding and watershed segmentation
   3. Segmentation labelling
   4. Visualization segmentation

Usage:
   python main_pipeline_libraries.py /path/to/data/

Requirements:
   - Python 3.x
   - Required libraries:
       * numpy
       * scipy
       * scikit-image

"""

# -----------------------------
# Load Python packages
import time

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.segmentation import watershed

from load_dataset import load_tif_sequence
from visualisation import plot_3d_orthogonal_planes

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
plot_3d_orthogonal_planes(image3d_cubecrop, snapshot_view=(30, 300))

# -----------------------------
# 4. Apply a thresholding
# Compute Otsu's threshold
thresh_value = filters.threshold_otsu(image3d_cubecrop)

# Apply threshold
binary_cubecrop = image3d_cubecrop > thresh_value  # returns a boolean array

# -----------------------------
# 5. Apply distance transform and watershed segmentation
# Record start time of segmentation procedure
start_time = time.time()  # Record start time
# Compute distance transform
distance_cubecrop = ndi.distance_transform_edt(binary_cubecrop)
# Find local maxima to use as markers
local_maxi = morphology.local_maxima(distance_cubecrop)
# Label markers
markers = measure.label(local_maxi)
# Apply watershed
labels_cubecrop = watershed(-distance_cubecrop, markers, mask=binary_cubecrop)
# Record end time of segmentation procedure
end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"To to execute segmentation: {elapsed_time:.6f} seconds")
unique_labels_cubecrop = np.unique(labels_cubecrop)
print("Number of regions (excluding background):", len(unique_labels_cubecrop) - 1)

# Create orthogonal slice visualization
plot_3d_orthogonal_planes(
    labels_cubecrop, cmap="nipy_spectral", snapshot_view=(30, 300)
)

# -----------------------------
