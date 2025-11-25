#!/usr/bin/env python3
"""Function to determine local extrema.

Script Name:    local_extrema.py
Created:        25/11/2025
Description:

Requirements:
- Python 3.14
- Required libraries:
* numpy
"""

# -----------------------------
# Load Python packages
import numpy as np

# -----------------------------


def find_local_minima(img):
    """Determine the local minima of an image.

    This function determines the local minima of an image.
    It takes all 26 neighbours into consideration and obtains
    absolute local minima.

    Parameters:
         img (np.ndarray): 3D array, ideally the
         distance transform of an image

    Returns:
        np.ndarray: number of local minimax3

    """
    # Step 1: Pad the array with +inf to handle boundary conditions safely
    padded = np.pad(img, pad_width=1, mode="constant", constant_values=np.inf)

    # Step 2: Extract the center region (same shape as original array)
    center = padded[1:-1, 1:-1, 1:-1]

    # Step 3: Generate all 26 neighbor offsets
    # This creates a 3x3x3 grid of offsets, then filters out the (0, 0, 0) center
    dx, dy, dz = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
    offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    offsets = offsets[np.any(offsets != 0, axis=1)]  # Remove the center offset

    # Step 4: Collect all 26 neighbor slices using the offsets
    # Each neighbor is a shifted view of the padded array
    neighbors = np.stack(
        [
            padded[
                1 + ox : 1 + ox + img.shape[0],
                1 + oy : 1 + oy + img.shape[1],
                1 + oz : 1 + oz + img.shape[2],
            ]
            for ox, oy, oz in offsets
        ],
        axis=0,
    )  # Shape: (26, X, Y, Z)

    # Step 5: Compare the center voxel to all 26 neighbors
    # A voxel is a local minimum if it's smaller than all neighbors
    is_min = np.all(center < neighbors, axis=0)  # Shape: (X, Y, Z)

    # Step 6: Label each minimum voxel with a unique integer
    labels = np.zeros_like(img, dtype=np.int32)
    coords = np.argwhere(is_min)
    for i, (z, y, x) in enumerate(coords, start=1):
        labels[z, y, x] = i

    return labels
