#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
===========================================================
 Script Name:    load_dataset.py
 Author:         David Buchner, Imperial College London
 Created:        08/10/2025
 Last Modified:  08/10/2025
 Description:
    Functions to load imaging datasets and extract respective metadata.

 Requirements:
    - Python 3.x
    - Required libraries:
        * numpy
        * imageio

 Notes:

===========================================================
'''
# -----------------------------
# Load Python packages
import os
import numpy as np
import imageio.v3 as iio

# -----------------------------
def load_tif_sequence(directory, prefix="Small particles_rec", start=None, end=None):
    """
    Load a sequence of TIFF files from a directory into a 3D NumPy array.

    Parameters:
        directory (str): Path to the folder containing TIFF files.
        prefix (str): Common prefix of the TIFF filenames.
        start (int, optional): Starting index. If None, inferred from the files.
        end (int, optional): Ending index. If None, inferred from the files.

    Returns:
        np.ndarray: 3D array of shape (num_files, height, width) containing the loaded images.

    # Example usage:
    stack = load_tif_sequence("../imaging_data/spherical_particles",start=110,end=160)

    """
    directory = os.path.abspath(directory)
    all_files = os.listdir(directory)

    # Find all files matching the prefix and ending with .tif
    matched_files = []
    for f in all_files:
        if f.startswith(prefix) and f.endswith(".tif"):
            num_part = f[len(prefix):-4]
            if num_part.isdigit():
                matched_files.append((f, int(num_part)))

    if not matched_files:
        print("No matching TIFF files found.")
        return np.array([])

    # Sort files by numeric index
    matched_files.sort(key=lambda x: x[1])
    indices = [idx for _, idx in matched_files]

    # Determine start and end indices
    min_idx, max_idx = indices[0], indices[-1]
    start_idx = start if start is not None else min_idx
    end_idx = end if end is not None else max_idx

    # Warn if requested indices are outside available range and clip them
    if start_idx < min_idx:
        print(
            f"Warning: start={start_idx} is smaller than the minimum available index {min_idx}. Clipping to {min_idx}.")
        start_idx = min_idx
    if end_idx > max_idx:
        print(f"Warning: end={end_idx} is larger than the maximum available index {max_idx}. Clipping to {max_idx}.")
        end_idx = max_idx
    if start_idx > max_idx or end_idx < min_idx:
        print("Warning: requested range is outside available file indices. No images to load.")
        return np.array([])

    # Filter files to load within start/end
    files_to_load = [f for f, idx in matched_files if start_idx <= idx <= end_idx]

    images = []
    for filename in files_to_load:
        filepath = os.path.join(directory, filename)
        try:
            img = iio.imread(filepath)
            images.append(img)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if images:
        print(f"Successfully loaded {len(images)} images from '{directory}'")
    else:
        print("No images were loaded!")

    return np.array(images)
