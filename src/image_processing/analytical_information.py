#!/usr/bin/env python3
"""Extract analytical information.

Script Name:    distance_transform.py
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


def compute_volume_and_com(labels, ignore_background=True):
    """Compute volume and center of mass for labeled regions using NumPy.

    Compute the voxel-based volume and center of mass (COM) for each label
    in a 3D segmented NumPy array using a pure NumPy approach.

    This function calculates:
        - Volume as the number of voxels per label.
        - Center of mass in voxel index coordinates (z, y, x).

    It uses NumPy's array indexing and masking to efficiently compute these
    metrics.

    Parameters:
        labels (np.ndarray): 3D integer array of shape (Z, Y, X) representing
                             segmented regions. Background is typically 0.
        ignore_background (bool): If True, skip label 0 in the output.

    Returns:
        dict: A dictionary mapping each label to its metrics:
              {
                  label: {
                      "voxel_count": int,
                      "com_index": (float, float, float)
                  }
              }

    """
    # Extract labels
    unique_labels = np.unique(labels)
    if ignore_background:
        unique_labels = unique_labels[unique_labels != 0]

    # Index grids
    z_idx, y_idx, x_idx = np.indices(labels.shape)

    volumes = []
    coms = []

    for lab in unique_labels:
        mask = labels == lab
        voxel_count = int(mask.sum())

        if voxel_count == 0:
            volumes.append(0)
            coms.append([np.nan, np.nan, np.nan])
            continue

        # Float COM
        com_z = z_idx[mask].sum() / voxel_count
        com_y = y_idx[mask].sum() / voxel_count
        com_x = x_idx[mask].sum() / voxel_count

        volumes.append(voxel_count)
        coms.append([com_z, com_y, com_x])

    return unique_labels, np.array(volumes, dtype=int), np.array(coms, dtype=float)