#!/usr/bin/env python3
"""watershed segmentation algorithm functions.

Script Name:    watershed_segmentation.py
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


def watershed_3d(distance, binary, marker):
    """Watershed algorithm.

    This function executes a simple version of the Meyer's flooding algorithm.

    This function implements a voxel-wise 3D watershed on a distance-transformed
    image. It propagates labels from a marker array to segment connected regions
    within a binary mask based on distance levels. It is roughly based on
    Meyer's flooding algorithm.

    Parameters:
        distance (np.ndarray): 3D array representing the distance transform
            of the image. Higher values indicate voxels farther from the background.
        binary (np.ndarray): 3D boolean array defining the region to segment.
        marker (np.ndarray): 3D array of initial seed labels for local minima.

    Returns:
        np.ndarray: 4D array of shape (num_levels, X, Y, Z) containing the
        watershed labels at each distance level. The last element along the
        first axis corresponds to the fully flooded label map.

    Notes:
        - The function may be slow for large 3D images due to voxel-wise iteration.

    References:
        - https://en.wikipedia.org/wiki/Watershed_(image_processing)

    """
    # Get and sort distance levels
    distance_levels = np.unique(distance)[::-1]

    # Initialize arrays
    labels = np.zeros_like(distance)
    labels_levels = np.zeros((len(distance_levels), *distance.shape), dtype=int)

    for dl_i in range(len(distance_levels) - 1):
        labels_step = labels.copy()

        # Create mask of new distance level (only new level is true)
        dl = np.logical_and(distance == distance_levels[dl_i], binary)
        # Extract coordinates of every voxel which is part of the new level
        dl_coords = np.argwhere(dl)
        shape = np.array(dl.shape)

        # 26-connected neighbor offsets
        offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(
            -1, 3
        )
        # Remove the center voxel itself
        neighbors = offsets[np.any(offsets != 0, axis=1)]

        for idx, (x, y, z) in enumerate(dl_coords):
            # Compute neighbor coordinates of voxel
            neigh = neigh = neighbors + np.array([x, y, z])

            # Check if all neighbors are inside the image boundaries
            valid = np.all((neigh >= 0) & (neigh < shape), axis=1)
            neigh = neigh[valid]

            # unpack into separate arrays for indexing
            nx, ny, nz = neigh.T

            # Compare against labeled array (only previous level at this step)
            neigh_vals = labels[nx, ny, nz]

            # Check if any of the neighbors are not false,
            # i.e. already part of a labeled region
            if np.all(neigh_vals == 0):
                labels_step[x, y, z] = marker[x, y, z]
            else:
                neigh_true = neigh_vals[neigh_vals != 0]
                if neigh_true.size == 1:
                    labels_step[x, y, z] = neigh_true[0]
                elif np.all(neigh_true == neigh_true[0]):
                    labels_step[x, y, z] = neigh_true[0]
                else:
                    votes = {}
                    for (dx, dy, dz), lbl in zip(
                        neigh - np.array([x, y, z]), neigh_vals
                    ):
                        if lbl == 0:
                            continue
                        nzc = np.count_nonzero([dx, dy, dz])
                        w = 3 if nzc == 1 else (2 if nzc == 2 else 1)
                        votes[lbl] = votes.get(lbl, 0) + w
                    labels_step[x, y, z] = max(votes, key=votes.get)

        # Update labels and store in output array
        labels = labels_step
        # Output array hold the progressing/flooding steps
        # Needs to be removed for larger images
        labels_levels[dl_i] = labels_step

    return labels_levels
