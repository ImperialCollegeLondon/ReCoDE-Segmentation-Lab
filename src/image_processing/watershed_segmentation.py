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
            neig_values = labels[nx, ny, nz]

            # Check if any of the neighbors are not false,
            # i.e. already part of a labeled region
            if np.all(neig_values == 0):
                # This must be a new local minima
                # Check label from markers
                labels_step[x, y, z] = marker[x, y, z]
            else:
                # Has true neighbors at previous level
                # Reduce to only true neighbores
                neig_true = neig_values[neig_values != 0]

                if neig_true.size == 1:
                    # Only one labelled neighbor (must be part of the region)
                    labels_step[x, y, z] = neig_true
                elif np.all(neig_true == neig_true[0]):
                    # Multiple labelled neighbors, but of the same labelled region
                    # (must be same label region)
                    # This can in theory be replaced, else would result in the same.
                    labels_step[x, y, z] = neig_true[0]
                else:
                    # Multiple labelled neighbors, of different regions
                    # Voxel is assigned to region that has more bordering voxels
                    _values, counts = np.unique(neig_true, return_counts=True)
                    labels_step[x, y, z] = neig_true[np.argmax(counts)]

        # Update labels and store in output array
        labels = labels_step
        # Output array hold the progressing/flooding steps
        # Needs to be removed for larger images
        labels_levels[dl_i] = labels_step

    return labels_levels
