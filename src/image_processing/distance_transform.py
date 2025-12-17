#!/usr/bin/env python3
"""Distance transform functions.

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
from numpy.lib.recfunctions import unstructured_to_structured


# -----------------------------
# Main and most recent Chamfer distance transform implementation
def chamfer_distance_transform(img):
    """Compute the chamfer distance transform using Numpy sweeps.

    Compute the chamfer distance transform of a 3D binary NumPy array
    using an optimized NumPy approach.

    This function approximates the Euclidean distance transform using a chamfer mask,
    which propagates integer-valued distances through local neighborhoods. It is
    suitable for binary 3D arrays where foreground voxels are non-zero and background
    voxels are zero.

    This NumPy-only chamfer distance transform uses np.roll to simulate neighbor
    propagation efficiently. While not part of the classical Borgefors
    implementation, this technique is inspired by stencil-based methods and
    array-shifting strategies commonly used in high-performance computing
    and image processing. See Sling Academy for np.roll examples.

    Parameters:
        img (np.ndarray): 3D binary array of shape (X, Y, Z) with 0 for background
                          and non-zero for foreground.

    Returns:
        np.ndarray: 3D array of same shape as `img`, where each voxel contains an
                    integer-valued approximation of the Euclidean distance to the
                    nearest foreground voxel.

    ------------------------------------------------------------------------
    Example Usage:
        distance_map = chamfer_distance_3d_optimized(binary_volume)

    ------------------------------------------------------------------------

    References:
    - Borgefors, G. (1986). "Distance transformations in digital images."
      Computer Vision, Graphics, and Image Processing, 34(3), 344-371.
    - Wikipedia: https://en.wikipedia.org/wiki/Distance_transform#Chamfer_distance_transform
    - Stack Overflow: https://stackoverflow.com/questions/53678520/speed-up-computation-for-distance-transform-on-image-in-python
    - https://www.slingacademy.com/article/understanding-numpy-roll-function-6-examples/
    """
    # Define chamfer mask: each tuple is (dx, dy, dz, weight)
    # These represent relative neighbor positions and their associated movement cost.
    # Includes both positive and negative directions to ensure symmetric propagation.
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)

    offsets = [
        (1, 0, 0, 1.0),
        (-1, 0, 0, 1.0),
        (0, 1, 0, 1.0),
        (0, -1, 0, 1.0),
        (0, 0, 1, 1.0),
        (0, 0, -1, 1.0),
        (1, 1, 0, sqrt2),
        (-1, -1, 0, sqrt2),
        (1, 0, 1, sqrt2),
        (-1, 0, -1, sqrt2),
        (0, 1, 1, sqrt2),
        (0, -1, -1, sqrt2),
        (1, 1, 1, sqrt3),
        (-1, -1, -1, sqrt3),
    ]

    # Pad the input volume with a 1-voxel border to simplify boundary handling.
    # Padding ensures that neighbor access won't go out of bounds.
    padded = np.pad(img != 0, pad_width=1, mode="constant", constant_values=0)

    # Initialize the distance map:
    # Foreground voxels (non-zero in original image) get distance 0.
    # Background voxels (zero in original image) get a large initial value.
    dt = np.where(padded, 65535, 0).astype(np.uint32)

    # Iterative sweeping: repeat the neighbor propagation multiple times.
    # Each sweep allows distances to propagate further through the volume.
    max_iter = max(img.shape)  # Dynamic upper bound based on volume shape
    for i in range(max_iter):
        prev_dt = dt.copy()

        for dx, dy, dz, w in offsets:
            # Shift the entire distance map by (dx, dy, dz) to
            # simulate neighbor access.
            # This gives the neighbor values for every voxel in
            # one operation.
            shifted = np.roll(dt, shift=(dx, dy, dz), axis=(0, 1, 2))
            # Update each voxel with the minimum of its current value
            # and the neighbor's value + weight.
            # This is the core of the chamfer propagation: finding
            # shorter paths via neighbors.
            dt = np.minimum(dt, shifted + w)

        # Check for convergence: if no values changed, break early
        if np.array_equal(dt, prev_dt):
            print(f"Chamfer converged after {i + 1} iterations. ")
            break
    else:
        print(f"Reached max_iter={max_iter} without full convergence.")

    # Remove the padding to return a result of the original shape.
    return dt[1:-1, 1:-1, 1:-1]
