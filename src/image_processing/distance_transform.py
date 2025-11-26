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
    """Compute the chamfer distance transform with argwhere.

    Chamfer distance transform using argwhere..
    Performs forward and backward sweeps with integer-weighted neighbors.
    Includes timing output for performance analysis.

    Parameters:
        img (np.ndarray): 3D binary array (non-zero = foreground, 0 = background)

    Returns:
        np.ndarray: Chamfer distance map (same shape as input)
    """

    # Define 26-neighbor mask for sweeps:
    # Forward uses neighbors with negative deltas; backward uses positive deltas.

    neighbours = []
    weights = []

    # Faces
    for dx, dy, dz in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        neighbours.append([dx, dy, dz])
        weights.append(1.0)

    # Edges
    for dx, dy, dz in [
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (-1, 0, 1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (0, -1, 1),
        (0, -1, -1),
    ]:
        neighbours.append([dx, dy, dz])
        weights.append(np.sqrt(2))

    # Corners
    for dx, dy, dz in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                       (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]:
        neighbours.append([dx, dy, dz])
        weights.append(np.sqrt(3))

    neighbours = np.array(neighbours)
    weights = np.array(weights)


    sx, sy, sz = img.shape


    # *** IMPORTANT: float + np.inf initialization ***
    # Distance FROM BACKGROUND: background = 0 (sources), foreground = inf (to be updated)
    dt = np.where(img == 0, 0.0, np.inf).astype(float)

    # Sweep all voxels in lexicographic order (x, then y, then z)
    coords = np.argwhere(np.ones_like(img, dtype=bool))
    coords = coords[np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))]

    # Forward sweep (propagate from already-updated neighbors with negative deltas)
    for x, y, z in coords:
        # Skip background source voxels (optional optimization)
        # if img[x, y, z] == 0: continue
        for (dx, dy, dz), w in zip(neighbours, weights):
            nx, ny, nz = x - dx, y - dy, z - dz
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                cand = dt[nx, ny, nz] + w
                if cand < dt[x, y, z]:
                    dt[x, y, z] = cand

    # Backward sweep (reverse order, positive deltas neighbors)
    for x, y, z in coords[::-1]:
        # if img[x, y, z] == 0: continue
        for (dx, dy, dz), w in zip(neighbours, weights):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                cand = dt[nx, ny, nz] + w
                if cand < dt[x, y, z]:
                    dt[x, y, z] = cand

    return dt



# -----------------------------
# Alternative, non optimized Chamfer Distance transform implementations
def chamfer_distance_3d(img):
    """Compute the chamfer distance transform.

    Compute the chamfer distance transform of a 3D binary NumPy array.

    This function approximates the Euclidean distance transform using a chamfer mask,
    which propagates integer-valued distances through local neighborhoods. It is
    suitable for binary 3D arrays where foreground voxels are non-zero and background
    voxels are zero.

    Mathematical background:
    The chamfer method uses a fixed set of weighted offsets to simulate Euclidean
    distances through iterative forward and backward passes. Each voxel updates its
    value based on neighboring voxels and predefined weights, approximating the
    shortest path to the nearest foreground voxel.

    This implementation uses a 3x3x3 mask with integer weights:
        - 3 for axis-aligned neighbors
        - 4 for face-diagonal neighbors
        - 5 for corner-diagonal neighbors

    The algorithm consists of two phases:
        - Forward pass: propagates distances from top-left-front to
          bottom-right-back
        - Backward pass: refines distances from bottom-right-back to
          top-left-front

    This method is significantly faster than exact Euclidean transforms and is well-
    suited for large volumes where performance is critical.

    References:
    - Borgefors, G. (1986). "Distance transformations in digital images."
      Computer Vision, Graphics, and Image Processing, 34(3), 344-371.
    - Wikipedia overview: https://en.wikipedia.org/wiki/Distance_transform#Chamfer_distance_transform
    - Stack Overflow discussion on NumPy-only distance transform performance:
      https://stackoverflow.com/questions/53678520/speed-up-computation-for-distance-transform-on-image-in-python

    Parameters:
        img (np.ndarray): 3D binary array of shape (X, Y, Z) with 0 for background
                          and non-zero for foreground.

    Returns:
        np.ndarray: 3D array of same shape with chamfer distances (integer
                    approximation of Euclidean distance).

    Example usage:
        distance_map = chamfer_distance_3d(binary_volume)
    """
    # Define chamfer weights (3x3x3 neighborhood)
    weights = [
        (1, 0, 0, 3),
        (0, 1, 0, 3),
        (0, 0, 1, 3),
        (1, 1, 0, 4),
        (1, 0, 1, 4),
        (0, 1, 1, 4),
        (1, 1, 1, 5),
    ]

    shape = img.shape

    # Initialize distance map: background gets large value, foreground gets
    dt = np.where(img == 0, np.max(img.shape), 0).astype(np.uint32)

    # Forward pass: iterate from top-left-front to bottom-right-back
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if img[x, y, z] == 0:
                    for dx, dy, dz, w in weights:
                        nx, ny, nz = x - dx, y - dy, z - dz
                        if (
                            0 <= nx < shape[0]
                            and 0 <= ny < shape[1]
                            and 0 <= nz < shape[2]
                        ):
                            dt[x, y, z] = min(dt[x, y, z], dt[nx, ny, nz] + w)

    # Backward pass: iterate from bottom-right-back to top-left-front
    for z in reversed(range(shape[2])):
        for y in reversed(range(shape[1])):
            for x in reversed(range(shape[0])):
                if img[x, y, z] == 0:
                    for dx, dy, dz, w in weights:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (
                            0 <= nx < shape[0]
                            and 0 <= ny < shape[1]
                            and 0 <= nz < shape[2]
                        ):
                            dt[x, y, z] = min(dt[x, y, z], dt[nx, ny, nz] + w)

    return dt


def chamfer_distance_3d_structured(img):
    """Compute the chamfer distance transform with structured arrays.

    Chamfer distance transform using structured arrays and sorted voxel iteration.
    Performs single sweep (forward and backward) with integer-weighted neighbors,
    image padded to avoid if statement.

    Parameters:
        img (np.ndarray): 3D binary array (non-zero = foreground, 0 = background)

    Returns:
        np.ndarray: Chamfer distance map (same shape as input)
    """
    # Define symmetric chamfer mask: neighbor offsets and weights
    offsets = [
        (1, 0, 0, 3),
        (-1, 0, 0, 3),
        (0, 1, 0, 3),
        (0, -1, 0, 3),
        (0, 0, 1, 3),
        (0, 0, -1, 3),
        (1, 1, 0, 4),
        (-1, -1, 0, 4),
        (1, 0, 1, 4),
        (-1, 0, -1, 4),
        (0, 1, 1, 4),
        (0, -1, -1, 4),
        (1, 1, 1, 5),
        (-1, -1, -1, 5),
    ]

    # Pad the input volume with a 1-voxel border
    padded = np.pad(img == 0, pad_width=1, mode="constant", constant_values=0)

    # Initialize distance map: foreground = 0, background = large value
    dt = np.where(padded, 65535, 0).astype(np.uint32)

    # Get indices of background voxels in original image space
    indices = np.argwhere(img == 0)
    indices += 1  # shift to match padded coordinates

    # Convert to structured array for sorting
    dtype = [("x", int), ("y", int), ("z", int)]
    structured_indices = unstructured_to_structured(indices, dtype=dtype)
    structured_indices = np.sort(structured_indices, order=["z", "y", "x"])

    # Combined sweep without boundary checks
    for idx in structured_indices:
        x, y, z = idx["x"], idx["y"], idx["z"]
        for dx, dy, dz, w in offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            dt[z, y, x] = min(dt[z, y, x], dt[nz, ny, nx] + w)

    return dt[1:-1, 1:-1, 1:-1]

def chamfer_distance_nroll(img):
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

